
from typing import TypedDict, List, Optional, Literal
from langgraph.graph import StateGraph, END

from src.agents.retriever_agent import RetrieverAgent
from src.agents.synthesizer import synthesize_answer
from src.agents.validator import validate
from src.agents.guardrails import redact_pii, safety_filter
from src.agents.planner import decompose_query
from src.agents.router import route_intent

print("[BOOT] Orchestrator loaded from:", __file__)


class AgentState(TypedDict, total=False):
    """Typed state for LangGraph nodes."""
    query: str
    user_id: Optional[str]
    max_docs: int
    intent: str
    sub_queries: List[str]
    current_step: int
    docs: List[str]
    answer: str
    citations: List[str]
    assets: List[str]
    follow_ups: List[str]
    should_stop: bool


class Orchestrator:
    """Agentic RAG Orchestrator with proper state management."""

    def __init__(self, retriever_agent: RetrieverAgent):
        self.retriever_agent = retriever_agent
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)

        # 1. Guardrails 
        def guardrail_node(state: AgentState) -> AgentState:
            print("[NODE] guardrails")
            query = state.get("query", "")
            print(f"[DEBUG] Guardrail received query: '{query}' (type: {type(query)})")
            
            if not query or not isinstance(query, str) or not query.strip():
                print(f"[WARN] Invalid query in guardrails: {repr(query)}")
                return {
                    **state,
                    "answer": "Missing or invalid query.",
                    "citations": [],
                    "follow_ups": [],
                    "should_stop": True
                }

            # Redact PII
            redacted = redact_pii(query)
            if not redacted or not redacted.strip():
                print("[WARN] Redaction removed all content, using original")
                redacted = query

            # Safety filter
            if not safety_filter(redacted):
                print("[WARN] Query blocked by safety filter")
                return {
                    **state,
                    "query": redacted,
                    "answer": "Query blocked due to unsafe content.",
                    "citations": [],
                    "follow_ups": [],
                    "should_stop": True
                }

            print(f"[DEBUG] Guardrails passed, query: '{redacted}'")
            return {
                **state,
                "query": redacted,
                "should_stop": False
            }

        # Conditional routing after guardrails
        def should_continue_after_guardrails(state: AgentState) -> Literal["router", "output"]:
            if state.get("should_stop", False):
                print("[DEBUG] Stopping after guardrails")
                return "output"
            return "router"
        
        #2. Router 
        def router_node(state: AgentState) -> AgentState:
            print("[NODE] router")
            query = state.get("query", "")
            intent = route_intent(query)
            print(f"[DEBUG] Routed intent: {intent}")
            return {**state, "intent": intent}

        #3. Planner
        def planner_node(state: AgentState) -> AgentState:
            print("[NODE] planner")
            query = state.get("query", "")
            intent = state.get("intent", "faq")
            
            if not query.strip():
                print("[WARN] Empty query in planner")
                return {**state, "sub_queries": [], "should_stop": True}
            
            if intent == "multi-step":
                sub_queries = decompose_query(query)
                print(f"[DEBUG] Multi-step decomposed into {len(sub_queries)} sub-queries")
            else:
                sub_queries = [query]
            
            print(f"[DEBUG] Sub-queries: {sub_queries}")
            return {**state, "sub_queries": sub_queries}

        # Conditional routing after planner
        def should_continue_after_planner(state: AgentState) -> Literal["retriever", "output"]:
            if state.get("should_stop", False) or not state.get("sub_queries", []):
                print("[DEBUG] Stopping after planner")
                return "output"
            return "retriever"

        #4. Retriever
        def retriever_node(state: AgentState) -> AgentState:
            print("[NODE] retriever")
            sub_queries = state.get("sub_queries", [])
            current_step = state.get("current_step", 0)
            max_docs = state.get("max_docs", 5)

            if not sub_queries:
                print("[WARN] No sub-queries to retrieve")
                return {**state, "docs": [], "should_stop": True}

            if current_step >= len(sub_queries):
                print("[WARN] Current step out of range")
                return {**state, "should_stop": True}

            sub_query = sub_queries[current_step]
            if not sub_query.strip():
                print("[WARN] Empty sub-query at current step")
                return {**state, "docs": [], "should_stop": True}

            print(f"[DEBUG] Retrieving for: '{sub_query}'")
            
            try:
                docs = self.retriever_agent.run(sub_query, top_k=max_docs, mode="hybrid")
                print(f"[DEBUG] Retrieved {len(docs)} documents")
                
                # Append to existing docs
                existing_docs = state.get("docs", [])
                all_docs = existing_docs + docs
                
                return {**state, "docs": all_docs}
            except Exception as e:
                print(f"[ERROR] Retrieval failed: {e}")
                return {**state, "docs": [], "should_stop": True}

        # Conditional routing after retriever
        def should_continue_after_retriever(state: AgentState) -> Literal["synthesizer", "output"]:
            if state.get("should_stop", False) or not state.get("docs", []):
                print("[DEBUG] Stopping after retriever")
                return "output"
            return "synthesizer"

        #5. Synthesizer
        def synthesizer_node(state: AgentState) -> AgentState:
            print("[NODE] synthesizer")
            query = state.get("query", "")
            docs = state.get("docs", [])

            if not docs:
                print("[WARN] No documents to synthesize")
                return {
                    **state,
                    "answer": "No relevant documents found in the knowledge base.",
                    "citations": [],
                    "follow_ups": []
                }

            print(f"[DEBUG] Synthesizing answer from {len(docs)} documents")
            
            try:
                result = synthesize_answer(query, docs)
                if result and result.get("answer"):
                    print(f"[DEBUG] Answer generated: {result['answer'][:100]}...")
                    return {
                        **state,
                        "answer": result.get("answer", ""),
                        "citations": result.get("citations", []),
                        "assets": result.get("assets", []),
                        "follow_ups": result.get("follow_ups", [])
                    }
                else:
                    print("[WARN] Synthesizer returned empty result")
                    return {
                        **state,
                        "answer": "Unable to generate answer from retrieved documents.",
                        "citations": [],
                        "follow_ups": []
                    }
            except Exception as e:
                print(f"[ERROR] Synthesis failed: {e}")
                import traceback
                traceback.print_exc()
                return {
                    **state,
                    "answer": "Error generating answer.",
                    "citations": [],
                    "follow_ups": []
                }

        #6. Validator
        def validator_node(state: AgentState) -> AgentState:
            print("[NODE] validator")
            answer = state.get("answer", "")
            citations = state.get("citations", [])
            
            if not answer:
                print("[WARN] No answer to validate")
                return state
            
            if not validate(answer, citations):
                print("[WARN] Validation failed")
                return {
                    **state,
                    "answer": "Validation failed. Please refer to official documentation.",
                    "citations": []
                }
            
            print("[DEBUG] Validation passed")
            return state

        #Output 
        def output_node(state: AgentState) -> AgentState:
            print("[NODE] output")
            
            # Construct final output
            output = {
                "query": state.get("query", ""),
                "answer": state.get("answer", "Unable to process query."),
                "citations": state.get("citations", []),
                "assets": state.get("assets", []),
                "follow_ups": state.get("follow_ups", [])
            }
            
            print(f"[DEBUG] Final output prepared: {output}")
            return {**state, "output": output}

        # Build Graph 
        graph.add_node("guardrails", guardrail_node)
        graph.add_node("router", router_node)
        graph.add_node("planner", planner_node)
        graph.add_node("retriever", retriever_node)
        graph.add_node("synthesizer", synthesizer_node)
        graph.add_node("validator", validator_node)
        graph.add_node("output", output_node)

        # Set entry point
        graph.set_entry_point("guardrails")
        
        # Add conditional edges
        graph.add_conditional_edges(
            "guardrails",
            should_continue_after_guardrails,
            {"router": "router", "output": "output"}
        )
        
        graph.add_edge("router", "planner")
        
        graph.add_conditional_edges(
            "planner",
            should_continue_after_planner,
            {"retriever": "retriever", "output": "output"}
        )
        
        graph.add_conditional_edges(
            "retriever",
            should_continue_after_retriever,
            {"synthesizer": "synthesizer", "output": "output"}
        )
        
        graph.add_edge("synthesizer", "validator")
        graph.add_edge("validator", "output")
        graph.add_edge("output", END)

        return graph.compile()

    def run(self, query: str, user_id=None, max_docs: int = 5):
        """Execute the LangGraph pipeline with full error handling."""
        
        if not query or not isinstance(query, str) or not query.strip():
            print(f"[ERROR] Invalid query: {repr(query)}")
            return {
                "query": str(query) if query else "",
                "answer": "Query cannot be empty.",
                "citations": [],
                "assets": [],
                "follow_ups": []
            }

        print(f"\n[ORCHESTRATOR] Running query: '{query}'")
        
        # Initialize state with all required fields
        initial_state: AgentState = {
            "query": query.strip(),
            "user_id": user_id,
            "max_docs": max_docs,
            "current_step": 0,
            "should_stop": False,
            "docs": [],
            "citations": [],
            "assets": [],
            "follow_ups": []
        }
        
        print(f"[DEBUG] Initial state created with query: '{initial_state['query']}'")

        try:
            result_state = self.graph.invoke(initial_state)
            print(f"[DEBUG] Graph execution completed")
            
            if not result_state:
                print("[ERROR] Graph returned None")
                return {
                    "query": query,
                    "answer": "Graph execution returned no state.",
                    "citations": [],
                    "assets": [],
                    "follow_ups": []
                }
            
            print(f"[DEBUG] Result state keys: {list(result_state.keys())}")
            
            # Extract output
            if "output" in result_state:
                return result_state["output"]
            
            # Construct output from state
            return {
                "query": result_state.get("query", query),
                "answer": result_state.get("answer", "No answer generated."),
                "citations": result_state.get("citations", []),
                "assets": result_state.get("assets", []),
                "follow_ups": result_state.get("follow_ups", [])
            }
            
        except Exception as e:
            print(f"[ERROR] LangGraph execution failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "query": query,
                "answer": "Internal error during execution.",
                "citations": [],
                "assets": [],
                "follow_ups": []
            }


if __name__ == "__main__":
    from src.utils.ingestion.retriever import HybridRetriever

    retriever = HybridRetriever()
    retriever_agent = RetrieverAgent(retriever)
    orchestrator = Orchestrator(retriever_agent)

    # Test queries
    test_queries = [
        "What is Lucid Air?",
        "How to reset infotainment system?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        result = orchestrator.run(query)
        print("\n--- Agentic RAG Output ---")
        print(f"Answer: {result['answer']}")
        print(f"Citations: {result['citations']}")



