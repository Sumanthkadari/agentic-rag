
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_pdfs(pdf_files, chunk_size=1000, chunk_overlap=200):
    all_text = ""
    for pdf in pdf_files:
        all_text += load_pdf(pdf)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_text(all_text)
    return chunks

if __name__ == "__main__":
    pdf_files = ["data\lucid_owners_manual.pdf", "data\wells_fargo.pdf"]  
    chunks = chunk_pdfs(pdf_files)
    print(f"Total chunks created: {len(chunks)}")
    
    # save chunks to a file for later use
    with open("chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n---\n")
