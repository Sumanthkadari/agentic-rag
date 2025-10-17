from ingestion.scrapers import fetch_page, extract_links, page_to_doc

# List of websites to scrape
websites = [
    "https://lucidmotors.com/air",
    "https://www.wellsfargo.com/help/"
]

scraped_docs = []

for url in websites:
    try:
        # Fetch the page HTML
        html = fetch_page(url)

        # Convert page to a document
        doc = page_to_doc(html, url)
        scraped_docs.append(doc)

        # Optionally, extract internal links
        links = extract_links(html, url)
        print(f"Found {len(links)} internal links on {url}")

    except Exception as e:
        print(f"Error scraping {url}: {e}")

# Print scraped content for both sites
for doc in scraped_docs:
    print("\n---")
    print(f"URL: {doc['url']}")
    print(f"Title: {doc['title']}")
    print(f"Text snippet: {doc['text'][:300]}...")  