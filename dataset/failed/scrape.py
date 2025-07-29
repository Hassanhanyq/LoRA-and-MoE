import arxiv
import time
from pathlib import Path
import fitz

SAVE_DIR = Path("data")
PDF_DIR = Path("pdfs")
MAX_RESULTS = 100
CATEGORY = "cs.CL"
QUERY = "large language models"
MIN_YEAR = 2022

SAVE_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)

def extract_text_without_intro(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        lowered = full_text.lower()
        for marker in ["1 introduction", "introduction", "i. introduction"]:
            if marker in lowered:
                idx = lowered.find(marker)
                return full_text[idx + len(marker):].strip()

        
        return full_text.strip()

    except Exception as e:
        print(f"Failed to extract from {pdf_path.name}: {e}")
        return None
def main():
    client = arxiv.Client(
        page_size=25,
        delay_seconds=3,
        num_retries=3
    )

    search = arxiv.Search(
        query=f"cat:{CATEGORY} AND {QUERY}",
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    results_generator = client.results(search)

    for result in results_generator:
        paper_id = result.get_short_id()
        year = result.published.year
        if year < MIN_YEAR:
            print(f"[{paper_id}] Skipping old paper ({year})")
            continue

        pdf_path = PDF_DIR / f"{paper_id}.pdf"
        if not pdf_path.exists():
            try:
                result.download_pdf(dirpath=PDF_DIR)
                print(f"[{paper_id}] Downloaded.")
                time.sleep(1)
            except Exception as e:
                print(f"[{paper_id}] Failed to download: {e}")
                continue

        body_text = extract_text_without_intro(pdf_path)
        if not body_text:
         print(f"[{paper_id}] No body text extracted.")
         continue

    with open(PDF_DIR / f"{paper_id}.txt", "w", encoding="utf-8") as f:
        f.write(body_text)
        print(f"[{paper_id}] Saved body text.")


if __name__ == "__main__":
    main()
