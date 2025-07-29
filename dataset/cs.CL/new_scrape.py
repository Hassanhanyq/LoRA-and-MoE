import arxiv
import time
import json
from pathlib import Path

SAVE_DIR = Path("data")
SAVE_DIR.mkdir(exist_ok=True)

CATEGORY = "cs.CL"
QUERY = "(large language models OR LLM OR transformer OR instruction tuning OR language modeling OR Fine-Tuned OR Multimodal)"
MAX_RESULTS = 300
MIN_YEAR = 2022

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
    save_path = SAVE_DIR / "papers.json"

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("[\n")

        first = True
        count = 0

        for result in results_generator:
            paper_id = result.get_short_id()
            year = result.published.year

            if year < MIN_YEAR:
                print(f"[{paper_id}] Skipping old paper ({year})")
                continue

            title = result.title.strip().replace("\n", " ")
            abstract = result.summary.strip().replace("\n", " ")

            paper_data = {
                "title": title,
                "abstract": abstract
            }

            if not first:
                f.write(",\n")
            json.dump(paper_data, f, ensure_ascii=False)
            first = False

            print(f"[{paper_id}] Collected: {title[:60]}...")
            count += 1
            time.sleep(1)

        f.write("\n]\n")

    print(f"Saved {count} papers to {save_path}")

if __name__ == "__main__":
    main()
