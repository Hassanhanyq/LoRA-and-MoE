import arxiv
import time
import json
from pathlib import Path

SAVE_DIR = Path("data")
SAVE_DIR.mkdir(exist_ok=True)

CATEGORY = "cs.LG"
QUERY = "(information bottleneck OR minimum description length OR compression bounds OR PAC-Bayes theory OR flat minima OR sharpness-aware minimization OR lottery ticket hypothesis OR neural collapse OR double descent OR dynamical isometry OR scaling laws OR signal propagation OR neural scaling hypothesis OR margin theory OR Fisher-Rao norm OR NTK linear regime OR sparsity-inducing priors)"
MAX_RESULTS = 50
MIN_YEAR = 2022

def main():
    client = arxiv.Client(
        page_size=500,
        delay_seconds=3,
        num_retries=10
    )

    search = arxiv.Search(
        query=f"cat:{CATEGORY} AND {QUERY}",
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    results_generator = client.results(search)
    save_path = SAVE_DIR / "paperslg3.json"

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
