import arxiv
import time
import json
from pathlib import Path

SAVE_DIR = Path("data")
SAVE_DIR.mkdir(exist_ok=True)

CATEGORY = "cs.LG"
QUERY = "(gradient descent OR stochastic gradient descent OR adaptive optimization OR convex optimization OR non-convex optimization OR online optimization OR distributed optimization OR federated learning OR meta-learning OR transfer learning OR representation learning OR deep learning architectures  OR graph neural networks OR neural collapse OR contrastive learning OR reinforcement learning theory OR generalization theory OR overparameterization OR regularization techniques OR convergence analysis OR optimization landscapes OR saddle points OR variance reduction OR proximal methods OR coordinate descent OR second-order methods OR implicit bias OR neural tangent kernel OR spectral methods OR sparsity OR robust optimization OR adversarial training OR curriculum learning OR batch normalization theory OR attention mechanisms OR multi-task learning OR few-shot learning OR neural architecture search  OR explainability OR interpretability OR Bayesian optimization OR variational inference OR causal inference OR stochastic processes OR theoretical guarantees OR sample complexity OR algorithmic stability OR PAC learning OR VC dimension OR Rademacher complexity OR kernel methods OR smoothness assumptions OR Lipschitz continuity OR gradient clipping OR normalization methods OR energy-based models  OR implicit models OR implicit differentiation OR fixed point theory OR convex conjugate OR primal-dual methods)"
MAX_RESULTS = 300
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
    save_path = SAVE_DIR / "papersro.json"

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
