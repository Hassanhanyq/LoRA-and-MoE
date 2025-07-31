import arxiv
import time
import json
from pathlib import Path

SAVE_DIR = Path("data")
SAVE_DIR.mkdir(exist_ok=True)

CATEGORY = "cs.RO"
QUERY = "(robotics OR sensor fusion OR robot learning OR robot manipulation OR grasping OR autonomous navigation OR motion planning OR trajectory optimization OR visual servoing OR impedance control OR force control OR whole-body control OR contact-rich manipulation OR model predictive control OR MPC OR sim2real OR domain randomization OR tactile sensing OR proprioception OR multimodal fusion OR 6DoF pose estimation OR stereo vision OR depth sensing OR point cloud registration OR NeRF OR event camera OR visual SLAM OR VO OR LiDAR SLAM OR scene reconstruction OR self-supervised robot learning OR sensorimotor learning OR representation learning OR diffusion policy OR imitation learning OR behavior cloning OR offline reinforcement learning OR RLHF for robots OR multi-robot coordination OR swarm robotics OR mobile manipulation OR robot autonomy stack OR real-time control OR kinodynamic planning OR receding horizon control OR time-optimal control OR nonlinear control OR differential dynamic programming OR learning-based control OR hybrid control OR hierarchical control OR control barrier function OR locomotion OR quadruped robot OR legged robot OR aerial robotics OR robot perception OR visuotactile sensing OR RGB-D fusion OR dynamic obstacle avoidance OR teleoperation OR dexterous manipulation OR kinodynamic planning OR neural implicit representation)"
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
