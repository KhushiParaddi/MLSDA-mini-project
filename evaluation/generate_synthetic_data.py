import sys
import os
import json
import random

# Fix import path for running directly or via module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def generate_synthetic_edge_cases(output_path="data/synthetic_eval_set.json"):
    """
    Generates synthetic edge cases combining multi-variable constraints:
    - Plot dimensions (depth, width)
    - Building heights (borderline cases: 11.4m vs 11.6m)
    - Road width limits (less than 9.0m, 9.0m-12.0m, >12.0m)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    synthetic_cases = []
    plot_depths = [8.0, 10.5, 14.0, 20.0, 26.0]
    plot_widths = [5.5, 8.5, 11.0, 15.0, 22.0]
    heights = [9.0, 11.5, 12.0, 15.0, 24.5]
    road_widths = [6.0, 8.5, 9.0, 12.0, 18.0]

    case_id = 1
    for depth in plot_depths:
        for width in plot_widths:
            height = random.choice(heights)
            road_width = random.choice(road_widths)

            # Rule logic evaluation for ground truth
            high_rise = height > 11.5 or height >= 24.0
            road_restricted = road_width < 9.0

            if road_restricted and high_rise:
                verdict = "PROHIBITED"
                reasoning = f"Road width ({road_width}m) < 9.0m strictly restricts height to 11.5m maximum."
            else:
                verdict = "PERMITTED_WITH_SETBACKS"
                reasoning = f"Calculated under BDA Table 8/9 for height {height}m and plot size {depth}x{width}m."

            query = (
                f"Is it permissible to construct a building of height {height}m on a plot of "
                f"depth {depth}m and width {width}m abutting an {road_width}m wide road?"
            )

            synthetic_cases.append({
                "id": f"SYN-TC-{case_id:03d}",
                "category": "Synthetic Edge Case",
                "query": query,
                "ground_truth": {
                    "verdict": verdict,
                    "height": height,
                    "road_width": road_width,
                    "reasoning": reasoning
                },
                "expected_chunks": ["Passage_A_Setbacks", "Passage_B_Road_Width"]
            })
            case_id += 1

    payload = {
        "metadata": {
            "total_synthetic_samples": len(synthetic_cases),
            "description": "Synthetic stress test dataset targeting boundary conditions."
        },
        "test_cases": synthetic_cases
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    print(f"Generated {len(synthetic_cases)} synthetic test cases at: {output_path}")

if __name__ == "__main__":
    generate_synthetic_edge_cases()