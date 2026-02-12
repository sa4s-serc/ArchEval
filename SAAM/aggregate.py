import os
import json
import statistics
from collections import defaultdict
from pathlib import Path

# Configuration
OUTPUT_DIR = Path(__file__).parent / "agents" / "outputs"
OUTPUT_FILE = Path(__file__).parent / "consolidated_scores.json"

def nested_dict():
    """Creates a recursive dictionary that auto-initializes."""
    return defaultdict(nested_dict)

def collect_values(source, target):
    """
    Recursively traverses the source JSON and appends numerical values 
    to lists in the target dictionary structure.
    """
    if not isinstance(source, dict):
        return

    for key, value in source.items():
        if key == "metadata" or key == "reasoning":
            continue
            
        if isinstance(value, dict):
            collect_values(value, target[key])
        elif isinstance(value, (int, float)):
            if "values" not in target[key]:
                target[key]["values"] = []
            target[key]["values"].append(value)
        elif isinstance(value, list):
            if "values_list" not in target[key]:
                target[key]["values_list"] = []
            target[key]["values_list"].append(value)

def calculate_stats(values):
    """Calculates Mean, Median, and StdDev for a list of numbers."""
    clean_vals = [v for v in values if v is not None]
    if not clean_vals:
        return None
    
    try:
        mean_val = statistics.mean(clean_vals)
        median_val = statistics.median(clean_vals)
        std_val = statistics.stdev(clean_vals) if len(clean_vals) > 1 else 0.0
        
        return {
            "mean": round(mean_val, 4),
            "median": round(median_val, 4),
            "std": round(std_val, 4)
        }
    except Exception:
        return None

def calculate_stats_for_lists(list_of_lists):
    """Calculates stats index-wise for lists (e.g. rougeL [P, R, F])."""
    if not list_of_lists:
        return None
    
    max_len = max(len(x) for x in list_of_lists)
    transposed = []
    for i in range(max_len):
        col = [x[i] for x in list_of_lists if len(x) > i]
        transposed.append(col)
        
    stats_per_index = [calculate_stats(col) for col in transposed]
    
    def get_stat(stats, key):
        return stats[key] if stats else 0.0

    return {
        "mean": [get_stat(s, "mean") for s in stats_per_index],
        "median": [get_stat(s, "median") for s in stats_per_index],
        "std": [get_stat(s, "std") for s in stats_per_index]
    }

def process_collected_data(data_node):
    """
    Recursively transforms lists of raw values into statistical summaries.
    """
    result = {}
    
    sub_keys = [k for k in data_node.keys() if k not in ["values", "values_list"]]
    
    if sub_keys:
        for key in sub_keys:
            processed = process_collected_data(data_node[key])
            if processed is not None:
                result[key] = processed
        return result
    else:
        if "values_list" in data_node:
            return calculate_stats_for_lists(data_node["values_list"])
        elif "values" in data_node:
            return calculate_stats(data_node["values"])
    
    return None

def main():
    if not OUTPUT_DIR.exists():
        print(f"Error: Directory '{OUTPUT_DIR}' not found.")
        return

    final_output = {}

    for model_dir in OUTPUT_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        print(f"Processing model: {model_name}...")

        phase_data = nested_dict()
        global_data = nested_dict()

        file_count = 0
        
        # In SAAM, score files are in <model_dir>/outputs/*_saam_evaluation_*.json
        scores_subdir = model_dir / "outputs"
        if not scores_subdir.exists():
            # Try exploring deeper just in case
            score_files = list(model_dir.rglob("*_saam_evaluation_*.json"))
        else:
            score_files = list(scores_subdir.glob("*_saam_evaluation_*.json"))

        for score_file in score_files:
            file_count += 1
            with open(score_file, "r", encoding="utf-8") as f:
                try:
                    content = json.load(f)
                    
                    # Store data by section
                    collect_values(content, phase_data)
                    
                    # Store data globally (across sections)
                    for key, section_content in content.items():
                        if key == "metadata": continue
                        collect_values(section_content, global_data)
                        
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON: {score_file}")

        if file_count == 0:
            print(f"No evaluation JSON files found for {model_name}")
            continue

        # Calculate Statistics
        model_results = process_collected_data(phase_data)
        
        if model_results is None: model_results = {}
        
        global_stats = process_collected_data(global_data)
        model_results["global_average"] = global_stats

        final_output[model_name] = model_results

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
    
    print(f"\nSuccess! Consolidated scores saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
