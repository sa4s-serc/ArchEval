import os
import json
from pathlib import Path

def aggregate_efficiency_metrics(root_dir, output_file):
    """
    Aggregates efficiency metrics from model directories and calculates
    total and average statistics per model.
    """
    final_output = {
        "models": []
    }

    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Error: Directory '{root_dir}' not found.")
        return

    # Iterate over each folder in the root directory (representing model names)
    for model_dir in root_path.iterdir():
        if not model_dir.is_dir():
            continue

        print(f"Processing model: {model_dir.name}...")
        
        # Initialize model entry
        model_entry = {
            "model_name": model_dir.name
        }
        
        # Variables for global statistics for this model
        grand_total_input = 0
        grand_total_output = 0
        grand_total_time = 0.0
        file_count = 0

        # Search recursively for efficiency files
        # Pattern: <model>/<project>/<project>_efficiency_<timestamp>.json
        for eff_file in model_dir.rglob("*_efficiency_*.json"):
            # Key name derived from filename (e.g., "FL-APU_efficiency")
            # Filename looks like FL-APU_efficiency_20260203_183339.json
            parts = eff_file.name.split('_efficiency_')
            if len(parts) < 2:
                continue
            
            project_name = parts[0]
            key_name = f"{project_name}_efficiency"

            try:
                with open(eff_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # File-specific counters
                    file_input = 0
                    file_output = 0
                    file_time = 0.0

                    # Parse the specific structure: root -> efficiency_metrics -> details list
                    if "efficiency_metrics" in data and isinstance(data["efficiency_metrics"], list):
                        for metric in data["efficiency_metrics"]:
                            if "details" in metric and isinstance(metric["details"], list):
                                for detail in metric["details"]:
                                    file_input += detail.get("input_tokens", 0)
                                    file_output += detail.get("output_tokens", 0)
                                    file_time += detail.get("time_taken", 0.0)

                    # Add file metrics to model entry
                    model_entry[key_name] = {
                        "input_tokens": file_input,
                        "output_tokens": file_output,
                        "time_taken": round(file_time, 2)
                    }
                    
                    # Update global totals for this model
                    grand_total_input += file_input
                    grand_total_output += file_output
                    grand_total_time += file_time
                    file_count += 1
                    
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON for {eff_file.name} in {model_dir.name}")
            except Exception as e:
                print(f"Error reading {eff_file.name}: {e}")

        # Calculate Averages
        if file_count > 0:
            avg_input = grand_total_input / file_count
            avg_output = grand_total_output / file_count
            avg_time = grand_total_time / file_count
        else:
            avg_input = 0
            avg_output = 0
            avg_time = 0

        # Add the Total Statistics block to the model entry
        model_entry["total_statistics"] = {
            "files_processed": file_count,
            "total_input_tokens": grand_total_input,
            "total_output_tokens": grand_total_output,
            "total_time_taken": round(grand_total_time, 2),
            "average_input_tokens": round(avg_input, 2),
            "average_output_tokens": round(avg_output, 2),
            "average_time_taken": round(avg_time, 2)
        }

        final_output["models"].append(model_entry)

    # Write the result to a file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)
    
    print(f"\nSuccess! Aggregated data written to: {output_file}")

# --- Configuration ---
ROOT_OUT_DIR = os.path.join(os.path.dirname(__file__), 'agents', 'outputs')
OUTPUT_FILENAME = os.path.join(os.path.dirname(__file__), 'consolidated_efficiency.json')

if __name__ == "__main__":
    aggregate_efficiency_metrics(ROOT_OUT_DIR, OUTPUT_FILENAME)
