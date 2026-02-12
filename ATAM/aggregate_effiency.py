import os
import json

def aggregate_efficiency_metrics(root_dir, output_file):
    """
    Aggregates efficiency metrics from model directories and calculates
    total and average statistics per model, including phase-wise breakdowns.
    """
    final_output = {
        "models": []
    }

    # Check if directory exists
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' not found.")
        return

    # Iterate over each folder in the root directory (representing model names)
    for model_dir in os.listdir(root_dir):
        model_path = os.path.join(root_dir, model_dir)

        # Ensure we are processing a directory
        if os.path.isdir(model_path):
            print(f"Processing model: {model_dir}...")
            
            # Initialize model entry
            model_entry = {
                "model_name": model_dir
            }
            
            # Variables for global statistics for this model
            grand_total_input = 0
            grand_total_output = 0
            grand_total_time = 0.0
            file_count = 0

            # Loop through all files in the model's directory
            for filename in os.listdir(model_path):
                if filename.endswith("_efficiency.json"):
                    file_path = os.path.join(model_path, filename)
                    
                    # Key name derived from filename (e.g., "banking_efficiency")
                    key_name = filename.replace(".json", "")

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                            # File-specific counters
                            file_input = 0
                            file_output = 0
                            file_time = 0.0
                            
                            # List to store phase-wise breakdown
                            phases_list = []

                            # Parse the specific structure: root -> efficiency_metrics -> details list
                            if "efficiency_metrics" in data and isinstance(data["efficiency_metrics"], list):
                                for index, metric in enumerate(data["efficiency_metrics"]):
                                    
                                    # Phase-specific counters
                                    phase_input = 0
                                    phase_output = 0
                                    phase_time = 0.0
                                    
                                    if "details" in metric and isinstance(metric["details"], list):
                                        for detail in metric["details"]:
                                            p_in = detail.get("input_tokens", 0)
                                            p_out = detail.get("output_tokens", 0)
                                            p_time = detail.get("time_taken", 0.0)
                                            
                                            # Add to phase totals
                                            phase_input += p_in
                                            phase_output += p_out
                                            phase_time += p_time

                                    # Clean up task description for cleaner JSON
                                    task_desc = metric.get("task", "").strip()

                                    # Append phase stats
                                    phases_list.append({
                                        "phase_index": index + 1,
                                        "task": task_desc,
                                        "input_tokens": phase_input,
                                        "output_tokens": phase_output,
                                        "time_taken": round(phase_time, 2)
                                    })

                                    # Add phase totals to file totals
                                    file_input += phase_input
                                    file_output += phase_output
                                    file_time += phase_time

                            # Add file metrics to model entry, now including the 'phases' breakdown
                            model_entry[key_name] = {
                                "input_tokens": file_input,
                                "output_tokens": file_output,
                                "time_taken": round(file_time, 2),
                                "phases": phases_list
                            }
                            
                            # Update global totals for this model
                            grand_total_input += file_input
                            grand_total_output += file_output
                            grand_total_time += file_time
                            file_count += 1
                            
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse JSON for {filename} in {model_dir}")
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")

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
# Update 'logs' to the actual path of your logs folder if different
ROOT_LOGS_DIR = 'logs' 
OUTPUT_FILENAME = 'consolidated_efficiency.json'

if __name__ == "__main__":
    aggregate_efficiency_metrics(ROOT_LOGS_DIR, OUTPUT_FILENAME)