import os
import json
import glob

def calculate_averages():
    sections = ['scenarios', 'scenario_classification', 'scenario_evaluations', 'overall_results']
    lexical_metrics = ['rouge', 'bert', 'bleu', 'meteor']
    judge_metrics = ['relevance', 'coherence', 'completeness', 'conciseness']
    all_metrics = lexical_metrics + judge_metrics
    
    file_results = {}
    json_files = glob.glob("*.json")
    
    # We want to skip the output file if it already exists
    json_files = [f for f in json_files if f != "summary_averages.json"]
    
    if not json_files:
        print("No JSON files found in the current directory.")
        return

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, dict) or 'metadata' not in data:
                continue
                
            # For this file, calculate the average of each metric across the 4 sections
            per_file_metric_sums = {m: 0.0 for m in all_metrics}
            sections_found = {m: 0 for m in all_metrics}
            
            for section in sections:
                if section in data:
                    # Lexical
                    if 'lexical' in data[section]:
                        for m in lexical_metrics:
                            val = data[section]['lexical'].get(m, 0.0)
                            if val is not None:
                                per_file_metric_sums[m] += val
                                sections_found[m] += 1
                    
                    # Judge
                    if 'judge' in data[section]:
                        for m in judge_metrics:
                            val = data[section]['judge'].get(m, 0.0)
                            if val is not None:
                                per_file_metric_sums[m] += val
                                sections_found[m] += 1
            
            # Calculate averages for this file
            file_stats = {}
            for m in all_metrics:
                if sections_found[m] > 0:
                    file_stats[m] = per_file_metric_sums[m] / sections_found[m]
                else:
                    file_stats[m] = 0.0
            
            file_results[file_path] = file_stats
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not file_results:
        print("No valid JSON files were processed.")
        return

    # Calculate global averages for each metric across all processed files
    final_averages = {}
    for m in all_metrics:
        total_sum = sum(res[m] for res in file_results.values())
        final_averages[m] = total_sum / len(file_results)

    # Output to console
    print(f"Processed {len(file_results)} files.\n")
    print("Global Metric Averages (across 4 sections and all files):")
    for m in all_metrics:
        print(f"  - {m}: {final_averages[m]:.4f}")

    # Save to summary_averages.json
    output_summary = {
        "final_averages": final_averages,
        "json_results": file_results
    }
    
    with open("summary_averages.json", "w") as f:
        json.dump(output_summary, f, indent=4)
        print("\nDetailed summary saved to summary_averages.json")

if __name__ == "__main__":
    calculate_averages()
