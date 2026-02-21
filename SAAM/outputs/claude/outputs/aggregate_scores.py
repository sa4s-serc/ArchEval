import os
import json
import glob

def calculate_averages():
    sections = ['scenarios', 'scenario_classification', 'scenario_evaluations', 'overall_results']
    lexical_metrics = ['rouge', 'bert', 'bleu', 'meteor']
    judge_metrics = ['relevance', 'coherence', 'completeness', 'conciseness']
    
    # Store sums and counts for each metric in each section
    totals_lexical = {s: {m: 0.0 for m in lexical_metrics} for s in sections}
    totals_judge = {s: {m: 0.0 for m in judge_metrics} for s in sections}
    file_count = 0
    
    # Find all JSON files in the current directory
    json_files = glob.glob("*.json")
    
    if not json_files:
        print("No JSON files found in the current directory.")
        return

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Basic validation to ensure it's one of our target JSONs
            if not isinstance(data, dict) or 'metadata' not in data:
                continue
                
            file_count += 1
            for section in sections:
                if section in data:
                    # Collect Lexical Metrics
                    if 'lexical' in data[section]:
                        lexical = data[section]['lexical']
                        for metric in lexical_metrics:
                            val = lexical.get(metric, 0.0)
                            if val is None: val = 0.0
                            totals_lexical[section][metric] += val
                    
                    # Collect Judge Metrics
                    if 'judge' in data[section]:
                        judge = data[section]['judge']
                        for metric in judge_metrics:
                            val = judge.get(metric, 0.0)
                            if val is None: val = 0.0
                            # Ensure we handle non-numeric values gracefully if any
                            if isinstance(val, (int, float)):
                                totals_judge[section][metric] += val
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if file_count == 0:
        print("No valid JSON files were processed.")
        return

    print(f"Processed {file_count} files.\n")
    
    results = {}
    combined_averages = []
    
    for section in sections:
        print(f"Section: {section}")
        
        # Lexical Stats
        lex_avgs = {}
        lex_sum = 0.0
        print("  Lexical Metrics:")
        for m in lexical_metrics:
            avg = totals_lexical[section][m] / file_count
            lex_avgs[m] = avg
            lex_sum += avg
            print(f"    - {m}: {avg:.4f}")
        lex_section_avg = lex_sum / len(lexical_metrics)
        print(f"    - Combined Lexical Average: {lex_section_avg:.4f}")

        # Judge Stats
        judge_avgs = {}
        judge_sum = 0.0
        print("  Judge Metrics:")
        for m in judge_metrics:
            avg = totals_judge[section][m] / file_count
            judge_avgs[m] = avg
            judge_sum += avg
            print(f"    - {m}: {avg:.4f}")
        judge_section_avg = judge_sum / len(judge_metrics)
        print(f"    - Combined Judge Average: {judge_section_avg:.4f}")

        # Section Combined (Normalized Judge to 0-1 for a fair combination if desired, 
        # but here we just average the two categories' averages)
        section_combined = (lex_section_avg + (judge_section_avg / 5.0)) / 2.0
        combined_averages.append(section_combined)
        print(f"  - Combined Normalized Section Average (Lexical + Judge/5): {section_combined:.4f}\n")
        
        results[section] = {
            "lexical": {
                "metrics": lex_avgs,
                "average": lex_section_avg
            },
            "judge": {
                "metrics": judge_avgs,
                "average": judge_section_avg
            },
            "combined_normalized_average": section_combined
        }

    final_avg = sum(combined_averages) / len(combined_averages)
    print(f"Final Combined Normalized Average (across all 4 sections): {final_avg:.4f}")
    
    # Optional: Save results to a file
    output_summary = {
        "file_count": file_count,
        "section_results": results,
        "final_combined_normalized_average": final_avg
    }
    
    with open("summary_averages.json", "w") as f:
        json.dump(output_summary, f, indent=4)
        print("\nDetailed summary saved to summary_averages.json")

if __name__ == "__main__":
    calculate_averages()
