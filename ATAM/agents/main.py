import json
import os
import sys
from client import LLMClient
from logger import Logger, EfficiencyTracker
from base import ATAMAgent
from prompt_manager import PromptManager
from evaluate import ATAMEvaluator

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")

class ATAMOrchestrator:
    def __init__(self, input_file: str, template_file: str, output_file: str, reasoning_file: str, efficiency_file: str):
        self.llm = LLMClient(model_name=MODEL_NAME)
        self.logger = Logger(filename=reasoning_file)
        self.tracker = EfficiencyTracker()
        self.output_file = output_file
        self.efficiency_file = efficiency_file
        
        PromptManager.setup(template_file)
        
        with open(input_file, 'r') as f:
            self.input_data = json.load(f)['ATAM_Evaluation']
            
        self.client_input = {
            "evaluation_inputs": self.input_data.get("evaluation_inputs", {})
        }

        self.agents = {
            "Customer": ATAMAgent("Customer", "CustomerProxy", self.llm, self.logger, self.tracker, knowledge_base=self.client_input),
            "Architect": ATAMAgent("Architect", "Architect", self.llm, self.logger, self.tracker),
            "Manager": ATAMAgent("Manager", "Manager", self.llm, self.logger, self.tracker),
            "EvaluationTeam": ATAMAgent("EvaluationTeam", "EvaluationTeam", self.llm, self.logger, self.tracker)
        }
        
        self.final_atam_state = {"ATAM_Evaluation": {"generated": {}}}

    def router(self, target_agent: str, query: str, tracking_label: str = "General") -> str:
        """
        Router now accepts tracking_label to ensure sub-calls are logged under the parent task.
        """
        if target_agent in self.agents:
            return self.agents[target_agent].answer_query(query, tracking_label)
        return "System Error: Agent not found."

    def run(self):
        print("🚀 Starting Agentic ATAM Evaluation ...")
        
        print("\n--- Phase 1: Architecture Extraction ---")
        
        arch_task = """
        Analyze the system description. 
        Populate the 'architectural_approaches' section of the MASTER DATA SCHEMA.
        Ensure you use the exact keys: 'approach_name', 'description', 'addressed_attributes'.
        """
        arch_data = self.agents["Architect"].process_request(
            arch_task, 
            self.router, 
            tracking_label=arch_task
        )
        self.final_atam_state["ATAM_Evaluation"]['generated']["architectural_approaches"] = arch_data.get("architectural_approaches", [])
        
        # Sync Knowledge
        self.agents["Manager"].knowledge_base.update(arch_data)
        self.agents["EvaluationTeam"].knowledge_base.update(arch_data)

        print("\n--- Phase 2: Utility Tree Generation ---")
        
        leader_task = """
        Generate the 'utility_tree' sections based on the MASTER DATA SCHEMA.
        Ensure 'utility_tree' follows the root -> quality_attribute_nodes -> children structure.
        Consult the Architect for technical review before finalizing.
        """
        utility_data = self.agents["Manager"].process_request(
            leader_task, 
            self.router, 
            tracking_label=leader_task
        )
        
        self.final_atam_state["ATAM_Evaluation"]['generated']["utility_tree"] = utility_data.get("utility_tree", {})
        self.agents["EvaluationTeam"].knowledge_base.update(utility_data)
        
        print("\n--- Phase 3: Scenario Generation ---")
        
        leader_task = """
        Generate scenarios and populate the 'scenarios' section of the MASTER DATA SCHEMA.
        Ensure each scenario strictly follows the 'scenario_id', 'scenario_text', and 'related_quality_attributes' keys.
        """
        scenario_data = self.agents["Manager"].process_request(
            leader_task,
            self.router,
            tracking_label=leader_task
        )
        
        self.final_atam_state["ATAM_Evaluation"]['generated']["scenarios"] = scenario_data.get("scenarios", [])
        self.agents["EvaluationTeam"].knowledge_base.update(scenario_data)
        
        print("\n--- Phase 4: Analysis (Risks & Sensitivities) ---")
        
        analyst_task = """
        Perform analysis. Populate the 'analysis_records' section of the MASTER DATA SCHEMA.
        Ensure findings include 'risks', 'non_risks', 'sensitivity_points', and 'tradeoff_points' strictly following the schema.
        """
        analysis_data = self.agents["EvaluationTeam"].process_request(
            analyst_task, 
            self.router, 
            tracking_label=analyst_task
        )
        self.final_atam_state["ATAM_Evaluation"]['generated']["analysis_records"] = analysis_data.get("analysis_records", [])

        # --- SAVE OUTPUT ---
        self.save_results()

    def save_results(self):
        full_output = {
            "ATAM_Evaluation": {
                **self.client_input.copy()
            }
        }
        
        full_output["ATAM_Evaluation"].update(self.final_atam_state["ATAM_Evaluation"])
        
        if not os.path.exists(os.path.dirname(self.output_file)):
            os.makedirs(os.path.dirname(self.output_file))
        
        with open(self.output_file, 'w') as f:
            json.dump(full_output, f, indent=2)
        
        # Save Efficiency Log
        self.tracker.save_to_file(self.efficiency_file)
        
        print(f"\n✅ Evaluation Complete. Results saved to {self.output_file}")
        print(f"✅ Efficiency Log saved to {self.efficiency_file}")

def process_file(input_file: str):
    file_name = os.path.basename(input_file)

    print(f"\n================ Processing Input File: {file_name} ================\n")
    if (not file_name.endswith('.json')) or (file_name == "template.json") or (not os.path.exists(input_file)):
        print(f"⚠️ Skipping invalid file: {file_name}. Ensure it is a valid JSON file.")
        return
    
    base_name = file_name.split('.')[0]
    model_base = MODEL_NAME.split('/')[-1]
    orchestrator = ATAMOrchestrator(
        input_file=input_file, 
        template_file="template.json", 
        output_file=f"outputs/{model_base}/{base_name}_output.json",
        reasoning_file=f"logs/{model_base}/{base_name}_reasoning.log",
        efficiency_file=f"logs/{model_base}/{base_name}_efficiency.json"
    )
    orchestrator.run()
    
    evaluator = ATAMEvaluator(reference_path=f"data/{base_name}.json", prediction_path=f"outputs/{model_base}/{base_name}_output.json")
    evaluator.evaluate(output_file=f"scores/{model_base}/{base_name}_evaluation.json")

if __name__ == "__main__":
    for input_folder in sys.argv[1:]:
        if not os.path.isdir(input_folder):
            process_file(input_folder)
            continue
        
        input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]
        
        for input_file in input_files:
            process_file(input_file)