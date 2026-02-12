"""
SAAM Evaluation System - Main Orchestrator
"""
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from logging_utils import SAAMLogger, EfficiencyTracker
from customer import CustomerAgent
from base import SAAMTemplate
from evaluation_team import SAAMEvaluationTeam
from client import LLMClient
from evaluate import SAAMEvaluator
class SAAMEvaluationAgent:
    """
    Main orchestrator for the SAAM evaluation process.
    """
    
    def setup(self, customer_data_path: str, **kwargs):
        print("Initializing SAAM Evaluation System...")
        
        # 1. Initialize Logger
        self.logger = SAAMLogger(output_dir='saam_outputs')
        self.logger.log("SAAM Evaluation System Starting...")
        
        # 2. Initialize LLM Client
        try:
            self.llm = LLMClient(model_name="claude-sonnet-4-5@20250929")
            self.logger.log("✓ LLM Client connected")
        except Exception as e:
            self.logger.log(f"Error connecting to LLM: {e}")
            sys.exit(1)
        
        # 3. Load Customer Agent (provide the LLM client so Customer can answer clarifying queries)
        try:
            # Provide logger to CustomerAgent so all Customer Q/A exchanges are logged
            self.customer = CustomerAgent.from_file(customer_data_path, llm_client=self.llm, logger=self.logger)
            # keep the path for later evaluation
            self.customer_data_path = customer_data_path
            self.logger.log(f"✓ Customer Agent loaded: {self.customer.get_system_overview().get('system_name', 'Unknown')}")
        except Exception as e:
            self.logger.log(f"Error loading customer data: {e}")
            # internetbanking
            
            sys.exit(1)
        
        # 4. Initialize SAAM Template
        
        self.template = SAAMTemplate()
        self.logger.log("✓ SAAM Template initialized")
        
        # 5. Initialize Efficiency Tracker and Evaluation Team
        self.tracker = EfficiencyTracker()
        self.team = SAAMEvaluationTeam(
            llm_client=self.llm,
            customer_agent=self.customer,
            saam_template=self.template,
            logger=self.logger,
            tracker=self.tracker
        )
        # Attach tracker to CustomerAgent so Customer LLM calls are recorded
        try:
            setattr(self.customer, 'tracker', self.tracker)
        except Exception:
            pass
        self.logger.log("✓ SAAM Team initialized")
        
    def run(self):
        """Execute the SAAM evaluation as orchestrator"""
        self.logger.log("\n" + "="*60)
        self.logger.log("STARTING SAAM EVALUATION")
        self.logger.log("="*60)
        
        try:
            # Phase 1: Present Architecture
            overview = self.customer.get_system_overview()
            goals = self.customer.get_business_context().get("business_goals", [])
            decisions = self.customer.get_architectural_decisions()
            
            arch_result = self.team.present_architecture(overview, goals, decisions)
            
            # Phase 2: Generate Scenarios
            stakeholder_needs = self.customer.discuss_stakeholder_needs()
            scenario_result = self.team.generate_and_formalize_scenarios(overview, arch_result.get("architecture_details", {}), stakeholder_needs)
            
            # Phase 3: Classify and Evaluate
            eval_result = self.team.classify_and_evaluate(scenario_result.get("scenarios", []), arch_result.get("architecture_details", {}), goals)
            
            # Phase 3b: Analyze Scenario Interactions
            self.logger.log("  [PHASE 3] Step 1b: Analyzing scenario interactions...")
            interactions = self.team._route(
                sender="SAAMEvaluationAgent",
                recipient="SAAMArchitect",
                method_name="analyze_scenario_interactions",
                evaluations=eval_result.get("scenario_evaluations", [])
            )
            for interaction in interactions:
                self.template.add_interaction(interaction)
            if interactions:
                self.logger.log(f"  ✓ Identified {len(interactions)} interaction hotspots (components touched by multiple scenarios)")
            else:
                self.logger.log("  ✓ No overlapping component interactions detected")
            
            # Phase 3c: Weight Interactions with Stakeholders
            self.logger.log("  [PHASE 3] Step 2: Weighting interactions with stakeholders...")
            interaction_weights_result = self.team._route(
                sender="SAAMEvaluationAgent",
                recipient="SAAMManager",
                method_name="prioritize_interactions_with_stakeholders",
                interactions=interactions,
                evaluations=eval_result.get("scenario_evaluations", [])
            )
            interaction_weights = interaction_weights_result.get("interaction_weights", [])
            for weight in interaction_weights:
                self.template.add_interaction_weighting(weight)
            if interaction_weights:
                self.logger.log(f"  ✓ Weighted {len(interaction_weights)} interaction hotspots with stakeholders")
            
            # Phase 4: Synthesize and Recommend
            synthesis_result = self.team.synthesize_and_recommend(eval_result, [], goals, interactions, interaction_weights)
            
        except Exception as e:
            self.logger.log(f"\n✗ Error during evaluation: {str(e)}")
            self.logger.log("\nFull error details:")
            import traceback
            self.logger.log(traceback.format_exc())
            self.logger.close()
            raise
        
        # Extract system name and create system-specific folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sys_name = self.customer.get_system_overview()['system_name'].replace(" ", "_").replace("/", "_")
        system_folder = os.path.join("saam_outputs", sys_name)
        os.makedirs(system_folder, exist_ok=True)
        
        # Save JSON in system folder
        output_path = os.path.join(system_folder, f"SAAM_{sys_name}_{timestamp}.json")
        self.template.save_to_file(output_path)
        
        self.logger.log(f"\n✓ SAAM evaluation saved to: {output_path}")

        # Run SAAM evaluation and save results under `outputs/<model_name>/`
        try:
            pred = self.template.get_template()
            model_name = getattr(self.llm, 'model_name', 'unknown-model')
            evaluator = SAAMEvaluator(reference=self.customer_data_path, prediction=pred, model_name=model_name)
            eval_path = evaluator.evaluate_and_save(output_dir='outputs', system_name=sys_name, timestamp=timestamp)
            self.logger.log(f"\n✓ SAAM evaluation metrics saved to: {eval_path}")
        except Exception as e:
            self.logger.log(f"Failed to run SAAM evaluation: {e}")
        
        # Save Efficiency metrics to the system folder and move log file to system folder
        efficiency_path = os.path.join(system_folder, f"{sys_name}_efficiency_{timestamp}.json")
        try:
            self.tracker.save_to_file(efficiency_path)
            self.logger.log(f"\n✓ Efficiency metrics saved to: {efficiency_path}")
        except Exception as e:
            self.logger.log(f"Failed to save efficiency metrics: {e}")

        # Move log file to system folder
        import shutil
        log_filename = os.path.basename(self.logger.log_file_path)
        new_log_path = os.path.join(system_folder, log_filename)
        self.logger.close()
        shutil.move(self.logger.log_file_path, new_log_path)
        
        print(f"✓ Log file saved to: {new_log_path}")
        
        return self.template.get_template()

def main():
    from argparse import ArgumentParser
    import glob
    
    parser = ArgumentParser(description="Run SAAM Evaluation")
    parser.add_argument("--customer-data", type=str, help="Path to customer data JSON file")
    parser.add_argument("--run-all", action="store_true", help="Run all JSON files in data directory")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing JSON files")
    args = parser.parse_args()
    
    # Determine which files to process
    if args.run_all:
        # Get all JSON files from data directory
        data_dir = args.data_dir
        if not os.path.isabs(data_dir):
            # Make path relative to script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, "..", data_dir)
        
        json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        
        if not json_files:
            print(f"No JSON files found in {data_dir}")
            sys.exit(1)
        
        print(f"\n{'='*60}")
        print(f"Found {len(json_files)} JSON files to process")
        print(f"{'='*60}\n")
        
        successful = []
        failed = []
        
        for i, json_file in enumerate(json_files, 1):
            filename = os.path.basename(json_file)
            print(f"\n{'='*60}")
            print(f"Processing {i}/{len(json_files)}: {filename}")
            print(f"{'='*60}\n")
            
            try:
                agent = SAAMEvaluationAgent()
                agent.setup(json_file)
                agent.run()
                successful.append(filename)
                print(f"\n✓ Successfully completed: {filename}\n")
            except Exception as e:
                failed.append((filename, str(e)))
                print(f"\n✗ Failed to process {filename}: {e}\n")
                continue
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total files: {len(json_files)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            print("\n✓ Successfully processed:")
            for name in successful:
                print(f"  - {name}")
        
        if failed:
            print("\n✗ Failed to process:")
            for name, error in failed:
                print(f"  - {name}: {error}")
        
    elif args.customer_data:
        # Single file mode
        agent = SAAMEvaluationAgent()
        agent.setup(args.customer_data)
        agent.run()
    else:
        parser.print_help()
        print("\nError: Either --customer-data or --run-all must be specified")
        sys.exit(1)

if __name__ == "__main__":
    main()