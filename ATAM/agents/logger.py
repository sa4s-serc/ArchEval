import logging
import json
from datetime import datetime
import os

class Logger:
    def __init__(self, filename=None):
        if filename is None:
            filename = f"atam_agentic_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.filename = filename
        # if folder does not exist, create it
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        
        with open(self.filename, "w") as f:
            f.write(f"ATAM AGENTIC LOG - STARTED {datetime.now()}\n")
            f.write("="*80 + "\n\n")

    def log_interaction(self, sender: str, recipient: str, content: str, interaction_type: str = "MESSAGE"):
        """Logs direct communication between agents."""
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] [{interaction_type}] {sender} -> {recipient}:\n"
        entry += f"{content}\n"
        entry += "-"*50 + "\n"
        self._write(entry)

    def log_reasoning(self, agent_role: str, task: str, thought_process: str):
        """Logs the internal reasoning for a specific JSON update."""
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] [REASONING] Agent: {agent_role}\n"
        entry += f"TASK: {task}\n"
        entry += f"RATIONALE:\n{thought_process}\n"
        entry += "="*80 + "\n"
        self._write(entry)

    def _write(self, text):
        with open(self.filename, "a") as f:
            f.write(text)
            
class EfficiencyTracker:
    def __init__(self):
        self.stats = {}

    def log_metric(self, task_label: str, agent_name: str, input_tokens: int, output_tokens: int, time_taken: float):
        if task_label not in self.stats:
            self.stats[task_label] = []
            
        self.stats[task_label].append({
            "agent_name": agent_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "time_taken": round(time_taken, 2)
        })

    def save_to_file(self, filename: str):
        output = {}
        output["efficiency_metrics"] = [{"task": label, "details": details} for label, details in self.stats.items()]
        
        with open(filename, "w") as f:
            json.dump(output, f, indent=4)
        print(f"✅ Efficiency metrics saved to {filename}")