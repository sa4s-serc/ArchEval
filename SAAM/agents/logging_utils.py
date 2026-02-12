"""
SAAM Logging Utilities
Centralized logging functions for separation of concerns.
Provides structured logging for agent interactions, LLM calls, and reasoning.
"""
import os
from datetime import datetime
from typing import Optional, TextIO
import json


class LogFormatter:
    """
    Formats log entries with consistent structure.
    Separates formatting logic from logging logic.
    """
    
    @staticmethod
    def format_header() -> str:
        """Format the log file header."""
        header = f"SAAM EVALUATION LOG - STARTED {datetime.now()}\n"
        header += "=" * 80 + "\n\n"
        return header
    
    @staticmethod
    def format_footer() -> str:
        """Format the log file footer."""
        return f"\nSAAM EVALUATION LOG - ENDED {datetime.now()}\n"
    
    @staticmethod
    def format_phase(phase: str) -> str:
        """Format a phase transition entry."""
        entry = "\n" + "=" * 60 + "\n"
        entry += f"PHASE: {phase}\n"
        entry += "=" * 60 + "\n"
        return entry
    
    @staticmethod
    def format_interaction(sender: str, recipient: str, content: str, 
                          interaction_type: str = "MESSAGE") -> str:
        """
        Format an interaction between agents/roles.
        
        Structure:
        [HH:MM:SS] [TYPE] Sender -> Recipient:
        Content
        --------------------------------------------------
        """
        timestamp = datetime.now().strftime('%H:%M:%S')
        entry = f"[{timestamp}] [{interaction_type}] {sender} -> {recipient}:\n"
        entry += f"{content}\n"
        entry += "-" * 50 + "\n"
        return entry
    
    @staticmethod
    def format_reasoning(agent_role: str, task: str, thought_process: str) -> str:
        """
        Format a reasoning/decision entry.
        
        Structure:
        [HH:MM:SS] [REASONING] Agent: RoleName
        TASK: TaskDescription
        RATIONALE:
        ThoughtProcess
        ================================================================================
        """
        timestamp = datetime.now().strftime('%H:%M:%S')
        entry = f"[{timestamp}] [REASONING] Agent: {agent_role}\n"
        entry += f"TASK: {task}\n"
        entry += f"RATIONALE:\n{thought_process}\n"
        entry += "=" * 80 + "\n"
        return entry
    
    @staticmethod
    def format_llm_call(role: str, prompt_summary: str, response_length: int, 
                       duration: Optional[float] = None) -> str:
        """
        Format an LLM API call entry.
        
        Structure:
        [HH:MM:SS] [LLM_CALL] Role: RoleName
        Prompt: Summary...
        Response Length: N chars | Duration: X.XXs
        --------------------------------------------------
        """
        timestamp = datetime.now().strftime('%H:%M:%S')
        entry = f"[{timestamp}] [LLM_CALL] Role: {role}\n"
        entry += f"Prompt: {prompt_summary[:100]}...\n"
        entry += f"Response Length: {response_length} chars"
        if duration:
            entry += f" | Duration: {duration:.2f}s"
        entry += "\n"
        entry += "-" * 50 + "\n"
        return entry
    
    @staticmethod
    def format_console_interaction(sender: str, recipient: str, interaction_type: str) -> str:
        """Format abbreviated console output for interactions."""
        return f"  [{interaction_type}] {sender} → {recipient}"


class LogWriter:
    """
    Handles writing to log files and console.
    Manages file I/O and ensures proper flushing.
    """
    
    def __init__(self, log_file: TextIO):
        self.log_file = log_file
    
    def write_to_file(self, content: str) -> None:
        """Write content to log file with automatic flush."""
        self.log_file.write(content)
        self.log_file.flush()
    
    def write_to_console(self, content: str) -> None:
        """Write content to console."""
        print(content)
    
    def write_to_both(self, content: str) -> None:
        """Write same content to both file and console."""
        self.write_to_console(content)
        self.write_to_file(content + "\n")


class SAAMLogger:
    """
    Main logger for SAAM evaluation process.
    Coordinates formatting and writing with separation of concerns.
    """
    
    def __init__(self, output_dir: str = "saam_outputs"):
        """
        Initialize logger with output directory.
        
        Args:
            output_dir: Directory where log files will be stored
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(output_dir, f"saam_log_{timestamp}.txt")
        self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
        
        # Initialize formatter and writer
        self.formatter = LogFormatter()
        self.writer = LogWriter(self.log_file)
        
        # Write header
        self.writer.write_to_file(self.formatter.format_header())
    
    def log(self, message: str):
        """
        Standard log message to both console and file.
        
        Args:
            message: Message to log
        """
        self.writer.write_to_both(message)
    
    def log_phase(self, phase: str):
        """
        Log a major phase transition.
        
        Args:
            phase: Name of the phase
        """
        formatted = self.formatter.format_phase(phase)
        self.writer.write_to_both(formatted)
    
    def log_interaction(self, sender: str, recipient: str, content: str, 
                       interaction_type: str = "MESSAGE"):
        """
        Log communication between agents/roles.
        
        Args:
            sender: Name of the sending agent/role
            recipient: Name of the receiving agent/role  
            content: The message or query content
            interaction_type: Type of interaction (MESSAGE, QUERY, RESPONSE, etc.)
        """
        # Write detailed entry to file
        formatted = self.formatter.format_interaction(sender, recipient, content, interaction_type)
        self.writer.write_to_file(formatted)
        
        # Write abbreviated version to console
        console_msg = self.formatter.format_console_interaction(sender, recipient, interaction_type)
        self.writer.write_to_console(console_msg)
    
    def log_reasoning(self, agent_role: str, task: str, thought_process: str):
        """
        Log internal reasoning process for a decision or action.
        
        Args:
            agent_role: The role performing the reasoning
            task: What task is being reasoned about
            thought_process: The reasoning or rationale
        """
        formatted = self.formatter.format_reasoning(agent_role, task, thought_process)
        self.writer.write_to_file(formatted)
    
    def log_llm_call(self, role: str, prompt_summary: str, response_length: int, 
                    duration: Optional[float] = None):
        """
        Log LLM API calls for debugging and cost tracking.
        
        Args:
            role: Which role made the LLM call
            prompt_summary: Brief summary of what was asked
            response_length: Length of response in characters
            duration: Optional duration in seconds
        """
        formatted = self.formatter.format_llm_call(role, prompt_summary, response_length, duration)
        self.writer.write_to_file(formatted)
    
    def close(self):
        """Close the log file with proper footer."""
        self.writer.write_to_file(self.formatter.format_footer())
        self.log_file.close()


# Utility functions for standalone use
def create_logger(output_dir: str = "saam_outputs") -> SAAMLogger:
    """
    Factory function to create a logger instance.
    
    Args:
        output_dir: Directory for log files
        
    Returns:
        SAAMLogger: Configured logger instance
    """
    return SAAMLogger(output_dir)


def format_timestamp() -> str:
    """Get current timestamp in standard format."""
    return datetime.now().strftime('%H:%M:%S')


def format_date() -> str:
    """Get current date in standard format."""
    return datetime.now().strftime('%Y-%m-%d')


def format_datetime() -> str:
    """Get current datetime in standard format."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class EfficiencyTracker:
    """
    Tracks simple efficiency metrics for LLM calls and agent tasks.
    Mirrors the ATAM `EfficiencyTracker` minimal interface.
    """
    def __init__(self):
        self.stats = {}

    def log_metric(self, task_label: str, agent_name: str, input_tokens: int, output_tokens: int, time_taken: float):
        # Normalize task labels so related actions aggregate under canonical names
        norm = self._normalize_task_label(task_label)
        if norm not in self.stats:
            self.stats[norm] = []
        self.stats[norm].append({
            "agent_name": agent_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "time_taken": round(time_taken, 2),
            "original_task_label": task_label
        })

    def _normalize_task_label(self, task_label: str) -> str:
        """Map various task label variants to canonical task names for reporting.

        Rules (case-insensitive substring matching):
        - Any task mentioning 'classifying' or 'analyzing architectural impact' or
          'presenting evaluation findings' or 'reviewing architect classifications'
          maps to 'Formalizing stakeholder scenarios with architectural context'
        - 'Reviewing recommendations for stakeholder agreement' maps to
          'Synthesizing evaluation findings into architectural assessment'
        - Otherwise, return the original label unchanged.
        """
        if not task_label:
            return "Unknown Task"

        tl = task_label.lower()
        if ("classifying" in tl or "analyzing architectural impact" in tl or
                "presenting evaluation findings" in tl or "reviewing architect classifications" in tl):
            return "Formalizing stakeholder scenarios with architectural context"
        if "reviewing recommendations for stakeholder agreement" in tl or "reviewing recommendations" in tl:
            return "Synthesizing evaluation findings into architectural assessment"
        if "assess phase 1: architecture presentation" in tl:
            return "Describing architecture for stakeholder presentation"
        if "assess phase 2: scenario generation" in tl:
            return "Formalizing stakeholder scenarios with architectural context"
        if "assess phase 3: classification and evaluation" in tl:
            return "Prioritizing scenario interactions with stakeholders"
        if "assess phase 4: synthesis and recommendations" in tl:
            return "Synthesizing evaluation findings into architectural assessment"
        # Fallback: return the original label
        return task_label

    def save_to_file(self, filename: str):
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        output = {}
        output["efficiency_metrics"] = [{"task": label, "details": details} for label, details in self.stats.items()]
        with open(filename, "w") as f:
            json.dump(output, f, indent=4)
        print(f"✅ Efficiency metrics saved to {filename}")
