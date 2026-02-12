import json
from client import LLMClient
from logger import Logger, EfficiencyTracker
from prompt_manager import PromptManager

class ATAMAgent:
    def __init__(self, name: str, role: str, llm: LLMClient, logger: Logger, tracker: EfficiencyTracker, knowledge_base: dict = None):
        self.name = name
        self.role = role
        self.llm = llm
        self.logger = logger
        self.tracker = tracker
        self.knowledge_base = knowledge_base or {} 
        self.conversation_history = []
        self.system_prompt = PromptManager.get_system_prompt(role)

    def process_request(self, task: str, router_callback, tracking_label: str = "General") -> dict:
        """
        Main execution loop. 
        Added tracking_label to group metrics in the efficiency log.
        """
        # Reinforce the requirement in the task prompt
        self.conversation_history.append(f"SYSTEM TASK: {task}")
        self.conversation_history.append("REMINDER: Your output 'data' must strictly match the structure defined in the MASTER DATA SCHEMA provided in your system instructions.")
        self.conversation_history.append(f"CURRENT KNOWLEDGE STATE: {json.dumps(self.knowledge_base)}")

        max_turns = 5
        for i in range(max_turns):
            if i == max_turns - 1:
                self.conversation_history.append("FINAL ATTEMPT: This is your last chance to provide a valid final answer. Give a payload of type 'final_answer' and not 'communication'.")
            response_text, usage = self.llm.generate(
                self.system_prompt,
                "\n".join(self.conversation_history[-9:]),
                max_tokens=8192 * (i+1)
            )
            
            self.tracker.log_metric(
                task_label=tracking_label,
                agent_name=self.name,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                time_taken=usage["time_taken"]
            )

            try:
                clean_json = response_text.replace("```json", "").replace("```", "").strip()
                payload = json.loads(clean_json)
                
                if payload.get("type") == "communication":
                    target = payload["target_agent"]
                    query = payload["content"]
                    
                    self.logger.log_interaction(self.name, target, query, "QUERY")
                    
                    response_from_target = router_callback(target, query, tracking_label)
                    
                    self.logger.log_interaction(target, self.name, response_from_target, "RESPONSE")
                    
                    self.conversation_history.append(f"RESPONSE FROM {target}: {response_from_target}")
                    continue 

                elif payload.get("type") == "final_answer":
                    reasoning = payload.get("reasoning", "")
                    data = payload.get("data", {})
                    if "ATAM_Evaluation" in data:
                        data = data["ATAM_Evaluation"]
                    if "ground_truth_outputs" in data:
                        data = data["ground_truth_outputs"]
                    
                    self.logger.log_reasoning(self.name, task, reasoning)
                    self.knowledge_base.update(data)
                    return data

            except json.JSONDecodeError:
                print(f"ERROR: {self.name} produced invalid JSON. Retrying...")
                continue
            
        print(f"ERROR: {self.name} failed to produce a valid final answer after {max_turns} attempts.")
        
        return {} 

    def answer_query(self, query: str, tracking_label: str = "General") -> str:
        """
        Called when another agent asks THIS agent a question.
        """
        context_prompt = f"""
        Another agent has asked you a question. 
        Answer based strictly on your known data: {json.dumps(self.knowledge_base)}
        QUESTION: {query}
        """
        
        response_text, usage = self.llm.generate(self.system_prompt, context_prompt)
        
        self.tracker.log_metric(
            task_label=tracking_label,
            agent_name=self.name,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            time_taken=usage["time_taken"]
        )

        return response_text