import json

class PromptManager:
    MASTER_TEMPLATE_STR = "{}"

    @classmethod
    def setup(cls, template_path: str):
        try:
            with open(template_path, 'r') as f:
                template_data = json.load(f)
                cls.MASTER_TEMPLATE_STR = json.dumps(template_data, indent=2)
        except Exception as e:
            print(f"⚠️ Warning: Could not load SAAM Template: {e}")

    @staticmethod
    def get_communication_protocol() -> str:
        return f"""
You are participating in a multi-agent SAAM evaluation.
Agents may request clarifications from the Customer agent and coordinate with Architect and Manager.

*** STRICT OUTPUT REQUIREMENT ***
When asked to produce structured output, respond in valid JSON.

MASTER DATA SCHEMA:
{PromptManager.MASTER_TEMPLATE_STR}

RESPONSE TYPES:
1) communication
{{
  "type": "communication",
  "target_agent": "<Name of Agent>",
  "content": "<Your specific question or request>"
}}

2) final_answer
{{
  "type": "final_answer",
  "reasoning": "<Explain why you populated the fields this way>",
  "data": {{ }}
}}

AGENTS:
- Customer: Holds evaluation_inputs and answers factual questions based on that JSON.
- Architect: Technical authority for architecture evaluation.
- Manager: Facilitates scenario workshops and stakeholder coordination.
- EvaluationTeam: Orchestrates phases and routes messages between agents.
"""

    @staticmethod
    def get_system_prompt(role: str) -> str:
        prompts = {
            "Customer": """
                You are the Customer representative. Answer questions strictly from the provided evaluation_inputs JSON.
                If a question cannot be answered from the data, return 'Unknown' or request clarification.
            """,
            "Architect": """
                You are the Architect. When you need missing data, ask the Customer using a `communication` message.
            """,
            "Manager": """
                You are the Manager. Coordinate stakeholders, ask Architect for technical clarifications and Customer for missing inputs.
            """,
            "EvaluationTeam": """
                You are the Evaluation Team. 
            """
        }
        return prompts.get(role, "") + "\n" + PromptManager.get_communication_protocol()
