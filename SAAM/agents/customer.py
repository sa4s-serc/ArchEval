"""
Customer Agent - Represents the system stakeholders
Handles parsing of the SAAM_Evaluation_Extract schema.
"""
from typing import Dict, List, Any, Optional
import time
import json

class CustomerAgent:
    
    def __init__(self, customer_data: Dict[str, Any], llm_client: Optional[object] = None, logger: Optional[object] = None):
        self.data = customer_data
        # Optional LLM client used to answer follow-up questions
        self.llm = llm_client
        # Optional logger to record interactions
        self.logger = logger
        
    def get_system_overview(self) -> Dict[str, Any]:
        """Returns high-level system info from the schema."""
        # Try to pull from the specific SAAM schema location
        ctx = self.data.get("SAAM_Evaluation_Extract", {}).get("evaluation_inputs", {}).get("system_context", {})
        if ctx:
            return ctx
        # Fallback if the file structure is flatter
        return {
            "system_name": self.data.get("system_name", "Unknown"),
            "description": self.data.get("system_description", "")
        }
        
    def get_architectural_decisions(self) -> List[Dict]:
        """Returns architectural info for the Architect."""
        # Pull architectures from the schema
        arch_section = self.data.get("SAAM_Evaluation_Extract", {}).get("evaluation_inputs", {}).get("architecture_description", {})
        architectures = arch_section.get("architectures", [])
        return architectures

    def get_business_context(self) -> Dict:
        """Returns business goals."""
        ctx = self.data.get("SAAM_Evaluation_Extract", {}).get("evaluation_inputs", {}).get("system_context", {})
        return {"business_goals": ctx.get("business_goals", [])}

    def discuss_stakeholder_needs(self) -> List[str]:
        """
        Extracts stakeholder needs from system description and business goals.
        These become the basis for scenario generation.
        """
        stakeholder_statements = []
        
        # Get system context
        ctx = self.data.get("SAAM_Evaluation_Extract", {}).get("evaluation_inputs", {}).get("system_context", {})
        system_description = ctx.get("system_description", "")
        business_goals = ctx.get("business_goals", "")
        
        # If there are business goals, convert them to stakeholder statements
        if business_goals:
            if isinstance(business_goals, str):
                # Split by common delimiters
                goals_list = [g.strip() for g in business_goals.split(";") if g.strip()]
            else:
                goals_list = business_goals if isinstance(business_goals, list) else []
            
            for goal in goals_list:
                if goal:
                    stakeholder_statements.append(f"As a stakeholder, our system must: {goal}")
        
        # Also add system description as a broader stakeholder concern
        if system_description:
            stakeholder_statements.append(f"Our system needs to: {system_description}")
        
        # If still no statements, generate generic ones from architecture
        if not stakeholder_statements:
            arch_desc = self.data.get("SAAM_Evaluation_Extract", {}).get("evaluation_inputs", {}).get("architecture_description", {}).get("description", "")
            if arch_desc:
                stakeholder_statements.append(f"As a user, the system must support: {arch_desc}")
        
        return stakeholder_statements

    def answer_query(self, query: str, tracking_label: str = "General", caller: str = "Unknown") -> str:
        """
        LLM-backed responder used when other agents route questions to the CustomerAgent.
        If an LLM client was provided during instantiation, the query will be forwarded to it
        with the evaluation_inputs JSON included for context. Otherwise, a best-effort
        extraction from the loaded data is returned.
        """
        q = query.strip()
        eval_inputs = self.data.get("SAAM_Evaluation_Extract", {}).get("evaluation_inputs", {})

        # If an LLM client is available, always use it (with evaluation_inputs as context).
        if getattr(self, 'llm', None) is not None:
            prompt = f"""
You are the authoritative Customer agent. Use only the provided evaluation_inputs JSON to answer.

EVALUATION_INPUTS:
{json.dumps(eval_inputs, indent=2)}

QUESTION: {q}

RESPONSE_FORMAT: Return a compact JSON value or object that directly answers the question. Do NOT invent facts outside the provided evaluation_inputs. If the answer is not present, return {{"accepted": false, "reason": "Unknown or not documented"}}.
"""
            try:
                # Log the query if logger is available
                if getattr(self, 'logger', None) is not None:
                    try:
                        self.logger.log_interaction(sender=caller, recipient="Customer", content=query, interaction_type="QUERY")
                    except Exception:
                        self.logger.log(f"[Customer] Logged query from {caller}")

                # Call the LLM and measure duration
                start_time = time.time()
                resp = self.llm.query(prompt)
                duration = time.time() - start_time

                usage = None
                resp_text = resp
                if isinstance(resp, tuple) and len(resp) >= 1:
                    resp_text = resp[0]
                    if len(resp) == 2:
                        usage = resp[1]
                elif hasattr(resp, 'text'):
                    resp_text = resp.text
                else:
                    resp_text = str(resp)

                resp_text = resp_text if isinstance(resp_text, str) else str(resp_text)

                # Log LLM call and response
                if getattr(self, 'logger', None) is not None:
                    try:
                        self.logger.log_llm_call(role="Customer", prompt_summary="Describing architecture for stakeholder presentation", response_length=len(resp_text) if resp_text else 0, duration=duration)
                        self.logger.log_interaction(sender="LLM", recipient="Customer", content=f"Response length: {len(resp_text) if resp_text else 0} chars | Duration: {duration:.2f}s", interaction_type="RESPONSE")
                    except Exception:
                        try:
                            self.logger.log(f"[Customer] LLM call returned {len(resp_text) if resp_text else 0} chars in {duration:.2f}s")
                        except Exception:
                            pass

                # Track efficiency metrics if tracker is attached
                try:
                    if getattr(self, 'tracker', None) is not None:
                        if isinstance(usage, dict):
                            input_tokens = int(usage.get('input_tokens', 0))
                            output_tokens = int(usage.get('output_tokens', 0))
                        else:
                            input_tokens = len(prompt.split()) if prompt else 0
                            output_tokens = len(resp_text.split()) if resp_text else 0
                        try:
                            self.tracker.log_metric(task_label="Describing architecture for stakeholder presentation", agent_name="Customer", input_tokens=input_tokens, output_tokens=output_tokens, time_taken=duration)
                        except Exception:
                            if getattr(self, 'logger', None) is not None:
                                self.logger.log_interaction(sender="Customer", recipient="Tracker", content="Failed to log Customer efficiency metric", interaction_type="ERROR")
                except Exception:
                    pass

                # Log Customer's reply to caller
                if getattr(self, 'logger', None) is not None:
                    try:
                        self.logger.log_interaction(sender="Customer", recipient=caller, content=(resp_text[:2000] if isinstance(resp_text, str) else str(resp_text)), interaction_type="RESPONSE")
                    except Exception:
                        self.logger.log(f"[Customer] Logged response to {caller}")

                return resp_text
            except Exception as e:
                # On LLM failure, return minimal JSON indicating unknown and log
                fallback = json.dumps({"accepted": False, "reason": "LLM error or unavailable"})
                if getattr(self, 'logger', None) is not None:
                    try:
                        self.logger.log_interaction(sender="Customer", recipient=caller, content=fallback, interaction_type="RESPONSE")
                        self.logger.log_interaction(sender="Customer", recipient="Internal", content=f"Customer LLM error: {str(e)}", interaction_type="ERROR")
                    except Exception:
                        pass
                return fallback

        # Local heuristic fallback (no LLM available)
        ctx = eval_inputs.get("system_context", {})
        arch = eval_inputs.get("architecture_description", {})

        ql = q.lower()
        if "system" in ql or "overview" in ql or "description" in ql:
            return json.dumps(ctx)
        if "architecture" in ql or "component" in ql or "architect" in ql:
            return json.dumps(arch)
        if "goal" in ql or "business" in ql:
            return json.dumps({"business_goals": ctx.get("business_goals", [])})

        # Fallback: return the evaluation_inputs dictionary
        return json.dumps(eval_inputs)

    @classmethod
    def from_file(cls, filepath: str, llm_client: Optional[object] = None, logger: Optional[object] = None):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(data, llm_client=llm_client, logger=logger)