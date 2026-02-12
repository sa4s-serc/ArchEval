"""
SAAM Role Definitions
Defines specific personas for the Evaluation Team.
Enhanced with detailed logging for LLM interactions and reasoning.
"""
import json
import re
import time
from typing import Dict, Any, List

class BaseRole:
    def __init__(self, llm_client, logger, tracker=None, customer_agent=None):
        self.llm = llm_client
        self.logger = logger
        self.tracker = tracker
        # Optional reference to the Customer agent for on-demand queries
        self.customer = customer_agent

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, handling markdown code blocks and extra text"""
        # Log the extraction attempt
        self.logger.log_interaction(
            sender="JSONExtractor",
            recipient="Parser",
            content=f"Attempting to extract JSON from response of length {len(text)}",
            interaction_type="PARSE"
        )
        
        # First try to extract from markdown code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                result = json.loads(json_str)
                self.logger.log_interaction(
                    sender="JSONExtractor",
                    recipient="Parser",
                    content="Successfully extracted JSON from markdown code block",
                    interaction_type="SUCCESS"
                )
                return result
            except json.JSONDecodeError as e:
                self.logger.log_interaction(
                    sender="JSONExtractor",
                    recipient="Parser",
                    content=f"Failed to parse JSON from markdown block: {str(e)}\nJSON content (first 500 chars): {json_str[:500]}",
                    interaction_type="ERROR"
                )
        
        # Fallback: find the first complete JSON object
        start = text.find('{')
        if start != -1:
            # Find the matching closing brace by counting braces
            brace_count = 0
            in_string = False
            escape_next = False
            
            for i in range(start, len(text)):
                char = text[i]
                
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == '\\':
                    escape_next = True
                    continue
                    
                if char == '"' and not escape_next:
                    in_string = not in_string
                    
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found complete JSON object
                            json_str = text[start:i+1]
                            try:
                                result = json.loads(json_str)
                                self.logger.log_interaction(
                                    sender="JSONExtractor",
                                    recipient="Parser",
                                    content="Successfully extracted JSON using brace matching",
                                    interaction_type="SUCCESS"
                                )
                                return result
                            except json.JSONDecodeError as e:
                                self.logger.log_interaction(
                                    sender="JSONExtractor",
                                    recipient="Parser",
                                    content=f"Failed to parse JSON from brace-matched content: {str(e)}\nJSON content (first 500 chars): {json_str[:500]}",
                                    interaction_type="ERROR"
                                )
        
        # If all attempts failed, log the full response for debugging
        error_msg = f"No valid JSON found in response. Response preview (first 1000 chars):\n{text[:1000]}"
        self.logger.log_interaction(
            sender="JSONExtractor",
            recipient="Parser",
            content=error_msg,
            interaction_type="FATAL_ERROR"
        )
        # Fallback: if a Customer agent is available, ask Customer to reformat/extract JSON
        if getattr(self, 'customer', None) is not None:
            try:
                self.logger.log_interaction(sender="JSONExtractor", recipient="Customer", content="Attempting fallback JSON extraction via Customer agent", interaction_type="FALLBACK")
                fallback_prompt = f"The following LLM response failed to parse as JSON. Extract and return a single valid JSON object that captures the structured content. Do NOT add commentary. Respond ONLY with JSON.\n\nRESPONSE:\n{text}"
                cust_resp = self.customer.answer_query(fallback_prompt, caller="JSONExtractor")
                # If Customer returned a dict already, accept it
                if isinstance(cust_resp, dict):
                    return cust_resp
                # Otherwise try to parse text
                try:
                    parsed = json.loads(cust_resp)
                    self.logger.log_interaction(sender="JSONExtractor", recipient="Parser", content="Successfully obtained JSON via Customer fallback", interaction_type="SUCCESS")
                    return parsed
                except Exception:
                    # Continue to raise below
                    self.logger.log_interaction(sender="JSONExtractor", recipient="Parser", content="Customer fallback did not return valid JSON", interaction_type="ERROR")
            except Exception:
                # If fallback failed, log and raise the original error
                self.logger.log_interaction(sender="JSONExtractor", recipient="Customer", content="Customer fallback extraction failed", interaction_type="ERROR")
        raise ValueError(error_msg)
    
    def _call_llm_with_logging(self, role_name: str, task_description: str, prompt: str, **kwargs) -> str:
        """
        Call LLM with comprehensive logging.
        
        Args:
            role_name: Name of the role making the call
            task_description: Brief description of what's being asked
            prompt: The actual prompt
            **kwargs: Additional parameters for LLM
            
        Returns:
            str: LLM response
        """
        # Log the interaction
        self.logger.log_interaction(
            sender=role_name,
            recipient="LLM",
            content=f"Task: {task_description}\nPrompt length: {len(prompt)} chars",
            interaction_type="QUERY"
        )
        
        # Make the LLM call with timing
        start_time = time.time()
        response = self.llm.query(prompt, **kwargs)
        duration = time.time() - start_time

        # Support clients that return (text, usage) tuples or plain strings
        response_text = response
        usage = None
        if isinstance(response, tuple) and len(response) == 2:
            response_text, usage = response
        elif hasattr(response, 'text'):
            try:
                response_text = response.text
            except Exception:
                response_text = str(response)
        
        # Log LLM call metrics
        self.logger.log_llm_call(
            role=role_name,
            prompt_summary=task_description,
            response_length=len(response_text) if response_text else 0,
            duration=duration
        )
        
        # Log response received
        self.logger.log_interaction(
            sender="LLM",
            recipient=role_name,
            content=f"Response length: {len(response_text) if response_text else 0} chars | Duration: {duration:.2f}s",
            interaction_type="RESPONSE"
        )

        # Track efficiency metrics when a tracker is available. Use word counts as a simple proxy for tokens.
        try:
            if getattr(self, 'tracker', None) is not None:
                if isinstance(usage, dict):
                    input_tokens = int(usage.get('input_tokens', 0))
                    output_tokens = int(usage.get('output_tokens', 0))
                else:
                    # Fallback to word-count proxy
                    input_tokens = len(prompt.split()) if prompt else 0
                    output_tokens = len(response_text.split()) if response_text else 0

                self.tracker.log_metric(task_label=task_description.strip(), agent_name=role_name,
                                        input_tokens=input_tokens, output_tokens=output_tokens,
                                        time_taken=duration)
        except Exception:
            # Don't let tracking failures block the main flow
            self.logger.log_interaction(sender="Tracker", recipient=role_name, content="Failed to log efficiency metric", interaction_type="ERROR")

        # Return the textual response to callers (preserve usage via tracker)
        return response_text

class SAAMManager(BaseRole):
    """
    Role: SAAM Facilitator / Workshop Manager
    
    Responsibilities:
    ─────────────────
    1. BRIDGE STAKEHOLDERS & ARCHITECTS: Translate informal needs into architecture-aware scenarios
    2. ENFORCE ARCHITECTURAL CONSTRAINTS: Ensure scenarios are realistically tied to architecture
    3. FEEDBACK LOOPS: Present evaluation findings back to stakeholders for validation
    4. SCENARIO REFINEMENT: Manage iterative cycles between phases 2-3
    
    Key Inputs:
    - Stakeholder statements (informal needs/concerns)
    - Architecture context (constraints, capabilities, components)
    - Business goals (strategic drivers)
    
    Key Outputs:
    - Architecture-aware scenarios
    - Refined scenarios incorporating stakeholder feedback
    """
    
    def conduct_elicitation_workshop(self, system_context: Dict, architecture_context: Dict, 
                                     stakeholder_statements: List[str]) -> Dict:
        """
        PHASE 2: Scenario Generation (ARCHITECTURE-AWARE)
        
        The Manager formalizes stakeholder needs into SAAM scenarios while considering:
        - What the current architecture CAN do (direct scenarios)
        - What would require changes (indirect scenarios)
        - Which components are involved
        """
        self.logger.log("  [Manager→LLM]: Formalizing scenarios with architectural constraints...")

        # If the manager has no stakeholder statements, request them from the Customer agent
        if (not stakeholder_statements or len(stakeholder_statements) == 0) and getattr(self, 'customer', None) is not None:
            try:
                self.logger.log("  [Manager→Customer]: Requesting stakeholder statements from Customer agent...")
                resp = self.customer.answer_query("Please list stakeholder needs and business goals present in evaluation_inputs as a JSON array of concise statements.", caller="SAAMManager")
                # Attempt to parse JSON array response
                parsed = json.loads(resp)
                if isinstance(parsed, list):
                    stakeholder_statements = parsed
                    self.logger.log(f"  ✓ Retrieved {len(parsed)} stakeholder statements from Customer agent")
                else:
                    # Not a list; keep fallback behavior
                    self.logger.log("  ! Customer returned non-list response for stakeholder statements; using defaults")
            except Exception:
                self.logger.log("  ! Failed to retrieve stakeholder statements from Customer agent; proceeding with provided inputs")

        # Prepare architecture summary for Manager to use as constraint reference
        arch_summary = {
            "styles": architecture_context.get("architecture_styles", []),
            "components": [c.get("component_name") for c in architecture_context.get("key_components", [])],
            "description": architecture_context.get("description", "")
        }
        
        base_prompt = f"""
CONTEXT - You are a SAAM Workshop Facilitator and Requirements Analyst
Your role: Bridge the gap between stakeholders and architects by creating scenarios that are:
1. Grounded in real stakeholder concerns
2. Aware of current architecture capabilities and constraints
3. Properly classified for impact analysis
4. Cover diverse stakeholder perspectives: End-user, Developer, Maintainer, System Administrator (create at least one scenario per perspective; synthesize if not explicitly stated)

SYSTEM BEING EVALUATED:
Name: {system_context.get('system_name')}
Description: {system_context.get('system_description')}

CURRENT ARCHITECTURE (Available to stakeholders):
Architecture Styles: {arch_summary.get('styles')}
Key Components: {arch_summary.get('components')}
Description: {arch_summary.get('description')}

STAKEHOLDER NEEDS/CONCERNS EXPRESSED:
{json.dumps(stakeholder_statements, indent=2)}

TASK - Convert stakeholder statements into formal SAAM Scenarios

For EACH stakeholder statement:
1. Create a formal scenario text (specific, measurable, testable)
2. Identify scenario type:
   - "Use Case": Current functionality stakeholders rely on
   - "Change (Growth)": New capabilities needed in future
   - "Change (Exploratory)": Exploratory/optional changes
3. Identify which components are involved (reference current architecture)
4. Assess whether scenario directly matches current architecture or would require changes
5. Assign priority based on business impact
6. Identify the stakeholder perspective (End-user | Developer | Maintainer | System Administrator)
7. Add notes on architectural feasibility

CRITICAL: Scenarios must be SPECIFIC and directly evaluable against the architecture.

Output JSON:
{{
    "scenarios": [
        {{
            "scenario_id": "S-01",
            "scenario_text": "Specific, measurable scenario statement...",
            "scenario_type": "Use Case | Change (Growth) | Change (Exploratory)",
            "involved_components": ["Component1", "Component2"],
            "feasibility_note": "Notes on architectural fit (direct vs. indirect)",
            "priority": "High | Medium | Low",
            "stakeholder_role": "End-user | Developer | Maintainer | System Administrator",
            "stakeholder_rationale": "Why this matters to stakeholders"
        }}
    ],
    "notes": "Overall notes about scenario generation and any architectural constraints that shaped scenarios"
}}
        """
        
        # FORCED CONSENSUS MECHANISM - ensure we get meaningful scenarios
        max_attempts = 3
        result = {}
        
        for attempt in range(1, max_attempts + 1):
            if attempt == 1:
                prompt = base_prompt
            elif attempt == 2:
                prompt = base_prompt + f"""

⚠️ REMINDER: This is attempt {attempt} of {max_attempts}.
You MUST provide at least 3 complete scenarios with all required fields.
Each scenario must have:
- Clear, specific scenario_text
- Proper scenario_type
- Identified involved_components
- Priority level
- Stakeholder_role

Do NOT provide empty or placeholder scenarios.
"""
            else:  # Final attempt
                prompt = base_prompt + f"""

🚨 FINAL ATTEMPT ({attempt} of {max_attempts}) - CONSENSUS REQUIRED 🚨

This is your FINAL CHANCE to generate meaningful scenarios.
You MUST provide at least 3 complete, evaluable scenarios NOW.

If you fail to provide adequate scenarios, the evaluation cannot proceed.
"""
            
            try:
                response = self._call_llm_with_logging(
                    role_name="SAAMManager",
                    task_description=f"Formalizing stakeholder scenarios (Attempt {attempt}/{max_attempts})",
                    prompt=prompt
                )
                
                result = self._extract_json(response)
                scenarios = result.get('scenarios', [])
                
                # Validate that we have meaningful scenarios
                if len(scenarios) >= 2:  # At least 2 scenarios needed
                    valid_scenarios = [s for s in scenarios if 
                                     s.get('scenario_text', '').strip() and 
                                     s.get('scenario_type', '').strip()]
                    
                    if len(valid_scenarios) >= 2:
                        self.logger.log(f"  ✓ Generated {len(scenarios)} scenarios on attempt {attempt}/{max_attempts}")
                        break
                
                self.logger.log(f"  ⚠️ Attempt {attempt}/{max_attempts}: Insufficient scenarios ({len(scenarios)} generated)")
                if attempt == max_attempts:
                    self.logger.log("  ⚠️ WARNING: Maximum attempts reached. Using best-effort scenarios.")
                    
            except Exception as e:
                self.logger.log(f"  ✗ Error on attempt {attempt}/{max_attempts}: {str(e)}")
                if attempt == max_attempts:
                    result = {
                        "scenarios": [],
                        "notes": f"Failed to generate scenarios after {max_attempts} attempts"
                    }
        
        # Log reasoning
        self.logger.log_reasoning(
            agent_role="SAAMManager",
            task="Scenario Elicitation Workshop",
            thought_process=f"Formalized {len(result.get('scenarios', []))} scenarios from stakeholder statements. {result.get('notes', '')}"
        )
        
        # After generating scenarios, request approval from Architect and Customer
        approval = {"architect_approved": False, "customer_approved": False, "notes": []}
        try:
            if getattr(self, 'customer', None) is not None:
                # Ask customer to confirm scenario alignment with business goals
                cust_q = f"Please review the following generated scenarios and indicate acceptance or provide corrections as JSON: {json.dumps(result.get('scenarios', []), indent=2)}"
                cust_resp = self.customer.answer_query(cust_q, caller="SAAMManager")
                try:
                    parsed = json.loads(cust_resp)
                    if parsed.get('accepted') is True or parsed.get('accept') is True:
                        approval['customer_approved'] = True
                    else:
                        approval['notes'].append({'customer': parsed})
                except Exception:
                    approval['notes'].append({'customer_raw': cust_resp})

        except Exception:
            self.logger.log_interaction(sender="Manager", recipient="Customer", content="Customer approval step failed", interaction_type="ERROR")

        # Architect approval can be requested via a simplified check if architect reference is provided
        # Note: The evaluation orchestrator (SAAMEvaluationTeam) may call Architect for deeper checks.
        result['approval'] = approval
        return result
    
    def present_evaluation_to_stakeholders(self, scenarios: List[Dict], evaluations: List[Dict],
                                         classifications: List[Dict], architecture: Dict) -> Dict:
        """
        PHASE 3 (Feedback Loop): Present architectural evaluation findings to stakeholders
        
        The Manager translates technical evaluation results into stakeholder-friendly feedback
        and gathers validation/refinement feedback.
        """
        self.logger.log("  [Manager→Stakeholders]: Presenting evaluation findings and gathering feedback...")
        
        prompt = f"""
CONTEXT - You are a SAAM Facilitator presenting evaluation findings to stakeholders
You have evaluated stakeholder scenarios against the architecture. Now you must:
1. Translate technical findings into stakeholder-understandable language
2. Identify areas of concern (scenarios requiring significant changes)
3. Highlight areas of strength (scenarios well-supported by architecture)
4. Solicit stakeholder feedback on scenario relevance and priority

SCENARIOS EVALUATED:
{json.dumps(scenarios, indent=2)}

ARCHITECTURAL EVALUATION RESULTS:
{json.dumps(evaluations, indent=2)}

CLASSIFICATIONS:
{json.dumps(classifications, indent=2)}

TASK - Generate stakeholder feedback and validation items

For each scenario:
1. Determine if evaluation findings are acceptable to stakeholders
2. Identify if scenarios need refinement or re-prioritization
3. Flag any surprises or misalignments between expectations and architecture

Output JSON:
{{
    "feedback": [
        {{
            "scenario_id": "S-01",
            "concern": "Stakeholder-friendly summary of evaluation findings",
            "severity": "High | Medium | Low",
            "action_required": "Refinement | Re-prioritization | Acceptance",
            "stakeholder_validation": "Is this finding acceptable?"
        }}
    ],
    "overall_sentiment": "Summary of stakeholder satisfaction with evaluation findings"
}}
        """
        response = self._call_llm_with_logging(
            role_name="SAAMManager",
            task_description="Presenting evaluation findings to stakeholders for validation",
            prompt=prompt
        )
        
        result = self._extract_json(response)
        
        # Log reasoning
        self.logger.log_reasoning(
            agent_role="SAAMManager",
            task="Stakeholder Feedback Collection",
            thought_process=f"Gathered feedback on {len(result.get('feedback', []))} evaluation items. Sentiment: {result.get('overall_sentiment', 'N/A')}"
        )
        
        return result

    def review_classifications(self, classifications: List[Dict], architecture: Dict = None) -> Dict:
        """
        Manager reviews Architect classifications and attempts to reach agreement.
        Returns a simple agreement report; if disagreements are detected, provides notes.
        """
        self.logger.log("  [Manager→LLM]: Reviewing classifications for agreement with stakeholder priorities...")
        prompt = f"""
You are the SAAM Manager. Review the following scenario classifications produced by the Architect and indicate whether you, as the facilitator, agree with them. If you disagree, explain why and propose adjustments.

CLASSIFICATIONS:
{json.dumps(classifications, indent=2)}

ARCHITECTURE_CONTEXT:
{json.dumps(architecture, indent=2) if architecture else 'Not provided'}

Output JSON:
{{
  "agreement": true | false,
  "issues": [{{"scenario_id": "S-01", "note": "Why manager disagrees"}}]
}}
"""
        resp = self._call_llm_with_logging(
            role_name="SAAMManager",
            task_description="Reviewing architect classifications for agreement",
            prompt=prompt
        )
        try:
            parsed = self._extract_json(resp)
            return parsed
        except Exception:
            return {"agreement": False, "issues": [{"note": "Failed to parse manager review response"}]}

    def review_recommendations(self, recommendations: List[Dict], architecture: Dict = None, evaluations: List[Dict] = None) -> Dict:
        """
        Manager facilitates stakeholder agreement on recommendations. Returns approval status and notes.
        """
        self.logger.log("  [Manager→LLM]: Reviewing recommendations for stakeholder agreement...")
        prompt = f"""
You are the SAAM Manager. Review the following recommendations and indicate whether stakeholders (Customer, Architect, Manager) would agree. Provide JSON with approval booleans and any requested changes.

RECOMMENDATIONS:
{json.dumps(recommendations, indent=2)}

ARCHITECTURE_CONTEXT:
{json.dumps(architecture, indent=2) if architecture else 'Not provided'}

EVALUATIONS:
{json.dumps(evaluations, indent=2) if evaluations else 'Not provided'}

Output JSON:
{{
  "customer_approved": true | false,
  "architect_approved": true | false,
  "manager_approved": true | false,
  "notes": [{{"who": "Customer|Architect|Manager", "note": "Text"}}]
}}
"""
        resp = self._call_llm_with_logging(
            role_name="SAAMManager",
            task_description="Reviewing recommendations for stakeholder agreement",
            prompt=prompt
        )
        try:
            parsed = self._extract_json(resp)
            # Optionally check Customer factual acceptance via Customer agent
            if getattr(self, 'customer', None) is not None:
                try:
                    cust_q = f"Please confirm whether the following recommendations align with business goals and factual inputs (return JSON with 'accepted': true/false and optional notes): {json.dumps(recommendations, indent=2)}"
                    cust_resp = self.customer.answer_query(cust_q, caller="SAAMManager")
                    cust_parsed = None
                    try:
                        cust_parsed = json.loads(cust_resp)
                    except Exception:
                        cust_parsed = None
                    if isinstance(cust_parsed, dict):
                        parsed['customer_approved'] = bool(cust_parsed.get('accepted') or cust_parsed.get('accept') or parsed.get('customer_approved'))
                        if cust_parsed.get('notes'):
                            parsed.setdefault('notes', []).append({'who': 'Customer', 'note': cust_parsed.get('notes')})
                except Exception:
                    pass
            return parsed
        except Exception:
            return {"customer_approved": False, "architect_approved": False, "manager_approved": False, "notes": [{"note": "Failed to parse manager recommendations review response"}]}
    
    def prioritize_interactions_with_stakeholders(self, interactions: List[Dict], evaluations: List[Dict]) -> Dict:
        """PHASE 5: Weight scenario interactions/weaknesses with stakeholder input"""
        self.logger.log("  [Manager→Stakeholders]: Prioritizing interaction hotspots with stakeholders...")

        if not interactions:
            prompt = f"""
CONTEXT - You are facilitating a SAAM weighting session.
No interaction hotspots were detected (no components hit by multiple scenarios).

EVALUATION FINDINGS (for context):
{json.dumps(evaluations, indent=2)}

TASK - Since no hotspots exist, provide general advice on prioritization for this architecture evaluation and any potential areas to monitor.

Output JSON:
{{
  "interaction_weights": [],
  "notes": "General prioritization advice for the evaluated architecture"
}}
        """
        else:
            prompt = f"""
CONTEXT - You are facilitating a SAAM weighting session.
Goal: Rank the most critical interaction hotspots (components hit by multiple scenarios) after evaluation.

INTERACTION HOTSPOTS (from impact analysis):
{json.dumps(interactions, indent=2)}

EVALUATION FINDINGS (for context):
{json.dumps(evaluations, indent=2)}

TASK - With stakeholder perspective, assign weights/priorities to the interactions:
1. Rank by business/operational risk if not addressed.
2. Consider which stakeholder roles are most impacted (End-user, Developer, Maintainer, System Administrator).
3. Provide rationale that ties back to evaluated scenarios and affected components.

Output JSON:
{{
  "interaction_weights": [
    {{
      "component": "Component Name",
      "scenario_ids": ["S-01", "S-02"],
      "priority": "High | Medium | Low",
      "rank": 1,
      "impacted_roles": ["End-user", "Developer"],
      "rationale": "Why this hotspot matters and consequences of not addressing it"
    }}
  ],
  "notes": "Optional facilitation notes"
}}
        """

        response = self._call_llm_with_logging(
            role_name="SAAMManager",
            task_description="Prioritizing scenario interactions with stakeholders",
            prompt=prompt
        )

        result = self._extract_json(response)

        self.logger.log_reasoning(
            agent_role="SAAMManager",
            task="Interaction Weighting",
            thought_process=f"Weighted {len(result.get('interaction_weights', []))} interaction hotspots with stakeholder perspective."
        )

        return result


class SAAMArchitect(BaseRole):
    """
    Role: SAAM Architect / Evaluator
    
    Responsibilities:
    ─────────────────
    1. ARCHITECTURE PRESENTATION: Present architecture to stakeholders in Phase 1
    2. SCENARIO EVALUATION: Assess scenarios against architecture (Phase 3)
    3. IMPACT ANALYSIS: Identify affected components and change requirements
    4. SYNTHESIS: Aggregate findings into architectural assessment
    
    Key Inputs:
    - System overview and business goals
    - Architectural decisions and component descriptions
    - Formal scenarios (architecture-aware, from Manager)
    - Stakeholder feedback on evaluation findings
    
    Key Outputs:
    - Architecture description (presentation)
    - Scenario classifications (Direct vs. Indirect)
    - Impact analysis (which components change, effort estimates)
    - Overall architectural assessment
    """

    def describe_architecture(self, overview: Dict, decisions: List[Dict]) -> Dict:
        """
        PHASE 1: Architecture Presentation
        
        The Architect presents the architecture to stakeholders in a clear, 
        comprehensible way that informs scenario generation.
        """
        self.logger.log("  [Architect→Stakeholders]: Presenting architecture for stakeholder understanding...")
        
        prompt = f"""
CONTEXT - You are a Senior Architect presenting the system architecture
Your audience: Stakeholders who will use this information to assess scenarios
Your goal: Make the architecture clear, accessible, and evaluable

SYSTEM BEING EVALUATED:
Name: {overview.get('system_name')}
Description: {overview.get('system_description')}
Business Goals: {overview.get('business_goals')}

ARCHITECTURAL DECISIONS/COMPONENTS (From documentation):
{json.dumps(decisions, indent=2)}

TASK - Create a clear, stakeholder-accessible architecture description
1. Identify architectural style(s) (e.g., Layered, Microservices, Client-Server, Event-Driven)
2. Define the key components and their responsibilities
3. Describe data flow and dependencies between components
4. Highlight architectural constraints and quality characteristics
5. Make explicit what the architecture CAN and CANNOT easily accommodate
The description should be detailed enough for stakeholders to understand:
- What functionality is currently supported
- Which components would be affected by changes
- How difficult changes might be (in general terms)

Output JSON:
{{
    "architecture_styles": ["Style1", "Style2"],
    "architecture_details": {{
        "architecture_id": "A-01",
        "name": "Current Production Architecture",
        "description": "High-level description of architecture",
        "architecture_characteristics": "Key qualities (scalability, modularity, etc.)",
        "key_components": [
            {{
                "component_id": "C-01",
                "component_name": "Component Name",
                "responsibilities": "What this component does",
                "dependencies": ["C-02", "C-03"],
                "change_sensitivity": "High | Medium | Low"
            }}
        ],
        "data_flow": "Description of how data flows through components",
        "architectural_constraints": ["Constraint 1", "Constraint 2"],
        "notes": "Any additional architectural considerations"
    }}
}}
        """
        response = self._call_llm_with_logging(
            role_name="SAAMArchitect",
            task_description="Describing architecture for stakeholder presentation",
            prompt=prompt
        )

        result = self._extract_json(response)

        # If a Customer agent is available, perform a short multi-turn verification
        # loop where the Architect asks the Customer for factual confirmation and
        # possible missing components/constraints, then refines the description.
        if getattr(self, 'customer', None) is not None:
            try:
                arch_details = result.get('architecture_details', {})
                # Ask the customer to confirm or provide missing facts
                cust_q = f"Please review the following architecture summary and return any additional components or corrections as JSON. ArchitectureSummary: {json.dumps(arch_details, indent=2)}"
                cust_resp = self.customer.answer_query(cust_q, caller="SAAMManager")
                try:
                    parsed = json.loads(cust_resp)
                except Exception:
                    parsed = None

                # If customer provided additional components, merge them
                if isinstance(parsed, dict):
                    additional = parsed.get('additional_components') or parsed.get('components')
                    corrections = parsed.get('corrections')
                    if isinstance(additional, list) and additional:
                        existing = arch_details.get('key_components', [])
                        # Convert simple names to component dicts if needed
                        for a in additional:
                            if isinstance(a, str):
                                existing.append({
                                    'component_id': f"C-{len(existing)+1:02d}",
                                    'component_name': a,
                                    'responsibilities': 'Added from Customer confirmation',
                                    'dependencies': [],
                                    'change_sensitivity': 'Unknown'
                                })
                        arch_details['key_components'] = existing
                        result['architecture_details'] = arch_details

                    # If corrections exist, append to notes for LLM refinement
                    if corrections:
                        arch_details.setdefault('notes', '')
                        arch_details['notes'] = arch_details.get('notes', '') + '\nCustomer Corrections: ' + str(corrections)
                        result['architecture_details'] = arch_details

                    # Re-run the architect LLM once with the updated architecture details
                    refine_prompt = prompt + "\n\nREVISED_ARCHITECTURE_CONTEXT:\n" + json.dumps(result.get('architecture_details', {}), indent=2)
                    refine_resp = self._call_llm_with_logging(
                        role_name="SAAMArchitect",
                        task_description="Describing architecture for stakeholder presentation",
                        prompt=refine_prompt
                    )
                    try:
                        refined = self._extract_json(refine_resp)
                        # Prefer refined architecture_details when available
                        if refined.get('architecture_details'):
                            result = refined
                    except Exception:
                        # If parsing fails, keep the previous result
                        pass
            except Exception:
                # Do not allow customer-check failures to block the flow
                self.logger.log_interaction(sender="Architect", recipient="Customer", content="Customer verification step failed", interaction_type="ERROR")
        
        # Log reasoning
        arch_details = result.get('architecture_details', {})
        self.logger.log_reasoning(
            agent_role="SAAMArchitect",
            task="Architecture Description",
            thought_process=f"Identified {len(result.get('architecture_styles', []))} architectural styles and {len(arch_details.get('key_components', []))} key components for system: {arch_details.get('name', 'Unknown')}"
        )
        
        return result
    
    def analyze_scenario_interactions(self, evaluations: List[Dict]) -> Dict:
        """
        SAAM STEP 4: Reveal Scenario Interaction
        Count how many scenarios affect each component to identify high-coupling areas.
        """
        interaction_map = {}
        for evaluation in evaluations:
            for component in evaluation.get("affected_components", []):
                if component not in interaction_map:
                    interaction_map[component] = []
                interaction_map[component].append(evaluation.get("scenario_id"))

        # Format for template
        interactions = [
            {"component": k, "scenario_count": len(v), "scenarios": v} 
            for k, v in interaction_map.items() 
            if len(v) > 1 # Only interested in interactions (overlaps)
        ]
        return interactions
    def classify_and_evaluate_scenarios(self, scenarios: List[Dict], architecture: Dict,
                                       business_goals: List[str] = None) -> Dict:
        """
        PHASE 3: Scenario Classification & Impact Analysis
        
        The Architect evaluates each scenario against the architecture:
        1. Classification: Can the architecture support this scenario directly, or does it require changes?
        2. Impact Analysis: What changes are needed? Which components? How much effort?
        3. Feasibility Assessment: Is this achievable? What's the architectural risk?
        """
        self.logger.log("  [Architect→Analysis]: Classifying scenarios and analyzing architectural impact...")
        
        arch_summary = json.dumps(architecture, indent=2)
        scen_summary = json.dumps(scenarios, indent=2)
        goals_summary = ", ".join(business_goals) if business_goals else "Not specified"
        
        base_prompt = f"""
CONTEXT - You are a SAAM Architect evaluating scenarios against the architecture.

Task: Assess whether scenarios can be supported by the current architecture
       and identify what changes (if any) would be required.

CURRENT ARCHITECTURE:
{arch_summary}

SCENARIOS TO EVALUATE:
{scen_summary}

TASK - Classify scenarios and analyze architectural impact

For EACH scenario:

1. CLASSIFICATION:
   - "Direct": Architecture already supports this scenario with minimal or no changes
   - "Indirect": Scenario requires architectural changes or significant workarounds

2. RATIONALE:
   - For Direct: Explain which components/capabilities enable this
   - For Indirect: Explain why changes are needed and what constraints prevent direct support

3. IMPACT ANALYSIS (only for Indirect scenarios):
   - Affected components: Which components must change?
   - Change description: What specific changes are required?
   - Change extent: Is this "Localized" (single component) or "Widespread" (multiple components)?
   - Effort estimate: Low (days) | Medium (weeks) | High (months+)
   - Risk assessment: What's the architectural risk? (Low/Medium/High)
   - Issues identified: Specific problems or concerns

4. ALTERNATIVE APPROACHES:
   - For challenging scenarios, suggest alternative ways to achieve the goal within current architecture

Output JSON:
{{
    "scenario_classification": [
        {{
            "scenario_id": "S-01",
            "classification": "Direct | Indirect",
            "rationale": "Explanation of classification decision",
            "supporting_components": ["C-01", "C-02"],
            "confidence": "High | Medium | Low"
        }}
    ],
    "scenario_evaluations": [
        {{
            "scenario_id": "S-01",
            "architecture_id": "A-01",
            "classification": "Direct | Indirect",
            "affected_components": ["C-01", "C-02"],
            "change_description": "Specific changes required",
            "change_extent": "Localized | Widespread | None",
            "estimated_effort": "Low | Medium | High",
            "risk_level": "Low | Medium | High",
            "issues_identified": ["Issue 1", "Issue 2"],
            "alternative_approaches": "Alternative ways to achieve scenario goal"
        }}
    ]
}}"""
        
        # FORCED CONSENSUS MECHANISM - ensure complete classification and evaluation
        max_attempts = 3
        result = {}
        
        for attempt in range(1, max_attempts + 1):
            if attempt == 1:
                prompt = base_prompt
            elif attempt == 2:
                prompt = base_prompt + f"""

⚠️ REMINDER: This is attempt {attempt} of {max_attempts}.
You MUST classify and evaluate ALL {len(scenarios)} scenarios.
Each scenario needs:
- Complete classification with rationale
- Complete evaluation with affected components and impact analysis

Do NOT skip scenarios or provide incomplete analyses.
"""
            else:  # Final attempt
                prompt = base_prompt + f"""

🚨 FINAL ATTEMPT ({attempt} of {max_attempts}) - CONSENSUS REQUIRED 🚨

This is your FINAL CHANCE to classify and evaluate ALL scenarios.
You MUST provide complete classification and evaluation for ALL {len(scenarios)} scenarios NOW.

Without complete scenario evaluations, the synthesis phase cannot proceed.
"""
            
            try:
                response = self._call_llm_with_logging(
                    role_name="SAAMArchitect",
                    task_description=f"Classifying scenarios and analyzing impact (Attempt {attempt}/{max_attempts})",
                    prompt=prompt
                )
                
                result = self._extract_json(response)
                
                classifications = result.get('scenario_classification', [])
                evaluations = result.get('scenario_evaluations', [])
                
                # Validate completeness - should have classification and evaluation for each scenario
                expected_count = len(scenarios)
                if len(classifications) >= expected_count and len(evaluations) >= expected_count:
                    self.logger.log(f"  ✓ Classified and evaluated {len(classifications)} scenarios on attempt {attempt}/{max_attempts}")
                    break
                else:
                    self.logger.log(f"  ⚠️ Attempt {attempt}/{max_attempts}: Incomplete analysis. Expected {expected_count}, got {len(classifications)} classifications and {len(evaluations)} evaluations")
                    if attempt == max_attempts:
                        self.logger.log("  ⚠️ WARNING: Maximum attempts reached. Using partial results.")
                        
            except Exception as e:
                self.logger.log(f"  ✗ Error on attempt {attempt}/{max_attempts}: {str(e)}")
                if attempt == max_attempts:
                    result = {
                        "scenario_classification": [],
                        "scenario_evaluations": []
                    }
        
        # Log reasoning
        classifications = result.get('scenario_classification', [])
        evaluations = result.get('scenario_evaluations', [])
        direct_count = sum(1 for c in classifications if c.get('classification') == 'Direct')
        indirect_count = len(classifications) - direct_count
        
        self.logger.log_reasoning(
            agent_role="SAAMArchitect",
            task="Scenario Classification & Impact Analysis",
            thought_process=f"Classified {len(classifications)} scenarios: {direct_count} Direct, {indirect_count} Indirect. Generated {len(evaluations)} impact analyses."
        )
        
        return result
    
    def generate_overall_results(self, evaluations: List[Dict], stakeholder_feedback: List[Dict] = None,
                                business_goals: List[str] = None, interactions: List[Dict] = None,
                                interaction_weights: List[Dict] = None) -> Dict:
        """
        PHASE 4: Synthesis & Overall Assessment
        
        Aggregate scenario evaluations into:
        1. Architectural strengths (well-supported scenarios)
        2. Architectural weaknesses (problematic scenarios)
        3. Modifiability assessment (how easy/hard to support future needs?)
        4. Recommendations for improvement
        
        *** INCLUDES FORCED CONSENSUS MECHANISM ***
        Similar to ATAM, this method will retry up to max_attempts to ensure
        all required fields are populated with meaningful content.
        """
        self.logger.log("  [Architect→Synthesis]: Synthesizing evaluation findings into architectural assessment...")
        
        evals_summary = json.dumps(evaluations, indent=2)
        feedback_summary = json.dumps(stakeholder_feedback or []) if stakeholder_feedback else "No stakeholder feedback"
        goals_summary = ", ".join(business_goals) if business_goals else "Not specified"
        interactions_summary = json.dumps(interactions or [], indent=2)
        weights_summary = json.dumps(interaction_weights or [], indent=2)
        
        base_prompt = f"""
CONTEXT - You are a Senior Architect synthesizing evaluation findings

Task: Aggregate scenario evaluations into an overall architectural assessment
       considering business goals and stakeholder feedback.

SCENARIO EVALUATIONS (FROM IMPACT ANALYSIS):
{evals_summary}

STAKEHOLDER FEEDBACK (IF PROVIDED):
{feedback_summary}

SCENARIO INTERACTIONS (HOTSPOTS):
{interactions_summary}

WEIGHTING / PRIORITIZATION (Stakeholder-ranked hotspots):
{weights_summary}

TASK - Generate overall architectural assessment

1. ARCHITECTURAL STRENGTHS:
   - What does the architecture handle well?
   - Which scenarios are directly supported with confidence?
   - What architectural decisions enable good support?

2. ARCHITECTURAL WEAKNESSES:
   - Where does the architecture struggle?
   - Which scenarios require significant changes?
   - What architectural limitations are exposed?

3. MODIFIABILITY ASSESSMENT:
   - How easily can the architecture accommodate future changes?
   - Is it modular and flexible? Or rigid and coupled?
   - What's the overall assessment of architectural evolvability?

4. RECOMMENDATIONS:
   - How should the architecture evolve to better support stakeholder needs?
   - Are there high-priority refactorings or re-architectures needed?
   - What architectural patterns would improve modifiability?
   - Are there quick wins vs. long-term improvements?

Output JSON:
{{
    "architectural_strengths": [
        "Strength 1: Description with evidence from evaluations",
        "Strength 2: ...",
    ],
    "architectural_weaknesses": [
        "Weakness 1: Description with impact evidence",
        "Weakness 2: ...",
    ],
    "modifiability_assessment": "Overall assessment (1-2 sentences) of how well the architecture supports future changes",
    "recommendations": [
        {{
            "priority": "High | Medium | Low",
            "recommendation": "Specific recommendation for architectural improvement",
            "rationale": "Why this improves modifiability",
            "estimated_effort": "Low | Medium | High"
        }}
    ]
}}

CRITICAL GUIDANCE:
- Ground all assessments in the specific evaluations performed
- Consider stakeholder feedback in final recommendations
- Prioritize recommendations by impact on business goals and modifiability
- Be realistic about effort and feasibility
        """
        
        # FORCED CONSENSUS MECHANISM (similar to ATAM)
        max_attempts = 3
        result = {}
        
        for attempt in range(1, max_attempts + 1):
            # Add increasingly strong reminders on subsequent attempts
            if attempt == 1:
                prompt = base_prompt
            elif attempt == 2:
                prompt = base_prompt + f"""

⚠️ REMINDER: This is attempt {attempt} of {max_attempts}.
You MUST provide complete, non-empty outputs for ALL fields:
- architectural_strengths (at least 2 items)
- architectural_weaknesses (at least 2 items)
- modifiability_assessment (detailed paragraph)
- recommendations (at least 2 items with complete details)

Do NOT leave any fields empty or with placeholder text.
"""
            else:  # Final attempt
                prompt = base_prompt + f"""

🚨 FINAL ATTEMPT ({attempt} of {max_attempts}) - CONSENSUS REQUIRED 🚨

You MUST reach consensus and provide COMPLETE outputs now.
This is your FINAL CHANCE to provide comprehensive architectural assessment.

MANDATORY REQUIREMENTS:
✓ architectural_strengths: At least 2 detailed strengths with evidence
✓ architectural_weaknesses: At least 2 detailed weaknesses with impact analysis  
✓ modifiability_assessment: Comprehensive 2-3 sentence assessment
✓ recommendations: At least 2 detailed recommendations with priority, rationale, and effort

If you provide incomplete or empty outputs, the evaluation will be considered FAILED.
The team expects your professional synthesis NOW.
"""
            
            try:
                response = self._call_llm_with_logging(
                    role_name="SAAMArchitect",
                    task_description=f"Synthesizing evaluation findings (Attempt {attempt}/{max_attempts})",
                    prompt=prompt
                )
                
                result = self._extract_json(response)
                
                # Validate completeness
                strengths = result.get('architectural_strengths', [])
                weaknesses = result.get('architectural_weaknesses', [])
                assessment = result.get('modifiability_assessment', '')
                recommendations = result.get('recommendations', [])
                
                # Check if all fields are meaningfully populated
                is_complete = (
                    len(strengths) >= 1 and
                    len(weaknesses) >= 1 and
                    len(assessment.strip()) > 20 and  # At least a meaningful sentence
                    len(recommendations) >= 1
                )
                
                if is_complete:
                    self.logger.log(f"  ✓ Synthesis complete on attempt {attempt}/{max_attempts}")
                    self.logger.log_reasoning(
                        agent_role="SAAMArchitect",
                        task="Overall Results Synthesis",
                        thought_process=f"Successfully generated complete synthesis on attempt {attempt}. Strengths: {len(strengths)}, Weaknesses: {len(weaknesses)}, Recommendations: {len(recommendations)}. Assessment: {assessment[:100]}"
                    )
                    return result
                else:
                    missing = []
                    if len(strengths) < 1:
                        missing.append("architectural_strengths")
                    if len(weaknesses) < 1:
                        missing.append("architectural_weaknesses")
                    if len(assessment.strip()) <= 20:
                        missing.append("modifiability_assessment")
                    if len(recommendations) < 1:
                        missing.append("recommendations")
                    
                    self.logger.log(f"  ⚠️ Attempt {attempt}/{max_attempts}: Incomplete outputs. Missing or insufficient: {', '.join(missing)}")
                    
                    if attempt < max_attempts:
                        continue  # Try again
                    else:
                        self.logger.log(f"  ⚠️ WARNING: Maximum attempts reached. Returning best-effort results.")
                        # Return what we have, even if incomplete
                        return result
                        
            except Exception as e:
                self.logger.log(f"  ✗ Error on attempt {attempt}/{max_attempts}: {str(e)}")
                if attempt == max_attempts:
                    # Last attempt failed - return minimal structure
                    self.logger.log("  ⚠️ All attempts failed. Returning minimal structure.")
                    return {
                        "architectural_strengths": ["Analysis incomplete due to technical error"],
                        "architectural_weaknesses": ["Analysis incomplete due to technical error"],
                        "modifiability_assessment": "Unable to complete full assessment due to technical error",
                        "recommendations": []
                    }
        
        # Should not reach here, but return result if we do
        return result