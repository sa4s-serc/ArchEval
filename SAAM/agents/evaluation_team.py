"""
SAAM Team Agent - Orchestrator
Implements the tight coupling of Phases 1-3 (Presentation → Scenario Generation → Evaluation)
with stakeholder accessibility to all evaluation inputs throughout the process.
"""
from roles import SAAMManager, SAAMArchitect
import json
from typing import Dict, List

class SAAMEvaluationTeam:
    
    def __init__(self, llm_client, customer_agent, saam_template, logger, tracker=None, router=None):
        self.llm = llm_client
        self.customer = customer_agent
        self.template = saam_template
        self.logger = logger
        self.tracker = tracker
        self.external_router = router

        # Pass tracker and customer into role instances so they can log metrics
        # and request clarifications from the Customer agent when needed.
        self.manager = SAAMManager(llm_client, logger, tracker, customer_agent=customer_agent)
        self.architect = SAAMArchitect(llm_client, logger, tracker, customer_agent=customer_agent)

        # State tracking for iterative phases
        self.current_architecture = None
        self.current_scenarios = []
        self.evaluation_feedback = []

    def _route(self, sender: str, recipient: str, method_name: str, *args, tracking_label: str = None, **kwargs):
        """
        Simple in-team router: logs the routed interaction and invokes the recipient's method.
        Keeps messaging explicit without changing phase logic or method signatures.
        """
        # Log the routed interaction
        content = f"Routing call `{method_name}` from {sender} to {recipient}`"
        if tracking_label:
            content = f"[{tracking_label}] " + content

        try:
            self.logger.log_interaction(sender=sender, recipient=recipient, content=content, interaction_type="ROUTE")
        except Exception:
            # Fallback to plain log
            self.logger.log(f"[ROUTE] {sender} -> {recipient}: {method_name}")

        # Resolve recipient object
        mapping = {
            "SAAMManager": self.manager,
            "SAAMArchitect": self.architect,
            "CustomerAgent": self.customer,
            "SAAMEvaluationTeam": self
        }

        target = mapping.get(recipient)
        if target is None:
            # Unknown recipient; log and raise
            self.logger.log_interaction(sender="Router", recipient=sender, content=f"Unknown recipient: {recipient}", interaction_type="ERROR")
            raise ValueError(f"Unknown routing recipient: {recipient}")

        # Invoke the method on the target if available
        method = getattr(target, method_name, None)
        if not callable(method):
            self.logger.log_interaction(sender="Router", recipient=recipient, content=f"Recipient has no method {method_name}", interaction_type="ERROR")
            raise AttributeError(f"{recipient} has no method {method_name}")

        return method(*args, **kwargs)

    # --- New modular phase methods so SAAMEvaluationTeam can act as a participant agent ---
    def present_architecture(self, overview: Dict, goals: List[str], decisions: List[Dict]) -> Dict:
        """Phase 1: Architect presents architecture to stakeholders. Returns architecture data."""
        self.logger.log("  [Architect→Stakeholders]: Presenting architecture description...")
        arch_data = self._route(sender="SAAMEvaluationTeam", recipient="SAAMArchitect", method_name="describe_architecture", overview=overview, decisions=decisions)
        architecture_details = arch_data.get("architecture_details", {})
        self.current_architecture = architecture_details

        self.template.set_system_context(
            overview.get("system_name"),
            overview.get("description"),
            goals
        )
        self.template.set_architecture_description(
            arch_data.get("architecture_styles", []),
            architecture_details
        )

        self.logger.log(f"  ✓ Architecture Styles: {arch_data.get('architecture_styles')}")
        self.logger.log(f"  ✓ Key Components: {len(architecture_details.get('key_components', []))} identified")

        return arch_data

    def generate_and_formalize_scenarios(self, overview: Dict, architecture_details: Dict, stakeholder_needs: List[str]) -> Dict:
        """Phase 2: Manager formalizes scenarios using architecture context."""
        self.logger.log("  [PHASE 2] Step 2: Manager formalizes scenarios (architecture-aware)f....")
        elicitation_results = self._route(
            sender="SAAMEvaluationTeam",
            recipient="SAAMManager",
            method_name="conduct_elicitation_workshop",
            system_context=overview,
            architecture_context=architecture_details,
            stakeholder_statements=stakeholder_needs
        )
        formal_scenarios = elicitation_results.get("scenarios", [])
        for s in formal_scenarios:
            self.template.add_scenario(s)
        self.current_scenarios = formal_scenarios
        return elicitation_results

    def classify_and_evaluate(self, formal_scenarios: List[Dict], architecture_details: Dict, goals: List[str]) -> Dict:
        """Phase 3: Architect evaluates scenarios against architecture."""
        self.logger.log("  [PHASE 3] Step 1: Architect evaluates scenarios against architecture...")
        eval_results = self._route(
            sender="SAAMEvaluationTeam",
            recipient="SAAMArchitect",
            method_name="classify_and_evaluate_scenarios",
            scenarios=formal_scenarios,
            architecture=architecture_details,
            business_goals=goals
        )

        # Store classifications and evaluations in the template
        for c in eval_results.get("scenario_classification", []):
            self.template.add_classification(c)
        for e in eval_results.get("scenario_evaluations", []):
            self.template.add_evaluation(e)

        return eval_results

    def synthesize_and_recommend(self, eval_results: Dict, stakeholder_feedback: List[Dict], goals: List[str], interactions: List[Dict], interaction_weights: List[Dict]) -> Dict:
        """Phase 4: Synthesis and recommendations by Architect."""
        self.logger.log_phase("PHASE 4: SYNTHESIS & RECOMMENDATIONS")
        
        # Attempt synthesis with validation loop
        max_attempts = 3
        synthesis = {}
        
        for attempt in range(1, max_attempts + 1):
            synthesis = self._route(
                sender="SAAMEvaluationTeam",
                recipient="SAAMArchitect",
                method_name="generate_overall_results",
                evaluations=eval_results.get("scenario_evaluations", []),
                stakeholder_feedback=stakeholder_feedback,
                business_goals=goals,
                interactions=interactions,
                interaction_weights=interaction_weights
            )
            
            # Validate that synthesis has meaningful content
            strengths = synthesis.get("architectural_strengths", [])
            weaknesses = synthesis.get("architectural_weaknesses", [])
            assessment = synthesis.get("modifiability_assessment", "")
            recommendations = synthesis.get("recommendations", [])
            
            is_valid = (
                len(strengths) >= 1 and
                len(weaknesses) >= 1 and
                len(assessment.strip()) > 20 and
                len(recommendations) >= 1
            )
            
            if is_valid:
                self.logger.log(f"  ✓ Synthesis validated successfully on attempt {attempt}")
                break
            else:
                missing_fields = []
                if len(strengths) < 1:
                    missing_fields.append("architectural_strengths")
                if len(weaknesses) < 1:
                    missing_fields.append("architectural_weaknesses")
                if len(assessment.strip()) <= 20:
                    missing_fields.append("modifiability_assessment")
                if len(recommendations) < 1:
                    missing_fields.append("recommendations")
                
                self.logger.log(f"  ⚠️ Synthesis validation failed on attempt {attempt}/{max_attempts}. Incomplete: {', '.join(missing_fields)}")
                
                if attempt == max_attempts:
                    self.logger.log("  ⚠️ WARNING: Final synthesis still incomplete after maximum attempts. Using best-effort results.")

        self.template.set_overall_results(
            synthesis.get("architectural_strengths", []),
            synthesis.get("architectural_weaknesses", []),
            synthesis.get("modifiability_assessment", "Pending")
        )
        for rec in synthesis.get("recommendations", []):
            self.template.add_recommendation(rec)
        return synthesis

    def assess_phase(self, phase_name: str, summary: Dict) -> Dict:
        """Ask the SAAM Evaluation Team agent (this object) to agree/disagree on a phase output.
        Returns JSON: {"agreement": bool, "notes": "..."}
        """
        # Build a lightweight prompt for agreement
        try:
            prompt = f"Phase: {phase_name}\nSummary:\n{json.dumps(summary, indent=2)}\n\nRespond with JSON: {{'agreement': true | false, 'notes': '...' }}"
            # Log assessment request
            self.logger.log_interaction(sender="Orchestrator", recipient="SAAMEvaluationTeam", content=f"Requesting agreement for {phase_name}", interaction_type="QUERY")
            start = __import__('time').time()
            resp = self.llm.query(prompt)
            duration = __import__('time').time() - start
            resp_text = resp[0] if isinstance(resp, tuple) else (resp.text if hasattr(resp, 'text') else str(resp))
            # Log LLM call
            try:
                self.logger.log_llm_call(role="SAAMEvaluationTeam", prompt_summary=f"Assess {phase_name}", response_length=len(resp_text), duration=duration)
            except Exception:
                pass

            # Track efficiency metrics for the team agent
            try:
                if self.tracker is not None:
                    input_tokens = len(prompt.split())
                    output_tokens = len(resp_text.split())
                    self.tracker.log_metric(task_label=f"Assess {phase_name}", agent_name="SAAMEvaluationTeam",
                                            input_tokens=input_tokens, output_tokens=output_tokens,
                                            time_taken=duration)
            except Exception:
                pass

            # Extract JSON using robust parsing (handles markdown code blocks)
            parsed = self._extract_json_from_response(resp_text)
            return parsed
        except Exception as e:
            self.logger.log(f"  ⚠️ Assessment parsing error for {phase_name}: {str(e)}")
            return {"agreement": False, "notes": f"Assessment error: {e}"}
    
    def _extract_json_from_response(self, text: str) -> Dict:
        """Extract JSON from LLM response, handling markdown code blocks"""
        import re
        
        # First try to extract from markdown code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Fallback: find the first complete JSON object
        start = text.find('{')
        if start != -1:
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
                            json_str = text[start:i+1]
                            try:
                                return json.loads(json_str)
                            except json.JSONDecodeError:
                                pass
        
        # If all parsing attempts fail, return a failure response
        raise ValueError(f"No valid JSON found in response. Response preview: {text[:500]}")
        
    def perform_evaluation(self):
        """
        Execute the SAAM Workflow with tightly coupled Phases 1-3.
        
        SAAM Phase Structure (ITERATIVE):
        ─────────────────────────────────
        PHASE 1: PRESENTATION
        - Architect presents system context and architecture to stakeholders
        - All evaluation inputs accessible: goals, architecture, constraints
        - Output: Shared understanding
        
        PHASE 2: SCENARIO GENERATION (with FEEDBACK LOOP)
        - Stakeholders voice needs → Manager formalizes with full architectural context
        - CRITICAL: Scenarios are refined based on architectural feasibility
        - Output: Mutually understood, architecture-aware scenarios
        
        PHASE 3: ARCHITECTURAL EVALUATION
        - Architect evaluates scenarios → Manager presents findings to stakeholders
        - Stakeholders provide feedback on scenario relevance
        - Loop back if scenarios need refinement
        """
        
        # ═══════════════════════════════════════════════════════════
        # PHASE 1: PRESENTATION (Architecture Presented to Stakeholders)
        # ═══════════════════════════════════════════════════════════
        self.logger.log_phase("PHASE 1: PRESENTATION & SHARED UNDERSTANDING")
        
        overview = self.customer.get_system_overview()
        goals = self.customer.get_business_context().get("business_goals", [])
        decisions = self.customer.get_architectural_decisions()
        
        # 1.1: Set system context
        self.template.set_system_context(
            overview.get("system_name"),
            overview.get("description"),
            goals
        )
        self.logger.log(f"  ✓ System Context: {overview.get('system_name')}")
        self.logger.log(f"  ✓ Business Goals: {goals}")
        
        # 1.2: Architect presents architecture to stakeholders
        self.logger.log("  [Architect→Stakeholders]: Presenting architecture description...")
        arch_data = self._route(sender="SAAMEvaluationTeam", recipient="SAAMArchitect", method_name="describe_architecture", overview=overview, decisions=decisions)
        architecture_details = arch_data.get("architecture_details", {})
        self.current_architecture = architecture_details
        
        self.template.set_architecture_description(
            arch_data.get("architecture_styles", []),
            architecture_details
        )
        
        self.logger.log(f"  ✓ Architecture Styles: {arch_data.get('architecture_styles')}")
        self.logger.log(f"  ✓ Key Components: {len(architecture_details.get('key_components', []))} identified")
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # PHASE 2 & 3 (TIGHTLY COUPLED): ITERATIVE SCENARIO GENERATION & EVALUATION
        # ═══════════════════════════════════════════════════════════════════════════════
        self.logger.log_phase("PHASES 2-3: SCENARIO GENERATION & ARCHITECTURAL EVALUATION (ITERATIVE)")
        
        # 2.1: Stakeholders voice needs WITH full architectural context visible
        self.logger.log("  [PHASE 2] Step 1: Stakeholder needs elicitation...")
        stakeholder_needs = self.customer.discuss_stakeholder_needs()
        self.logger.log(f"  ✓ Stakeholders voiced {len(stakeholder_needs)} concerns/needs")
        
        # 2.2: Manager formalizes scenarios WITH architectural context & constraints
        #      This is where the "tight coupling" happens - scenarios are architecture-aware
        self.logger.log("  [PHASE 2] Step 2: Manager formalizes scenarios (architecture-aware)...")
        elicitation_results = self._route(
            sender="SAAMEvaluationTeam",
            recipient="SAAMManager",
            method_name="conduct_elicitation_workshop",
            system_context=overview,
            architecture_context=architecture_details,
            stakeholder_statements=stakeholder_needs
        )
        formal_scenarios = elicitation_results.get("scenarios", [])
        scenario_notes = elicitation_results.get("notes", "")
        approval_info = elicitation_results.get("approval", {})

        # Request an initial architect check of generated scenarios so Manager can get approval
        self.logger.log("  [PHASE 2] Requesting initial Architect review of generated scenarios for approval...")
        pre_eval = self._route(
            sender="SAAMEvaluationTeam",
            recipient="SAAMArchitect",
            method_name="classify_and_evaluate_scenarios",
            scenarios=formal_scenarios,
            architecture=architecture_details,
            business_goals=goals
        )

        # Have Manager review the Architect's classifications for agreement
        review = self._route(
            sender="SAAMEvaluationTeam",
            recipient="SAAMManager",
            method_name="review_classifications",
            classifications=pre_eval.get('scenario_classification', []),
            architecture=architecture_details
        )

        if not review.get('agreement'):
            self.logger.log("  ! Manager and Architect did not reach agreement on classifications; attempting one refinement cycle...")
            # Attempt one refinement: ask Manager to refine scenarios and re-run architect check
            refinement = self._route(
                sender="SAAMEvaluationTeam",
                recipient="SAAMManager",
                method_name="conduct_elicitation_workshop",
                system_context=overview,
                architecture_context=architecture_details,
                stakeholder_statements=stakeholder_needs
            )
            formal_scenarios = refinement.get('scenarios', formal_scenarios)
            # Re-run architect classification
            pre_eval = self._route(
                sender="SAAMEvaluationTeam",
                recipient="SAAMArchitect",
                method_name="classify_and_evaluate_scenarios",
                scenarios=formal_scenarios,
                architecture=architecture_details,
                business_goals=goals
            )
            review = self._route(
                sender="SAAMEvaluationTeam",
                recipient="SAAMManager",
                method_name="review_classifications",
                classifications=pre_eval.get('scenario_classification', []),
                architecture=architecture_details
            )

        # Use the (possibly refined) pre_eval results as the canonical evaluation going into Phase 3
        eval_results = pre_eval
        
        for s in formal_scenarios:
            self.template.add_scenario(s)
        self.current_scenarios = formal_scenarios
        
        self.logger.log(f"  ✓ Manager formalized {len(formal_scenarios)} scenarios")
        if scenario_notes:
            self.logger.log(f"  [Manager Notes]: {scenario_notes}")
        
        # 3.1: Architect evaluates formalized scenarios against architecture
        self.logger.log("  [PHASE 3] Step 1: Architect evaluates scenarios against architecture...")
        # Store classifications and evaluations
        for c in eval_results.get("scenario_classification", []):
            self.template.add_classification(c)
        for e in eval_results.get("scenario_evaluations", []):
            self.template.add_evaluation(e)
        
        self.logger.log(f"  ✓ Classified {len(eval_results.get('scenario_classification', []))} scenarios")
        self.logger.log(f"  ✓ Generated {len(eval_results.get('scenario_evaluations', []))} impact analyses")

        # 3.1b: Reveal scenario interactions (components hit by multiple scenarios)
        interactions = self._route(
            sender="SAAMEvaluationTeam",
            recipient="SAAMArchitect",
            method_name="analyze_scenario_interactions",
            evaluations=eval_results.get("scenario_evaluations", [])
        )
        for interaction in interactions:
            self.template.add_interaction(interaction)
        if interactions:
            self.logger.log(f"  ✓ Identified {len(interactions)} interaction hotspots (components touched by multiple scenarios)")
        else:
            self.logger.log("  ✓ No overlapping component interactions detected")
        
        # 3.2: Manager presents evaluation findings back to stakeholders for feedback
        #      (This enables the tight coupling feedback loop)
        self.logger.log("  [PHASE 3] Step 2: Manager presents findings to stakeholders for validation...")
        stakeholder_feedback = self._route(
            sender="SAAMEvaluationTeam",
            recipient="SAAMManager",
            method_name="present_evaluation_to_stakeholders",
            scenarios=formal_scenarios,
            evaluations=eval_results.get("scenario_evaluations", []),
            classifications=eval_results.get("scenario_classification", []),
            architecture=architecture_details
        )
        self.evaluation_feedback = stakeholder_feedback.get("feedback", [])
        
        if self.evaluation_feedback:
            self.logger.log(f"  ✓ Stakeholders provided feedback on {len(self.evaluation_feedback)} items")
            for fb in self.evaluation_feedback:
                self.logger.log(f"    - {fb.get('concern')}")
        
        # ═══════════════════════════════════════════════════════════
        # PHASE 4: SYNTHESIS (Architecture Assessment)
        # ═══════════════════════════════════════════════════════════
        self.logger.log_phase("PHASE 4: SYNTHESIS & RECOMMENDATIONS")

        # 4.1: Weight interactions/weaknesses with stakeholders (SAAM Step 5)
        interaction_weights = self._route(
            sender="SAAMEvaluationTeam",
            recipient="SAAMManager",
            method_name="prioritize_interactions_with_stakeholders",
            interactions=interactions,
            evaluations=eval_results.get("scenario_evaluations", [])
        )
        for weight in interaction_weights.get("interaction_weights", []):
            self.template.add_interaction_weighting(weight)
        if interaction_weights.get("interaction_weights"):
            self.logger.log(f"  ✓ Weighted {len(interaction_weights.get('interaction_weights', []))} interaction hotspots with stakeholders")

        synthesis = self._route(
            sender="SAAMEvaluationTeam",
            recipient="SAAMArchitect",
            method_name="generate_overall_results",
            evaluations=eval_results.get("scenario_evaluations", []),
            stakeholder_feedback=self.evaluation_feedback,
            business_goals=goals,
            interactions=interactions,
            interaction_weights=interaction_weights.get("interaction_weights", [])
        )
        
        self.template.set_overall_results(
            synthesis.get("architectural_strengths", []),
            synthesis.get("architectural_weaknesses", []),
            synthesis.get("modifiability_assessment", "Pending")
        )
        
        for rec in synthesis.get("recommendations", []):
            self.template.add_recommendation(rec)
        
        self.logger.log(f"  ✓ Identified {len(synthesis.get('architectural_strengths', []))} strengths")
        self.logger.log(f"  ✓ Identified {len(synthesis.get('architectural_weaknesses', []))} weaknesses")
        self.logger.log(f"  ✓ Generated {len(synthesis.get('recommendations', []))} recommendations")
        # Ask Manager to facilitate stakeholder agreement on recommendations
        self.logger.log("  [PHASE 4] Manager facilitating stakeholder agreement on recommendations...")
        rec_review = self._route(
            sender="SAAMEvaluationTeam",
            recipient="SAAMManager",
            method_name="review_recommendations",
            recommendations=synthesis.get('recommendations', []),
            architecture=architecture_details,
            evaluations=eval_results.get('scenario_evaluations', [])
        )

        self.logger.log(f"  ✓ Recommendation approval summary: {rec_review}")
        self.logger.log("  > SAAM Evaluation Complete.")