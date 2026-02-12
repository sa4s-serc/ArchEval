import os
import json
import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from difflib import SequenceMatcher

# Assuming client is in the same directory or python path
from client import LLMClient 

from dotenv import load_dotenv
load_dotenv()

# Ensure NLTK data is available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class SAAMEvaluator:
    """
    SAAM-specific evaluator.
    Handles 'ground_truth_outputs' (Reference) vs 'outputs' (Prediction) mismatch
    and performs content-based scenario matching.
    """
    def __init__(self,
                 reference: Optional[Any] = None,
                 prediction: Optional[Any] = None,
                 model_name: str = "unknown-model"):

        # Load reference
        if isinstance(reference, str):
            with open(reference, 'r') as f:
                raw_ref = json.load(f)
        else:
            raw_ref = reference or {}
        
        # EXTRACTOR FIX: Look for 'ground_truth_outputs' first, then 'outputs'
        self.ref_root = raw_ref.get('SAAM_Evaluation_Extract', raw_ref)
        self.ref_data = self.ref_root.get('ground_truth_outputs', self.ref_root.get('outputs', {}))

        # Load prediction
        if isinstance(prediction, str):
            with open(prediction, 'r') as f:
                raw_pred = json.load(f)
        else:
            raw_pred = prediction or {}
            
        self.pred_root = raw_pred.get('SAAM_Evaluation_Extract', raw_pred)
        self.pred_data = self.pred_root.get('outputs', {})

        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        print(f"Initializing SAAM Evaluator with model: {model_name}...")
        self.llm_client = LLMClient(model_name="gemini-3-flash-preview")
        
        # Pre-calculate scenario mapping (Prediction ID -> Reference ID) based on text similarity
        self.scenario_map = self._map_scenarios()

    def _map_scenarios(self) -> Dict[str, str]:
        """Matches predicted scenarios to reference scenarios based on text similarity."""
        ref_scenarios = self.ref_data.get('scenarios', [])
        pred_scenarios = self.pred_data.get('scenarios', [])
        mapping = {}
        used_refs = set()
        
        for p_scen in pred_scenarios:
            p_text = p_scen.get('scenario_text', '')
            p_id = p_scen.get('scenario_id')
            best_score = 0.0
            best_ref_id = None
            
            for r_scen in ref_scenarios:
                r_id = r_scen.get('scenario_id')
                if r_id in used_refs: continue
                r_text = r_scen.get('scenario_text', '')
                score = SequenceMatcher(None, p_text, r_text).ratio()
                if score > best_score:
                    best_score = score
                    best_ref_id = r_id
            
            if best_score > 0.2 and best_ref_id:
                mapping[p_id] = best_ref_id
                used_refs.add(best_ref_id)
        return mapping

    def _extract_scenarios(self, data: Dict, use_mapping: bool = False) -> List[str]:
        scenarios = data.get('scenarios', [])
        return [f"ID: {s.get('scenario_id')} | Text: {s.get('scenario_text', '')}" for s in scenarios]

    def _extract_classifications(self, data: Dict, is_pred: bool = False) -> List[str]:
        items = data.get('scenario_classification', [])
        texts = []
        for item in items:
            sid = item.get('scenario_id', 'Unknown')
            cls = item.get('classification', 'Unknown')
            rat = item.get('rationale', '')
            map_note = f"(Matches Ref {self.scenario_map[sid]})" if is_pred and sid in self.scenario_map else ""
            texts.append(f"[{sid}] {map_note} Type: {cls} | Rationale: {rat}")
        return texts

    def _extract_evaluations(self, data: Dict, is_pred: bool = False) -> List[str]:
        items = data.get('scenario_evaluations', [])
        texts = []
        for e in items:
            sid = e.get('scenario_id', '')
            arch = e.get('architecture_id', '')
            desc = e.get('change_description', '')
            affected = e.get('affected_components', [])
            if isinstance(affected, list): affected = ', '.join(affected)
            map_note = f"(Matches Ref {self.scenario_map[sid]})" if is_pred and sid in self.scenario_map else ""
            texts.append(f"[{sid}] {map_note} Arch: {arch} | Affected: {affected} | Desc: {desc}")
        return texts

    def _extract_overall_results(self, data: Dict) -> List[str]:
        overall = data.get('overall_results', {})
        texts = []
        if isinstance(overall, dict):
            strengths = overall.get('architectural_strengths', []) or []
            weaknesses = overall.get('architectural_weaknesses', []) or []
            mod = overall.get('modifiability_assessment', '')
            texts.append("STRENGTHS: " + "; ".join(strengths))
            texts.append("WEAKNESSES: " + "; ".join(weaknesses))
            if mod: texts.append(f"MODIFIABILITY: {mod}")
        return texts

    def _extract_recommendations(self, data: Dict) -> List[str]:
        recs = data.get('recommendations', [])
        texts = []
        for r in recs:
            if isinstance(r, str):
                texts.append(f"[Priority: Unknown] {r}")
            elif isinstance(r, dict):
                pr = r.get('priority', 'Unknown')
                rec_text = r.get('recommendation', '')
                texts.append(f"[Priority: {pr}] {rec_text}")
        return texts

    def _extract_arch_comparison(self, data: Dict) -> List[str]:
        comps = data.get('architecture_comparison', [])
        texts = []
        for c in comps:
            aid = c.get('architecture_id', '')
            rank = c.get('rank', '')
            rat = c.get('rationale', '')
            texts.append(f"Arch: {aid} | Rank: {rank} | Rationale: {rat}")
        return texts

    def _extract_interactions(self, data: Dict) -> List[str]:
        inters = data.get('scenario_interactions', [])
        weights = data.get('scenario_interaction_weights', [])
        texts = [f"Interaction: {str(i)}" for i in inters]
        texts.extend([f"Weight: {str(w)}" for w in weights])
        return texts

    def _compute_lexical_metrics(self, ref_texts: List[str], pred_texts: List[str]) -> Dict[str, Any]:
        if not ref_texts or not pred_texts:
            return {"rouge": 0.0, "bert": 0.0, "bleu": 0.0, "meteor": 0.0}

        ref_concat = " ".join(ref_texts)
        pred_concat = " ".join(pred_texts)
        
        rouge_res = self.rouge.score(ref_concat, pred_concat)
        rouge_l = rouge_res['rougeL'].fmeasure

        try:
            ref_tokens = ref_concat.split()
            pred_tokens = pred_concat.split()
            smoothing = SmoothingFunction().method1
            bleu_score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
        except Exception:
            bleu_score = 0.0

        try:
            meteor = meteor_score([ref_tokens], pred_tokens)
        except Exception:
            meteor = 0.0

        try:
            P, R, F1 = bert_score(
                [pred_concat], [ref_concat], lang="en", verbose=False,
                model_type="microsoft/deberta-xlarge-mnli"
            )
            bert = F1.mean().item()
        except Exception as e:
            print(f"Warning: BERTScore failed ({e}). Defaulting to 0.0")
            bert = 0.0

        return {"rouge": rouge_l, "bert": bert, "bleu": bleu_score, "meteor": meteor}

    def _run_judge(self, section_name: str, ref_context: List[str], pred_context: List[str]) -> Dict[str, Any]:
        if not ref_context:
            return {
                "relevance": 0, "coherence": 0, "completeness": 0, "conciseness": 0,
                "reasoning": "Reference data for this section was empty. Cannot evaluate."
            }

        system_instruction = f"""
### **Judge Prompt: Conceptual Coverage–Focused SAAM Evaluation**

You are an expert **Software Architecture Evaluation Judge** specializing in SAAM-style assessments.  
Your task is to evaluate the AI-generated SAAM analysis (**Prediction**) against the Human Ground Truth (**Reference**) for the section: **`{section_name}`**.

### **Important Evaluation Principle**

The **specific details, examples, or wording in the Reference are NOT important**.  
Instead, assess whether the Prediction **addresses the same underlying architectural concerns, evaluation dimensions, and reasoning categories** that the Reference covers.

Focus on **conceptual coverage and evaluative intent**, not exact matches.

### **SCORING RUBRIC (Likert Scale 1–5)**

**1 = Very Poor | 2 = Poor | 3 = Acceptable | 4 = Good | 5 = Excellent**

### **Evaluation Criteria**

#### **1. Relevance**
Does the Prediction meaningfully engage with the *architectural concerns and business drivers implied by the Reference*, even if the details differ?

#### **2. Coherence**
Is the Prediction logically structured, internally consistent, and clear in how it evaluates architectural tradeoffs or quality attributes?

#### **3. Completeness**
Does the Prediction cover the **key evaluation bases** present in the Reference (e.g., quality attributes, risks, tradeoffs, sensitivity points, or stakeholder concerns), regardless of implementation details or examples used?

#### **4. Conciseness**
Is the evaluation efficient and focused, avoiding unnecessary repetition or irrelevant elaboration while still covering required evaluation dimensions?

### **Scoring Guidance**

- Do **not penalize** differences in terminology, ordering, or illustrative examples.
- Do **penalize** missing entire evaluation dimensions or architectural concerns clearly addressed in the Reference.
- Reward evaluations that demonstrate **breadth of architectural reasoning**, even if depth or specificity differs.
"""

        user_content = f"""
        REFERENCE DATA (Ground Truth):
        {json.dumps(ref_context, indent=2)}

        PREDICTION DATA (AI Generated):
        {json.dumps(pred_context, indent=2)}

        Return a JSON object with this exact schema:
        {{
            "relevance": <int>,
            "coherence": <int>,
            "completeness": <int>,
            "conciseness": <int>,
            "reasoning": "<string>"
        }}
        """

        response_text, usage = self.llm_client.generate(system_instruction, user_content)
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            print(f"⚠️ JSON Decode Error for SAAM judge on {section_name}")
            return {"relevance": 0, "coherence": 0, "completeness": 0, "conciseness": 0, "reasoning": "JSON Decode Error"}

    def evaluate(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Runs the evaluation and returns the dictionary of results."""
        results = {"metadata": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                 "model_used": getattr(self.llm_client, 'model_name', 'unknown')}}

        sections = [
            ("scenarios", lambda d: self._extract_scenarios(d)),
            ("scenario_classification", lambda d: self._extract_classifications(d, is_pred=(d==self.pred_data))),
            ("scenario_evaluations", lambda d: self._extract_evaluations(d, is_pred=(d==self.pred_data))),
            ("overall_results", self._extract_overall_results),
            ("recommendations", self._extract_recommendations),
            ("architecture_comparison", self._extract_arch_comparison),
            ("scenario_interactions", self._extract_interactions)
        ]

        for name, extractor in sections:
            print(f"Evaluating {name}...")
            ref_content = extractor(self.ref_data)
            pred_content = extractor(self.pred_data)
            
            lexical = self._compute_lexical_metrics(ref_content, pred_content)
            judge = self._run_judge(name, ref_content, pred_content)
            
            results[name] = {"lexical": lexical, "judge": judge}

        if output_file:
            if not os.path.exists(os.path.dirname(output_file)) and os.path.dirname(output_file):
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Evaluation saved to {output_file}")
            
        return results


    def evaluate_and_save(self, output_dir: str, system_name: str, timestamp: Optional[str] = None) -> str:
        """
        Orchestrates evaluation and saves it to a structured directory.
        Used by the main SAAM orchestrator.
        """
        ts = timestamp or time.strftime("%Y%m%d_%H%M%S")
        
        # Sanitize model name for folder path (remove characters like ':')
        model_name = getattr(self.llm_client, 'model_name', 'unknown-model')
        model_folder = model_name.replace(":", "_").replace("/", "_")
        
        base_dir = os.path.join(output_dir, model_folder)
        os.makedirs(base_dir, exist_ok=True)
        
        filename = f"{system_name}_saam_evaluation_{ts}.json"
        out_path = os.path.join(base_dir, filename)
        
        print(f"Running evaluation for {system_name}...")
        # Run evaluate without saving yet, so we can save to the specific path calculated here
        results = self.evaluate(output_file=None)
        
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"✅ Results successfully saved to {out_path}")
        return out_path