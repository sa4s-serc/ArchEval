import os
import json
import sys
import time
import numpy as np
from typing import Optional, Dict, Any, List
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from client import LLMClient
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from dotenv import load_dotenv
load_dotenv()

# Ensure NLTK data is available for METEOR
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class ATAMEvaluator:
    def __init__(self, 
                 reference_path: str, 
                 prediction_path: str, 
                 model_name: str = "gemini-3-flash-preview"):
        
        self.ref_data = self._load_json(reference_path)
        self.pred_data = self._load_json(prediction_path)
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        print(f"Initializing LLM Client with model: {model_name}...")
        self.llm_client = LLMClient(model_name=model_name)

    def _load_json(self, path: str) -> Dict:
        with open(path, 'r') as f:
            data = json.load(f)
        return data.get("ATAM_Evaluation", data) if "ATAM_Evaluation" in data else data

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Writes evaluation results to the specified output file."""
        try:
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"✅ Results successfully saved to {output_path}")
        except IOError as e:
            print(f"❌ Error saving results: {e}")

    def _extract_approaches(self, data: Dict) -> List[str]:
        items = data.get("architectural_approaches", [])
        return [f"{x.get('approach_name', 'Unknown')}: {x.get('description', '')}" for x in items]

    def _extract_utility_tree(self, data: Dict) -> List[str]:
        texts = []
        root = data.get("utility_tree", {})
        if not isinstance(root, dict): 
            return []
        nodes = root.get("quality_attribute_nodes", [])
        for node in nodes:
            qa = node.get("attribute_name", "General")
            for child in node.get("children", []):
                desc = child.get("scenario_description", "")
                texts.append(f"[{qa}] {desc}")
        return texts

    def _extract_scenarios(self, data: Dict) -> List[str]:
        items = data.get("scenarios", [])
        return [x.get("scenario_text", "") for x in items]

    def _extract_analysis_records(self, data: Dict) -> List[str]:
        records = data.get("analysis_records", [])
        texts = []
        for rec in records:
            scenario = rec.get("scenario_reference", {}).get("text", "Unknown Scenario")
            findings = rec.get("findings", {})
            risks = [r.get("description", "") for r in findings.get("risks", [])]
            non_risks = [nr.get("description", "") for nr in findings.get("non_risks", [])]
            blob = f"Scenario: {scenario} | Risks: {'; '.join(risks)} | Non-Risks: {'; '.join(non_risks)}"
            texts.append(blob)
        return texts

    def _compute_lexical_metrics(self, ref_texts: List[str], pred_texts: List[str]) -> Dict[str, Any]:
        if not ref_texts or not pred_texts:
            return {"rouge": 0.0, "bert": 0.0, "bleu": 0.0, "meteor": 0.0}

        ref_concat = " ".join(ref_texts)
        pred_concat = " ".join(pred_texts)
        
        # ROUGE
        rouge_res = self.rouge.score(ref_concat, pred_concat)
        
        # BLEU
        ref_tokens = ref_concat.split()
        pred_tokens = pred_concat.split()
        smoothing = SmoothingFunction().method1
        bleu_score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)

        # METEOR
        meteor = meteor_score([ref_tokens], pred_tokens)

        # BERTScore
        try:
            P, R, F1 = bert_score(
                [pred_concat], 
                [ref_concat], 
                lang="en",
                verbose=False,
                model_type="microsoft/deberta-xlarge-mnli"
            )
            bert = {
                "precision": P.mean().item(),
                "recall": R.mean().item(),
                "f1": F1.mean().item()
            }
        except Exception as e:
            print(f"Warning: BERTScore failed ({e}). Defaulting to 0.0")
            bert = 0.0

        return {
            "rouge": rouge_res,
            "bert": bert,
            "bleu": bleu_score,
            "meteor": meteor
        }

    def _run_judge(self, section_name: str, ref_context: Any, pred_context: Any) -> Dict[str, Any]:
        system_instruction = f"""
### **Judge Prompt: Conceptual Coverage–Focused ATAM Evaluation**

You are an expert **Software Architecture Evaluation Judge** specializing in ATAM-style assessments.  
Your task is to evaluate the AI-generated ATAM analysis (**Prediction**) against the Human Ground Truth (**Reference**) for the section: **`{section_name}`**.

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
            print(f"⚠️ JSON Decode Error for {section_name}")
            return {
                "relevance": 0, "coherence": 0, "completeness": 0, "conciseness": 0,
                "reasoning": "JSON Decode Error"
            }

    def evaluate(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        results = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_used": self.llm_client.model_name
            }
        }

        sections = [
            ("architectural_approaches", self._extract_approaches),
            ("utility_tree", self._extract_utility_tree),
            ("scenarios", self._extract_scenarios),
            ("analysis_records", self._extract_analysis_records)
        ]

        for name, extractor_func in sections:
            print(f"🔍 Evaluating Section: {name}...")
            
            ref_content = extractor_func(self.ref_data.get("ground_truth_outputs"))
            pred_content = extractor_func(self.pred_data.get("generated"))
            
            lexical_scores = self._compute_lexical_metrics(ref_content, pred_content)
            judge_scores = self._run_judge(name, ref_content, pred_content)
            
            results[name] = {
                "lexical": lexical_scores,
                "judge": judge_scores
            }

        if output_file:
            self.save_results(results, output_file)

        return results
    
def evaluate_file(input_file: str, model_name: str):
    file_name = os.path.basename(input_file)

    print(f"\n================ Evaluating File: {file_name} ================\n")

    if (not file_name.endswith(".json")) or (file_name == "template.json") or (not os.path.exists(input_file)):
        print(f"⚠️ Skipping invalid file: {file_name}")
        return

    base_name = file_name.split(".")[0]

    ref_file = f"data/{base_name}.json"
    pred_file = f"outputs/{model_name}/{base_name}_output.json"
    output_file = f"scores/{model_name}/{base_name}_evaluation.json"

    if not os.path.exists(ref_file):
        print(f"⚠️ Reference file not found: {ref_file}")
        return

    if not os.path.exists(pred_file):
        print(f"⚠️ Prediction file not found: {pred_file}")
        return

    evaluator = ATAMEvaluator(
        reference_path=ref_file,
        prediction_path=pred_file,
    )

    print("🚀 Starting Evaluation...")
    evaluator.evaluate(output_file=output_file)


if __name__ == "__main__":
    MODEL_NAME = "gemini-3-flash-preview"

    for input_path in sys.argv[1:]:
        # If a single file is passed
        if not os.path.isdir(input_path):
            evaluate_file(input_path, MODEL_NAME)
            continue

        # If a folder is passed
        input_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".json")
        ]

        for input_file in input_files:
            evaluate_file(input_file, MODEL_NAME)
