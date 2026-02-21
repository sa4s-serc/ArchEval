# ArchEval

ArchEval is the replication package for the paper *"Agents as Architecture Evaluators: Automating ATAM and SAAM for Continuous Assessment"* (ICSA 2026). It contains a curated dataset of structured software architecture evaluations and an agentic workflow that autonomously conducts ATAM and SAAM evaluations on any given architectural description.

---

## 1. Dataset

The dataset is the primary contribution of this work. It consists of **39 human-conducted architecture evaluations** extracted from academic literature and converted into a structured JSON format — **18 ATAM** evaluations (in `ATAM/data/`) and **21 SAAM** evaluations (in `SAAM/data/`).

Each JSON file corresponds to one case study and is divided into two logically distinct sections to prevent data leakage during experiments:

- **`evaluation_inputs`** — Information available to an evaluator *before* analysis begins: system context, architectural descriptions, business goals, stakeholder constraints, and other inputs.
- **`ground_truth_outputs`** — The findings of the original human expert team: scenarios, risk identifications, utility trees, recommendations, etc.

### ATAM Dataset (`ATAM/data/`)

18 case studies covering systems such as banking, military wargames, transport, data centers, and more. Each file captures:

- Business context, drivers, and constraints
- Architectural approaches and styles
- Utility tree (quality attributes → scenarios with priority tuples)
- Analysis records: risks, non-risks, sensitivity points, tradeoff points
- Prioritized scenarios

### SAAM Dataset (`SAAM/data/`)

21 case studies spanning domains from embedded systems to healthcare EHRs to federated learning frameworks. Each file captures:

- System context and architecture description
- Evaluation setup and scenario elicitation metadata
- Scenarios with classifications (Direct / Indirect)
- Scenario-based change analysis and inter-scenario interactions
- Overall modifiability assessment, strengths, weaknesses, and recommendations

### Reference PDFs

The original academic papers used for extraction are stored in `ATAM/references/` and `SAAM/references/` for full traceability.

### JSON Templates

`ATAM/template.json` and `SAAM/template.json` define the canonical schema used for both extraction and agent output.

### Data Extraction Process

The dataset was created through a meticulous manual extraction process. The prompt used for extraction is documented in `ATAM/prompt.txt` and `SAAM/prompt.txt`, and the resulting JSON files were validated by both the authors to ensure consistency.

---

## 2. Agentic Workflow

The agentic system simulates a multi-stakeholder architecture evaluation team. Four specialized LLM agents — **Customer**, **Architect**, **Manager**, and **Evaluation Team** — collaborate via a shared JSON blackboard, guided by a deterministic orchestrator. The system supports both ATAM and SAAM workflows.

Models evaluated: **Claude 4.5 Sonnet**, **Gemini 3 Flash**, and **DeepSeek v3.2**.

### Running the Workflows

Both workflows are run from within their respective `agents/` directories. Set the required environment variables first (GCP project, model name, etc.) — see `.env.sample`.

**ATAM** — processes one or more input files from `ATAM/data/`:

```bash
cd ArchEval/ATAM/
python agents/main.py data/          # run all files
python agents/main.py data/banking.json  # run a single file
```

**SAAM** — processes one or more input files from `SAAM/data/`:

```bash
cd ArchEval/SAAM
python agents/main.py --run-all --data-dir data      # run all files
python agents/main.py --customer-data data/beer.json  # run a single file
```

After running, aggregate scores and efficiency metrics across all runs by running aggregate.py and aggregate_efficiency.py in each directory.

### Output Folders

All outputs are organized by model name.

**ATAM outputs:**

| Folder | Contents |
|--------|----------|
| `ATAM/outputs/<model>/` | Generated evaluation JSON for each case study (`<name>_output.json`) |
| `ATAM/logs/<model>/` | Per-run reasoning logs (`.log`) and token/latency efficiency metrics (`_efficiency.json`) |
| `ATAM/scores/<model>/` | Automated evaluation scores comparing agent output to ground truth (`_evaluation.json`) |
| `ATAM/consolidated_scores.json` | Aggregated scores across all runs and case studies |
| `ATAM/consolidated_efficiency.json` | Aggregated token usage and timing statistics |

**SAAM outputs:**

| Folder | Contents |
|--------|----------|
| `SAAM/outputs/<model>/<SystemName>/` | Generated SAAM report JSON and efficiency metrics per system |
| `SAAM/outputs/<model>/outputs/` | Automated evaluation scores per system (`_saam_evaluation_*.json`) |
| `SAAM/consolidated_scores.json` | Aggregated scores across all runs |
| `SAAM/consolidated_efficiency.json` | Aggregated efficiency metrics |

### Evaluation

Automated evaluation runs automatically at the end of each workflow execution and computes ROUGE-L, BLEU, METEOR, BERTScore, and an LLM-as-a-Judge score (using Gemini 3 Flash) comparing agent output against the ground truth. The evaluators are in `ATAM/agents/evaluate.py` and `SAAM/agents/evaluate.py`.

---

## 3. Other Information

### `human_eval/`

Contains the human evaluation rubric and the raw scores collected from the two authors who independently evaluated agent-generated reports. The rubric uses a 5-point Likert scale across four dimensions: Relevance, Coherence, Completeness, and Conciseness.

### `LLM/`

Contains a snapshot of the LMSYS Chatbot Arena leaderboard used to guide model selection for this study.