# Redrob AI Ranker

Production-oriented semantic candidate ranking for the Redrob Data & AI Challenge. The pipeline streams and validates 100,000 profiles, represents complete career evidence with `sentence-transformers/all-mpnet-base-v2`, combines semantic and structured signals, rejects internally inconsistent profiles, and writes an automatically validated top-100 submission.

## Architecture

```text
Job description DOCX ──► requirement + skill extraction ──► embeddings
                                                                  │
Candidate JSONL ──► validation ──► typed records ──► preprocessing ├─► hybrid scoring
                                                                  │        │
Redrob signals ───────────────────────────────────────────────────┘        ▼
                                                        stable ranking + grounded reasons
                                                                       │
                                                                       ▼
                                                        outputs/submission.csv
```

The candidate representation includes headline, summary, current title, all career titles and descriptions, skills, education, and certifications. JSONL records are consumed as a stream and converted immediately into compact typed records. Processed profiles and embeddings are cached with source/model fingerprints, so repeat ranking runs do not recompute unchanged work.

## Installation

Python 3.11 is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first run downloads `sentence-transformers/all-mpnet-base-v2` into `models/huggingface`. After that, `--local-files-only` guarantees an offline run.

## Usage

Run the complete pipeline with the challenge bundle paths:

```powershell
python main.py
```

The command produces:

- `outputs/submission.csv`: exact top-100 challenge submission.
- `outputs/ranked_candidates.parquet`: all candidates, component scores, rank, integrity factor, matched skills, and explainability JSON.
- `outputs/cache/processed_candidates.parquet`: cached semantic profiles and structured features.
- `outputs/cache/embeddings/`: candidate, JD, requirement, and skill embedding caches with integrity metadata.

Useful reproducibility options:

```powershell
python main.py --device cpu --local-files-only
python main.py --force --batch-size 128
python main.py --candidates data/candidates.jsonl --job-description data/job_description.docx --output outputs/submission.csv
```

`--device auto` is the default: CUDA or Apple MPS is used for embedding precomputation when available, with CPU fallback. Ranking and all structured scoring are NumPy/Pandas CPU operations. Cached reruns satisfy the offline ranking workflow without loading the transformer.

Validate the generated CSV independently:

```powershell
python data/validate_submission.py outputs/submission.csv
```

## Pipeline

1. `src/load_data.py` streams JSONL, validates against the supplied JSON Schema, creates immutable dataclasses, and reads DOCX content.
2. `src/preprocess.py` normalizes text and skill aliases, builds complete semantic profiles, extracts experience and Redrob signals, and derives JD requirements and skills.
3. `src/embed.py` batch-encodes candidates, the focused JD, individual requirements, and the skill vocabulary. Vectors are L2-normalized and atomically cached with SHA-256 text fingerprints.
4. `src/scoring.py` calculates semantic, skill, experience, and behavior scores. Exact and semantic skill evidence are combined. Record consistency checks down-weight impossible career dates, contradictory duration claims, and expert-skill anomalies.
5. `src/ranking.py` applies a stable descending sort and deterministic `candidate_id` tie-break, then exports the exact CSV contract.
6. `src/reasoning.py` creates two-sentence maximum explanations from actual title, company, experience, skills, career text, and behavioral values.
7. `main.py` persists full explainability, validates the CSV format and candidate membership, and prints total candidates, runtime, top candidate, and average top-100 score.

## Scoring

Every component is bounded to `[0, 1]` and the final score follows the required formula exactly:

```text
Final Score = 0.45 × Semantic
            + 0.20 × Skill
            + 0.15 × Experience
            + 0.20 × Behavior
```

### Semantic score

Cosine similarity combines the focused JD vector with the mean of the candidate's three strongest requirement-level similarities. Robust percentile normalization prevents narrow cosine ranges from collapsing score separation.

### Skill score

JD skills are automatically discovered from both a technical vocabulary and the candidate-pool vocabulary. Candidate skills receive exact-match credit and semantic relatedness credit from MPNet embeddings. This supports plain-language equivalents such as content matching, vector representations, and information-retrieval systems without relying on substring counts.

### Experience score

Experience evaluates the JD's flexible 5–9 year band, senior applied-AI role evidence, production delivery language, retrieval/ranking depth, and product-company exposure. The integrity factor checks career-duration consistency, current-role consistency, company founding dates, and impossible expert-skill duration patterns. It multiplies every component before the final weighted sum, preventing keyword-rich honeypots from surfacing.

### Behavior score

Behavior combines GitHub activity, recruiter response rate, profile completeness, recruiter saves, search appearances, interview completion, offer acceptance, open-to-work status, notice period, and relocation willingness. Missing-history sentinels are treated neutrally. Expected salary is retained as profile data but is not used for ranking.

Each row in `ranked_candidates.parquet` includes an `explainability` dictionary containing all component scores, the final score, the data-quality factor, and matched skills.

## Folder structure

```text
redrob-ai-ranker/
├── data/                    # supplied candidates, schema, JD, and validator
├── models/huggingface/      # local transformer model cache
├── outputs/                 # submission, full ranking, and reusable caches
├── src/
│   ├── embed.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── ranking.py
│   ├── reasoning.py
│   ├── scoring.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## Operational characteristics

- JSONL input is streamed; raw dictionaries are not retained.
- Embeddings are generated in batches, never with an all-pairs candidate matrix.
- Requirement similarity is `O(candidates × requirements)` with a small fixed requirement count.
- Skill similarity is calculated once for the unique skill vocabulary and reused by every candidate.
- Parquet uses Zstandard compression and output writes are atomic.
- Cache metadata prevents silent reuse with a changed dataset, model, or text input.
- No hosted APIs or network calls are used during ranking.

## Future improvements

With labeled relevance judgments, the fixed hybrid weights can be replaced by a calibrated learning-to-rank model and evaluated with grouped cross-validation. Recruiter feedback could also support query-specific behavioral calibration, temporal engagement decay, and monitored embedding refresh policies. Those extensions preserve the current deterministic, offline feature pipeline while learning better interactions between its explainable components.
