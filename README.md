# 🚀 Redrob AI Candidate Ranker
### AI-Powered Semantic Candidate Ranking using NLP, Semantic Search, Hybrid Scoring & Explainable AI

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Sentence Transformers](https://img.shields.io/badge/SentenceTransformers-NLP-success?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Prototype-brightgreen?style=for-the-badge)

</p>

---

## 📌 Overview

Recruiters receive thousands of applications for every role, making manual screening slow and inefficient. Traditional Applicant Tracking Systems (ATS) primarily rely on keyword matching, which often overlooks highly qualified candidates whose profiles use different terminology.

**Redrob AI Candidate Ranker** is a production-inspired AI recruitment system designed to move beyond keyword matching by combining **semantic understanding**, **structured profile analysis**, and **behavioral signals** to recommend the most relevant candidates for a job description.

The project is designed with scalability, explainability, and modularity in mind, making it suitable for large-scale recruitment workflows.

---

# 🎯 Problem Statement

Traditional ATS systems struggle because they:

- Depend heavily on exact keyword matching
- Fail to understand semantic similarity
- Ignore candidate behavioral signals
- Cannot explain ranking decisions
- Miss strong candidates using different wording

Our goal is to build an intelligent ranking engine that evaluates candidates more like an experienced recruiter rather than a search engine.

---

# 💡 Solution

The proposed system performs intelligent candidate ranking through a hybrid AI pipeline:

- Semantic understanding of Job Descriptions
- Candidate profile understanding
- Skill matching
- Experience analysis
- Behavioral signal analysis
- Explainable candidate recommendations

Instead of searching for keywords, the system understands **meaning**.

---

# ✨ Key Features

✅ Semantic Candidate Ranking

✅ Job Description Understanding

✅ Candidate Profile Parsing

✅ Hybrid Ranking Engine

✅ Skill Matching

✅ Experience Matching

✅ Behavioral Signal Analysis

✅ Explainable Candidate Recommendations

✅ Modular AI Pipeline

✅ Scalable Architecture for 100K+ Candidate Profiles

---

# 🏗 System Architecture

```text
                    Job Description
                           │
                           ▼
                Requirement Understanding
                           │
                           ▼
                Candidate Data Processing
                           │
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
 Candidate Semantic Profile           Behavioral Signals
        │                                     │
        └──────────────────┬──────────────────┘
                           ▼
               Semantic Embedding Engine
                           │
                           ▼
                 Hybrid Scoring Framework
        ┌──────────┬──────────┬──────────┬──────────┐
        ▼          ▼          ▼          ▼
   Semantic     Skills    Experience   Behavior
        └──────────┴──────────┴──────────┘
                           │
                           ▼
                  Candidate Ranking Engine
                           │
                           ▼
              Explainable Recommendation
                           │
                           ▼
                   Top Ranked Candidates
```

---

# ⚙️ Project Workflow

```
Job Description
      ↓
Text Preprocessing
      ↓
Semantic Representation
      ↓
Candidate Profile Construction
      ↓
Feature Extraction
      ↓
Hybrid Candidate Scoring
      ↓
Candidate Ranking
      ↓
Explainable Recommendations
```

---

# 🧠 Proposed Hybrid Scoring Strategy

The overall ranking score combines multiple perspectives instead of relying on a single similarity metric.

| Component | Description |
|------------|-------------|
| Semantic Match | Measures semantic similarity between candidate profile and job description |
| Skill Alignment | Matches required technical skills |
| Experience Fit | Evaluates relevance of professional experience |
| Behavioral Signals | Uses recruiter interaction and platform activity |

Example weighting:

| Score Component | Weight |
|----------------|---------|
| Semantic Similarity | 45% |
| Skill Match | 20% |
| Experience | 15% |
| Behavioral Signals | 20% |

---

# 🔍 Explainable AI

Instead of only producing a score, the system is designed to explain *why* a candidate is recommended.

Example:

```
Candidate Score : 92.4

Reason

✔ Strong semantic alignment with AI Engineer role

✔ Relevant Python and ML experience

✔ High recruiter engagement

✔ Strong profile completeness
```

---

# 🧩 Technology Stack

### Programming

- Python

### Machine Learning

- PyTorch
- Scikit-Learn
- Sentence Transformers

### Data Processing

- Pandas
- NumPy

### NLP

- Semantic Embeddings
- Cosine Similarity
- Text Preprocessing

### Utilities

- python-docx
- tqdm
- pathlib
- logging

---

# 📂 Project Structure

```
redrob-ai-ranker/

│

├── data/

├── docs/

├── models/

├── outputs/

├── src/

│   ├── load_data.py

│   ├── preprocess.py

│   ├── embed.py

│   ├── scoring.py

│   ├── ranking.py

│   ├── reasoning.py

│   └── utils.py

│

├── main.py

├── requirements.txt

├── README.md

└── LICENSE
```

---

# 🚀 Getting Started

```bash
git clone https://github.com/Vishwasgithu/redrob-ai-ranker.git

cd redrob-ai-ranker

pip install -r requirements.txt

python main.py
```

---

# 📈 Current Project Status

### Completed

- Dataset Analysis
- Candidate Schema Understanding
- Project Architecture
- Modular Project Structure
- Data Processing Pipeline Design

### In Progress

- Semantic Embedding Engine
- Hybrid Candidate Ranking
- Explainable AI
- Performance Optimization

---

# 🔮 Future Roadmap

This project is designed to evolve into a production-grade AI recruitment platform.

Future enhancements include:

- Retrieval-Augmented Generation (RAG) for grounded recruiter explanations
- FAISS-based vector search
- Cross-Encoder re-ranking
- Learning-to-Rank models
- Multi-Agent Recruiter Assistant
- FastAPI deployment
- Docker containerization
- Kubernetes deployment
- Interactive recruiter dashboard
- Resume ingestion API

---

# 📊 Engineering Highlights

- Modular software architecture
- Memory-efficient processing pipeline
- Designed for 100K+ candidate profiles
- Production-oriented code organization
- Explainable AI recommendations
- Hybrid ranking methodology
- RAG-ready architecture
- Easily deployable as a REST API

---

# 📜 License

This project is intended for educational, research, and portfolio purposes.

---

# 👨‍💻 Author

**Vishwas Choudhary**

AI & ML | Generative AI | LLMs | Computer Vision | Applied Machine Learning

GitHub:
https://github.com/Vishwasgithu

---

⭐ If you found this project interesting, consider giving it a star.