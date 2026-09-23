# Zepto AI/ML Capstone Project

**Certificate Program in Artificial Intelligence and Machine Learning**  
Single public repository containing all three modules.

| Module | Folder | Marks |
|--------|--------|-------|
| Data Pipeline | `/data_pipeline` | 25 |
| Analytics Pipeline | `/analytics` | 50 |
| Support Assistant | `/support_assistant` | 25 |
| **Total** | | **100** |

## Quick Start

```bash
git clone <your-repo-url>
cd zepto-ai-ml-capstone
pip install -r requirements.txt
```

### Module 1 – Data Pipeline
```bash
cd data_pipeline
python scrape_and_load.py
```
- Scrapes ≥ 60 books from books.toscrape.com across ≥ 3 categories  
- Cleans → converts GBP→INR with **fixed rate 1 GBP = 105.50 INR**  
- Loads into normalized SQLite (`categories` ↔ `books`)  
- Runs 5 SQL queries + pandas merge equivalence

### Module 2 – Analytics
```bash
cd analytics
python 01_eda.py
python 02_modeling.py
```
- Loads Titanic once, saves `titanic.csv` offline fallback  
- Missing-value handling with explicit threshold rule  
- Full EDA story + 4 multivariate charts with written interpretations  
- Stratified split → ColumnTransformer pipeline → 3 classifiers  
- Imbalance comparison (baseline / class_weight / SMOTE)  
- GridSearchCV + OOB score  
- Regression side-task (fare) with residual analysis  
- Saves complete `best_pipeline.joblib`

### Module 3 – Support Assistant
```bash
cd support_assistant
uvicorn main:app --host 0.0.0.0 --port 7860
```
- 8 Zepto policy documents embedded with `all-MiniLM-L6-v2` → ChromaDB  
- LangGraph (classify_intent → retrieve_and_answer / direct_answer)  
- **MOCK_LLM=1 (default)** is the fully offline graded baseline  
- FastAPI `POST /ask` returning Pydantic-validated JSON  
- Dockerfile for local container run

## Design Decisions (summary)

**Data Pipeline**  
- Fixed currency rate only (no live API) as required.  
- Unparseable ratings dropped (extremely rare).  
- Classic star-schema style two-table design.

**Analytics**  
- Missing % measured first, then threshold rule applied.  
- All preprocessing fit **only on training fold**.  
- SMOTE applied only inside training fold via imblearn Pipeline.  
- Final artefact is a full Pipeline (usable on raw data).

**Support Assistant**  
- Mock mode is deterministic and network-free.  
- Retrieval always real; only generation branches on MOCK_LLM.  
- Structured prompt contains Role-Context-Task-Format-Length + negative constraint + few-shot.

## Git Workflow
A feature branch was created, committed to multiple times, and merged back into `main` (see commit history).

## License
Academic / educational use.
