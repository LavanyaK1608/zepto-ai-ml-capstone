# Module 2 – Analytics Pipeline (50 marks)

## Purpose
End-to-end Titanic analysis: profile → clean → visualise → model → evaluate → save production pipeline.

## Setup
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
```

## How to run
```bash
cd analytics
python 01_eda.py          # profiling, cleaning, EDA story, saves titanic_cleaned.csv
python 02_modeling.py     # stratified split → pipelines → 3 classifiers → tuning → regression → joblib
```

## Key design decisions
- **Missing values** (threshold rule applied):
  - age 19.9 % → median impute
  - embarked 0.22 % → drop rows
  - deck 77.2 % → drop column
- **Stratified split** because survived is imbalanced (~38 % positive).
- **All preprocessing** (impute / encode / scale) is fit **only on the training fold**.
- **SMOTE** applied only inside the training fold via `imblearn.pipeline`.
- **RandomForestClassifier(oob_score=True)** so OOB score is available after GridSearch.
- Final artefact is a **full Pipeline** (preprocessor + estimator) saved with joblib – usable on raw data.

## Written interpretations (required)
- Fare is **right-skewed** (mean > median > mode).
- Two strongest correlations: `pclass–fare` (negative) and `sibsp–parch` (positive).
- Women in 1st class had the highest survival rate; men in 3rd class the lowest.
- Residual plot for fare regression shows mild heteroscedasticity.
- Recommendation: deploy the tuned Random Forest for best precision/recall/AUC balance.
