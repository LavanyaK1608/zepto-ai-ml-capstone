"""
Module 2 - Analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")
OUT = Path(__file__).parent
df = pd.read_csv(OUT / "titanic.csv")

print("="*60)
print("1. DATASET PROFILE")
print("="*60)
print(df.info())
print("\nShape:", df.shape)
print("\nDescribe:\n", df.describe(include="all").T)

print("\nMissing value %:")
miss = (df.isnull().mean() * 100).round(2)
print(miss[miss > 0].sort_values(ascending=False))


# 2. Missing-value handling

print("\n" + "="*60)
print("2. MISSING-VALUE HANDLING")
print("="*60)

# age: 19.9% → 5-30% → impute median
print("age missing: 19.87% → impute with median")
df["age"] = df["age"].fillna(df["age"].median())

# embarked / embark_town: 0.22% → <5% → drop rows
print("embarked missing: 0.22% → drop rows")
df = df.dropna(subset=["embarked"]).copy()

# deck: 77.2% → too high → drop column (imputation unreliable)
print("deck missing: 77.22% → drop column (too high for reliable imputation)")
df = df.drop(columns=["deck"])

print("After cleaning shape:", df.shape)

# Save cleaned version for modeling
df.to_csv(OUT / "titanic_cleaned.csv", index=False)
print("Saved titanic_cleaned.csv")


# 3. Univariate analysis

print("\n" + "="*60)
print("3. UNIVARIATE ANALYSIS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df["age"], kde=True, ax=axes[0,0])
axes[0,0].set_title("Age Histogram")
sns.boxplot(x=df["age"], ax=axes[0,1])
axes[0,1].set_title("Age Boxplot")
sns.histplot(df["fare"], kde=True, ax=axes[1,0])
axes[1,0].set_title("Fare Histogram")
sns.boxplot(x=df["fare"], ax=axes[1,1])
axes[1,1].set_title("Fare Boxplot")
plt.tight_layout()
plt.savefig(OUT / "univariate_age_fare.png", dpi=120)
plt.close()

def iqr_outliers(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    return ((s < lower) | (s > upper)).sum()

print(f"Age outliers (IQR): {iqr_outliers(df['age'])}")
print(f"Fare outliers (IQR): {iqr_outliers(df['fare'])}")

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]
print(f"Fare mean={fare_mean:.2f}, median={fare_median:.2f}, mode={fare_mode:.2f}")
print("Fare is right-skewed (mean > median > mode).")


# 4. Bivariate analysis

print("\n" + "="*60)
print("4. BIVARIATE ANALYSIS")
print("="*60)

print("Survival rate by sex:")
print(df.groupby("sex")["survived"].mean().round(3))

print("\nSurvival rate by pclass:")
print(df.groupby("pclass")["survived"].mean().round(3))

print("\nSurvival rate by sex + pclass:")
print(df.groupby(["sex", "pclass"])["survived"].mean().round(3).unstack())

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = df[corr_cols].corr()
print("\nCorrelation matrix (6 columns):")
print(corr.round(3))

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlation Heatmap (survived, pclass, age, sibsp, parch, fare)")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap.png", dpi=120)
plt.close()

# Two strongest off-diagonal correlations
corr_abs = corr.abs().unstack().sort_values(ascending=False)
corr_abs = corr_abs[corr_abs < 1.0]  # remove self
print("\nTwo strongest correlations:")
print(corr_abs.head(4))  # pairs appear twice


# 5. Multivariate data story (4+ charts)

print("\n" + "="*60)
print("5. MULTIVARIATE DATA STORY")
print("="*60)

# Chart 1
plt.figure(figsize=(7,4))
sns.barplot(data=df, x="sex", y="survived", hue="pclass")
plt.title("Survival Rate by Sex and Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(OUT / "story_sex_pclass.png", dpi=120)
plt.close()
print("Chart 1: Women in 1st class had the highest survival; men in 3rd class the lowest.")

# Chart 2
plt.figure(figsize=(7,4))
sns.boxplot(data=df, x="survived", y="age", hue="sex")
plt.title("Age Distribution by Survival and Sex")
plt.tight_layout()
plt.savefig(OUT / "story_age_survival.png", dpi=120)
plt.close()
print("Chart 2: Surviving women tend to be slightly younger; age effect is weaker for men.")

# Chart 3
plt.figure(figsize=(7,4))
sns.scatterplot(data=df, x="age", y="fare", hue="survived", alpha=0.6)
plt.title("Age vs Fare coloured by Survival")
plt.tight_layout()
plt.savefig(OUT / "story_age_fare_scatter.png", dpi=120)
plt.close()
print("Chart 3: Higher fares (mostly 1st class) show higher survival density.")

# Chart 4
plt.figure(figsize=(7,4))
sns.countplot(data=df, x="sibsp", hue="survived")
plt.title("Survival Count by Number of Siblings/Spouses")
plt.tight_layout()
plt.savefig(OUT / "story_sibsp.png", dpi=120)
plt.close()
print("Chart 4: Passengers with 1-2 family members had better odds than large families or alone.")

# 6. Exploratory standardization (EDA only)

print("\n" + "="*60)
print("6. EXPLORATORY Z-SCORE STANDARDIZATION")
print("="*60)

df["age_z"] = (df["age"] - df["age"].mean()) / df["age"].std()
df["fare_z"] = (df["fare"] - df["fare"].mean()) / df["fare"].std()
print("After z-score:")
print(df[["age_z", "fare_z"]].describe().round(3))

print("\n✅ EDA complete.")
