import pandas as pd

df = pd.read_csv("winequality-white.csv", sep=";")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))
print("\nQuality distribution:")
print(df["quality"].value_counts().sort_index())
print("\nMissing values:")
print(df.isnull().sum())

# Binarize quality: 1 = "good" (>=7), 0 = "not good" (<7)
df["quality_label"] = (df["quality"] >= 7).astype(int)

# Drop the original numeric quality so the model doesn't cheat by using it
df = df.drop(columns=["quality"])

print("\nNew target distribution:")
print(df["quality_label"].value_counts())
print("\nFinal shape:", df.shape)
print("\nColumns now:", df.columns.tolist())

from pycaret.classification import setup, compare_models, plot_model, save_model

# 1. Setup the experiment
exp = setup(
    data=df,
    target="quality_label",
    session_id=42,
    verbose=True,
)

# 2. Compare models and grab the top 3
top3 = compare_models(n_select=3)
print("\nTop 3 models:")
for i, model in enumerate(top3, 1):
    print(f"  {i}. {type(model).__name__}")

    # 3. Confusion matrix for the best model
best_model = top3[0]
plot_model(best_model, plot="confusion_matrix", save=True)

# 4. Save the best model pipeline for FastAPI to load later
save_model(best_model, "best_pipeline")
print("\nSaved best_pipeline.pkl")

# ============================================================
# Scikit-Learn Manual Workflow
# Rebuilding PyCaret's winner (ExtraTreesClassifier) by hand
# ============================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report

# 1. Separate features (X) and target (y)
X = df.drop(columns=["quality_label"])
y = df["quality_label"]

# 2. Train/test split — match PyCaret's 70/30 with the same seed
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 3. Manual scaling — fit on train only, then transform both
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train ExtraTrees with the same seed for fair comparison
sk_model = ExtraTreesClassifier(random_state=42)
sk_model.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred = sk_model.predict(X_test_scaled)
print("\n" + "=" * 60)
print("SCIKIT-LEARN MANUAL WORKFLOW — Classification Report")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=["not good", "good"]))

# ============================================================
# SYNTHESIS — PyCaret vs. Scikit-Learn Workflow Comparison
# ============================================================
# The PyCaret workflow was dramatically more efficient. With three
# function calls — setup(), compare_models(), and plot_model() — it
# automated train/test splitting, scaling, 10-fold cross-validation,
# and benchmarking across 14 classifiers in roughly two minutes.
# Reproducing just the winning model (ExtraTrees) in scikit-learn
# required manual feature/target separation, an explicit
# train_test_split with stratification, fitting a StandardScaler on
# training data only, fitting the classifier, and finally generating
# a classification_report. PyCaret eliminates dozens of decisions
# and lines of code while still surfacing the underlying scikit-learn
# estimator for inspection.
#
# The metrics differ slightly: PyCaret reported 0.866 accuracy and
# 0.62 F1, while the manual workflow reported 0.89 accuracy and 0.70
# F1. The difference is methodological, not algorithmic. PyCaret
# averages performance across 10 stratified cross-validation folds,
# producing a conservative and more reliable estimate of
# generalization. The manual run measured a single 30% holdout,
# which can swing favorably or unfavorably depending on the split.
# I trust the cross-validated numbers more. PyCaret is the better
# choice for rapid model selection; manual scikit-learn is the
# better choice when you need fine control over a specific pipeline
# in production.
# ============================================================  