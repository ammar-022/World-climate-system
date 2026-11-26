"""Script version of the notebook (identical code). Run from the project root: python train.py"""
import matplotlib
matplotlib.use("Agg")

import json, re, time, warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
CATEGORIES = ["Climate", "Pollution", "Environment", "Weather", "Ozone Holes", "Agriculture & Food"]

# project root = the folder that contains data/ (works from the root or from notebooks/)
ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
DATA_PATH = ROOT / "data" / "SciDCC.csv"
for sub in ("models", "reports", "images"):
    (ROOT / sub).mkdir(exist_ok=True)
print("scikit-learn", sklearn.__version__, "| root:", ROOT)

raw = pd.read_csv(DATA_PATH)
print("Original shape:", raw.shape)
print("Missing values:\n", raw.isna().sum()[lambda s: s > 0])
original_counts = raw["Category"].value_counts()

df = raw[raw["Category"].isin(CATEGORIES)].copy()
print("\nAfter keeping the six categories:", df.shape)
df["Category"].value_counts().reindex(CATEGORIES)

def clean_text(text):
    """Must stay identical to clean_text() in app.py."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def build_text(row):
    body = row["Body"] if isinstance(row["Body"], str) and row["Body"].strip() else row["Summary"]
    return clean_text(" ".join([str(row["Title"]), str(row["Summary"]), str(body)]))

df["text"] = df.apply(build_text, axis=1)
n_before = len(df)
df = df[df["text"].str.len() > 0]
n_empty_removed = n_before - len(df)
n_before = len(df)
df = df.drop_duplicates(subset="text").reset_index(drop=True)
n_dup_removed = n_before - len(df)
print(f"Empty texts removed: {n_empty_removed} | duplicate articles removed: {n_dup_removed} | records used: {len(df)}")

sns.set_theme(style="whitegrid")
counts = df["Category"].value_counts().reindex(CATEGORIES)

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=counts.values, y=counts.index, color="#1B7F4C", ax=ax)
for i, v in enumerate(counts.values):
    ax.text(v + 5, i, str(v), va="center")
ax.set_title("Articles per selected category"); ax.set_xlabel("Articles"); ax.set_ylabel("")
fig.tight_layout(); fig.savefig(ROOT / "images" / "category_distribution.png", dpi=130); plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
oc = original_counts.sort_values()
ax.barh(oc.index, oc.values, color=["#1B7F4C" if c in CATEGORIES else "#C9C9C9" for c in oc.index])
ax.set_title("Original dataset (green = the six categories used)"); ax.set_xlabel("Articles")
fig.tight_layout(); fig.savefig(ROOT / "images" / "original_category_distribution.png", dpi=130); plt.show()

df["n_words"] = df["text"].str.split().str.len()
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(df["n_words"].clip(upper=2500), bins=50, color="#1B7F4C", ax=ax)
ax.set_title("Article length (words, clipped at 2500)"); ax.set_xlabel("Words")
fig.tight_layout(); fig.savefig(ROOT / "images" / "article_length_distribution.png", dpi=130); plt.show()
print(df["n_words"].describe().round(1))

# Word clouds (real clouds if `wordcloud` is installed, otherwise a top-word frequency panel)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
try:
    from wordcloud import WordCloud
    for ax, cat in zip(axes.ravel(), CATEGORIES):
        txt = " ".join(df.loc[df.Category == cat, "text"])
        ax.imshow(WordCloud(width=600, height=360, background_color="white",
                            collocations=False).generate(txt), interpolation="bilinear")
        ax.set_title(cat); ax.axis("off")
except ImportError:
    for ax, cat in zip(axes.ravel(), CATEGORIES):
        cv_ = CountVectorizer(stop_words="english", max_features=12)
        m = cv_.fit_transform(df.loc[df.Category == cat, "text"])
        top = pd.Series(np.asarray(m.sum(axis=0)).ravel(), index=cv_.get_feature_names_out()).sort_values()
        ax.barh(top.index, top.values, color="#1B7F4C"); ax.set_title(cat)
fig.suptitle("Most frequent words per category", fontsize=14)
fig.tight_layout(); fig.savefig(ROOT / "images" / "wordcloud.png", dpi=110); plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["Category"], test_size=0.20, random_state=RANDOM_STATE, stratify=df["Category"])
print("Train:", len(X_train), "| Test:", len(X_test))
print(pd.DataFrame({"train": y_train.value_counts(), "test": y_test.value_counts()}).reindex(CATEGORIES))

def tfidf():
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=30000,
                           min_df=3, sublinear_tf=True)

def calibrated_svc(C):
    return CalibratedClassifierCV(LinearSVC(C=C, class_weight="balanced", random_state=RANDOM_STATE),
                                  method="sigmoid", cv=5)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
searches = {
    "Multinomial Naive Bayes": (Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB())]),
                                {"clf__alpha": [0.03, 0.1, 0.3, 1.0]}),
    "Logistic Regression": (Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(
                                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))]),
                            {"clf__C": [5, 10, 30, 100]}),
    "Support Vector Classifier (SVC)": (Pipeline([("tfidf", tfidf()), ("clf", calibrated_svc(1.0))]),
                                        {"clf__estimator__C": [0.3, 1.0, 3.0]}),
}
best_params = {}
for name, (pipe, grid) in searches.items():
    t0 = time.time()
    gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=1).fit(X_train, y_train)
    best_params[name] = gs.best_params_
    print(f"{name:34s} best={gs.best_params_}  CV macro-F1={gs.best_score_:.4f}  ({time.time()-t0:.0f}s)")

def make_candidates():
    lr_C = best_params["Logistic Regression"]["clf__C"]
    svc_C = best_params["Support Vector Classifier (SVC)"]["clf__estimator__C"]
    return {
        "Multinomial Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB(
            alpha=best_params["Multinomial Naive Bayes"]["clf__alpha"]))]),
        "Logistic Regression": Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(
            C=lr_C, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))]),
        "Support Vector Classifier (SVC)": Pipeline([("tfidf", tfidf()), ("clf", calibrated_svc(svc_C))]),
        "Ensemble (LogReg + SVC, soft voting)": Pipeline([("tfidf", tfidf()), ("clf", VotingClassifier(
            [("lr", LogisticRegression(C=lr_C, max_iter=2000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
             ("svc", calibrated_svc(svc_C))], voting="soft"))]),
    }

candidates = make_candidates()
scoring = {"accuracy": "accuracy", "macro_precision": "precision_macro", "macro_recall": "recall_macro",
           "macro_f1": "f1_macro", "weighted_f1": "f1_weighted"}
rows = []
for name, pipe in candidates.items():
    res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    rows.append({"Model": name, "Accuracy": res["test_accuracy"].mean(),
                 "Macro Precision": res["test_macro_precision"].mean(),
                 "Macro Recall": res["test_macro_recall"].mean(),
                 "Macro F1": res["test_macro_f1"].mean(), "Macro F1 std": res["test_macro_f1"].std(),
                 "Weighted F1": res["test_weighted_f1"].mean()})
comparison = pd.DataFrame(rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)
comparison.insert(0, "Rank", comparison.index + 1)
print(comparison.round(4).to_string(index=False))

best_name = comparison.loc[0, "Model"]
print("Selected model (highest mean CV Macro F1):", best_name)
comparison.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 4))
sns.barplot(data=comparison, x="Macro F1", y="Model", color="#1B7F4C", ax=ax)
for i, v in enumerate(comparison["Macro F1"]):
    ax.text(v + 0.003, i, f"{v:.4f}", va="center")
ax.set_xlim(0, 1); ax.set_title("5-fold CV Macro F1 (training set)"); ax.set_ylabel("")
fig.tight_layout(); fig.savefig(ROOT / "images" / "model_comparison.png", dpi=130); plt.show()

best_model = make_candidates()[best_name].fit(X_train, y_train)
y_pred = best_model.predict(X_test)
proba = best_model.predict_proba(X_test)
agree = float((best_model.classes_[proba.argmax(axis=1)] == y_pred).mean())
print(f"argmax(predict_proba) agrees with predict() for {agree:.2%} of test articles")

test_metrics = {
    "best_model": best_name,
    "test_accuracy": float(accuracy_score(y_test, y_pred)),
    "test_macro_precision": float(precision_score(y_test, y_pred, average="macro")),
    "test_macro_recall": float(recall_score(y_test, y_pred, average="macro")),
    "test_macro_f1": float(f1_score(y_test, y_pred, average="macro")),
    "test_weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
    "cv_macro_f1": float(comparison.loc[0, "Macro F1"]),
    "best_params": {k: str(v) for k, v in best_params.items()},
    "proba_argmax_agrees_with_predict": agree,
}
print(json.dumps(test_metrics, indent=2))
report_df = pd.DataFrame(classification_report(y_test, y_pred, labels=CATEGORIES,
                                               output_dict=True, zero_division=0)).T
report_df.to_csv(ROOT / "reports" / "classification_report.csv")
print(classification_report(y_test, y_pred, labels=CATEGORIES, digits=4))

cm = confusion_matrix(y_test, y_pred, labels=CATEGORIES)
pd.DataFrame(cm, index=CATEGORIES, columns=CATEGORIES).to_csv(ROOT / "reports" / "confusion_matrix.csv")
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion matrix - {best_name}")
plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
fig.tight_layout(); fig.savefig(ROOT / "images" / "confusion_matrix.png", dpi=130); plt.show()

joblib.dump(best_model, ROOT / "models" / "best_model.joblib", compress=3)
(ROOT / "models" / "best_model_name.txt").write_text(best_name, encoding="utf-8")
(ROOT / "reports" / "best_model_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
stats = {
    "original_records": int(len(raw)), "original_categories": int(raw["Category"].nunique()),
    "number_of_selected_categories": len(CATEGORIES),
    "selected_records": int(raw["Category"].isin(CATEGORIES).sum()),
    "empty_texts_removed": int(n_empty_removed), "duplicates_removed": int(n_dup_removed),
    "records_used": int(len(df)), "training_records": int(len(X_train)), "testing_records": int(len(X_test)),
    "category_counts": {c: int(counts[c]) for c in CATEGORIES},
    "sklearn_version": sklearn.__version__,
}
(ROOT / "reports" / "dataset_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
print("Saved models/best_model.joblib ->", round((ROOT / "models" / "best_model.joblib").stat().st_size / 1e6, 1), "MB")

def classify(pipe, text):
    p = pipe.predict_proba([clean_text(text)])[0]
    out = pd.DataFrame({"Category": pipe.classes_, "Probability (%)": (p * 100).round(2)})
    return out.sort_values("Probability (%)", ascending=False).reset_index(drop=True)

loaded = joblib.load(ROOT / "models" / "best_model.joblib")
assert (loaded.predict(X_test) == y_pred).all()      # saved model == trained model

samples = {
    "Weather": "Climate researchers analyzing decades of records report that heat waves, droughts and extreme rainfall events are becoming more frequent. Forecast models show that shifts in the jet stream can cause weather patterns to stall, leading to prolonged heat or days of flooding in the same region.",
    "Ozone Holes": "Scientists monitoring the Antarctic ozone hole report that chlorofluorocarbons banned under the Montreal Protocol are slowly declining, allowing the stratospheric ozone layer to recover and reducing harmful ultraviolet radiation reaching the surface.",
    "Agriculture & Food": "A new study shows that wheat and rice farmers can increase crop yields by using drought-tolerant varieties and improved fertilizer management, which could help improve food security and reduce hunger in developing countries.",
    "Pollution": "Researchers measured fine particulate matter and nitrogen dioxide in city air and found that traffic emissions and industrial smoke are the main sources of air pollution linked to asthma and heart disease.",
}
demo_out = {}
for expected, text in samples.items():
    r = classify(loaded, text)
    demo_out[expected] = {"final": r.loc[0, "Category"], "text": text,
                          "table": r.set_index("Category")["Probability (%)"].to_dict()}
    print(f"\n=== Example article (expected: {expected})")
    print(r.to_string(index=False))
    print(f"FINAL PREDICTION: {r.loc[0, 'Category'].upper()}  ({r.loc[0, 'Probability (%)']:.2f}%)   | sum = {r['Probability (%)'].sum():.2f}%")
(ROOT / "reports" / "sample_predictions.json").write_text(json.dumps(demo_out, indent=2), encoding="utf-8")
