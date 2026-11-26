# AI-Based Climate Change Article Classification System

An NLP and Machine Learning system that classifies English climate-related articles into **six** categories. For every article it shows the **percentage for each category** and then announces the **final category = the one with the highest percentage**, in a Streamlit dashboard.

## 1. Project introduction
The user pastes an English article or paragraph. A saved scikit-learn pipeline (TF-IDF + Logistic Regression) predicts a probability for each of the six categories; the highest-probability category is the final prediction.

## 2. Project objectives
- Build a supervised text-classification pipeline on `SciDCC.csv`.
- Compare classical models fairly (hyper-parameter search + cross-validation on the training set only).
- Evaluate the chosen model once on an untouched test set.
- Provide a Streamlit app that loads the saved model (no retraining) and displays genuine `predict_proba()` percentages.

## 3. Dataset
`SciDCC.csv`: 11,539 ScienceDaily articles in 20 categories (`Date, Link, Title, Summary, Body, Category, Year`). Missing values: only `Body` (5 records).

## 4. The six categories (the only ones used)
| Category | Records |
|---|---:|
| Climate | 700 |
| Pollution | 945 |
| Environment | 478 |
| Weather | 719 |
| Ozone Holes | 623 |
| Agriculture & Food | 844 |
| **Total** | **4,309** |

## 5. Preprocessing
- Text = `Title` + `Summary` + `Body` (Body falls back to Summary when empty).
- Lower-casing, removal of URLs / e-mails / punctuation, whitespace normalisation.
- `Date`, `Link`, `Year`, `Category` are never used as features.
- 0 empty texts and 2 duplicate articles removed -> **4,307** records.

## 6. Feature extraction
`TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=30000, min_df=3, sublinear_tf=True)` inside an sklearn `Pipeline`, so it is fitted on training data only (no leakage).

## 7. Train-test split
80/20, `random_state=42`, stratified: **3,445 training / 862 testing** articles.

## 8. Models trained
1. Multinomial Naive Bayes
2. Logistic Regression (`class_weight="balanced"`)
3. Support Vector Classifier: `LinearSVC` + `CalibratedClassifierCV` (sigmoid, 5-fold) for genuine probabilities
4. Soft-voting ensemble of Logistic Regression + calibrated SVC

Hyper-parameters (NB `alpha`, LR `C`, SVC `C`) were tuned with `GridSearchCV` (5-fold stratified, Macro F1) on the **training set only**. Selected: NB alpha=0.03, LR C=5, SVC C=1.0.

## 9. Model comparison (5-fold stratified CV, training set)
| Rank | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **Logistic Regression** | 0.7762 | 0.7681 | 0.7605 | **0.7617** | 0.7739 |
| 2 | Support Vector Classifier | 0.7788 | 0.7717 | 0.7571 | 0.7588 | 0.7736 |
| 3 | Ensemble (LogReg + SVC) | 0.7753 | 0.7675 | 0.7560 | 0.7577 | 0.7714 |
| 4 | Multinomial Naive Bayes | 0.6961 | 0.6886 | 0.6729 | 0.6750 | 0.6908 |

The model with the highest mean CV Macro F1 was selected: **Logistic Regression**. The top three are within one fold-standard-deviation (about 0.01) of each other; Naive Bayes is clearly weaker.

## 10. Final test results (untouched test set, 862 articles)
| Metric | Value |
|---|---:|
| Accuracy | 0.7738 |
| Macro Precision | 0.7602 |
| Macro Recall | 0.7549 |
| Macro F1 | 0.7545 |
| Weighted F1 | 0.7698 |

| Category | Precision | Recall | F1 | Test articles |
|---|---:|---:|---:|---:|
| Climate | 0.6471 | 0.7071 | 0.6758 | 140 |
| Pollution | 0.8129 | 0.7354 | 0.7722 | 189 |
| Environment | 0.5753 | 0.4375 | 0.4970 | 96 |
| Weather | 0.7843 | 0.8333 | 0.8081 | 144 |
| Ozone Holes | 0.9322 | 0.8871 | 0.9091 | 124 |
| Agriculture & Food | 0.8093 | 0.9290 | 0.8650 | 169 |

Ozone Holes and Agriculture & Food are classified best; Environment is the hardest category because it overlaps thematically with Climate, Pollution and Agriculture & Food. The final label is the highest-probability category (it matched the classifier's own `.predict()` for 100% of test articles).

## 11. Project structure
```
SciDCC_Climate_Article_Classifier/
├── app.py                      Streamlit dashboard (loads the saved model)
├── train.py                    Script version of the notebook (same code)
├── data/SciDCC.csv
├── notebooks/SciDCC_Climate_Change_Classification.ipynb
├── models/                     best_model.joblib, best_model_name.txt
├── reports/                    dataset_statistics.json, model_comparison.csv,
│                               best_model_metrics.json, classification_report.csv,
│                               confusion_matrix.csv, sample_predictions.json
├── images/                     category_distribution, wordcloud, model_comparison,
│                               confusion_matrix, article_length_distribution, original_category_distribution
├── requirements.txt, requirements-dev.txt, runtime.txt
├── README.md, LICENSE, .gitignore, .gitattributes
├── .streamlit/config.toml
└── .github/workflows/python-check.yml
```
**Important:** `app.py` must sit next to the `models/` and `reports/` folders. The error "Model file not found: models/best_model.joblib" appears when `best_model.joblib` is not inside a `models/` folder beside `app.py`.

## 12. Installation
Python 3.12 is recommended. The model was saved with scikit-learn 1.9.1, which `requirements.txt` pins.
```bash
cd SciDCC_Climate_Article_Classifier
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 13. Run the Streamlit app
```bash
streamlit run app.py
```

## 14. Re-train (optional)
```bash
pip install -r requirements-dev.txt
python train.py            # or open the notebook: Kernel -> Restart & Run All
```
Takes about 6-8 minutes on one CPU core and rewrites `models/`, `reports/` and `images/`. Installing `wordcloud` makes the notebook draw real word clouds; otherwise a top-word panel is saved.

## 15. GitHub upload
```bash
git init
git add .
git commit -m "AI-Based Climate Change Article Classification System"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
The largest file is `data/SciDCC.csv` (~44 MB, under GitHub's 100 MB limit); the model is only ~1.8 MB. The app needs only `models/` and `reports/`.

## 16. Streamlit Community Cloud deployment
1. Push the repository to GitHub.
2. Open <https://share.streamlit.io> -> **Create app**.
3. Choose the repository, branch `main`, main file `app.py`.
4. In **Advanced settings** choose **Python 3.12**, then **Deploy**.

## 17. Conclusion
The system takes an English article, computes a probability for **all six categories**, and reports the category with the highest percentage as the **final prediction**. Logistic Regression on TF-IDF features was selected by cross-validation and reached **77.4% accuracy and 0.7545 Macro F1** on 862 unseen test articles. Real output of the saved model for the example article from the app screenshot:

> *"Climate researchers analyzing decades of records report that heat waves, droughts and extreme rainfall events are becoming more frequent. Forecast models show that shifts in the jet stream can cause weather patterns to stall, leading to prolonged heat or days of flooding in the same region."*

| Category | Probability |
|---|---:|
| **Weather** | **98.80%** |
| Climate | 0.34% |
| Environment | 0.29% |
| Ozone Holes | 0.23% |
| Pollution | 0.19% |
| Agriculture & Food | 0.16% |
| **Total** | 100.00% |

**FINAL PREDICTION: WEATHER (98.80%)**

Other examples from the same model (see `reports/sample_predictions.json`):

| Article topic | Final prediction | Highest probability |
|---|---|---:|
| Antarctic ozone layer recovery | OZONE HOLES | 99.88% |
| Drought-tolerant wheat and rice, food security | AGRICULTURE & FOOD | 85.68% |
| Particulate matter and traffic emissions | POLLUTION | 97.66% |

The model always chooses one of its six categories, so a high percentage means "most similar to that category", not proof of membership.

## License
MIT - see `LICENSE` (replace `[YOUR NAME]`).
