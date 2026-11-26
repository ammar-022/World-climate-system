"""AI-Based Climate Change Article Classification System - Streamlit dashboard.

The app only LOADS the saved pipeline (models/best_model.joblib); it never retrains.
Run from the project root:  streamlit run app.py
"""
import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

def _find(*relative):
    """First existing path among the candidates (works for the standard folder layout AND a flat one)."""
    for rel in relative:
        for root in (BASE_DIR, Path.cwd()):
            if (root / rel).exists():
                return root / rel
    return BASE_DIR / relative[0]


MODEL_PATH = _find("models/best_model.joblib", "best_model.joblib")
MODEL_NAME_PATH = _find("models/best_model_name.txt", "best_model_name.txt")
METRICS_PATH = _find("reports/best_model_metrics.json", "best_model_metrics.json")
STATS_PATH = _find("reports/dataset_statistics.json", "dataset_statistics.json")

CATEGORIES = ["Climate", "Pollution", "Environment", "Weather", "Ozone Holes", "Agriculture & Food"]
MIN_WORDS_HARD = 10   # below this the input is rejected
MIN_WORDS_SOFT = 30   # below this a reliability warning is shown
PRIMARY = "#1B7F4C"


def clean_text(text):
    """Identical to clean_text() in the training notebook (Section 6)."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def validate_model(pipe):
    """Raise ValueError if the loaded object is not a usable six-category text pipeline."""
    if not hasattr(pipe, "predict_proba") or not hasattr(pipe, "classes_"):
        raise ValueError("The file does not contain a fitted classifier pipeline with predict_proba().")
    if set(pipe.classes_) != set(CATEGORIES):
        raise ValueError(f"The model's classes {list(pipe.classes_)} do not match the six expected categories.")


def classify_text(pipe, text):
    """Return a DataFrame (all six categories, sorted high -> low) of genuine predict_proba() percentages."""
    proba = pipe.predict_proba([clean_text(text)])[0]
    result = pd.DataFrame({"Category": list(pipe.classes_), "Probability (%)": proba * 100.0})
    return result.sort_values("Probability (%)", ascending=False).reset_index(drop=True)


@st.cache_resource(show_spinner="Loading the trained model...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found: models/best_model.joblib")
    pipe = joblib.load(MODEL_PATH)
    validate_model(pipe)
    return pipe


@st.cache_data(show_spinner=False)
def load_reports():
    """Load dashboard statistics from reports/. Returns (metrics, stats, best_model_name, problems)."""
    problems, metrics, stats, name = [], {}, {}, None
    for path, label in ((METRICS_PATH, "metrics"), (STATS_PATH, "stats")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            problems.append(f"Report not found: reports/{path.name}")
            continue
        except (json.JSONDecodeError, OSError) as exc:
            problems.append(f"Report reports/{path.name} could not be read ({exc.__class__.__name__}).")
            continue
        if label == "metrics":
            metrics = data
        else:
            stats = data
    try:
        name = MODEL_NAME_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        name = metrics.get("best_model")
    return metrics, stats, name, problems


def fmt(value, kind="int"):
    if value is None:
        return "N/A"
    return f"{value:,}" if kind == "int" else f"{value:.4f}"


def probability_chart(result):
    ordered = result.iloc[::-1]  # highest probability drawn at the top
    colors = [PRIMARY if c == result.loc[0, "Category"] else "#9DB8A8" for c in ordered["Category"]]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.barh(ordered["Category"], ordered["Probability (%)"], color=colors)
    for bar, value in zip(bars, ordered["Probability (%)"]):
        ax.text(min(value + 1.0, 99), bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Predicted probability (%)")
    ax.set_title("Probability comparison across the six categories")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


EXAMPLES = {
    "(choose an example)": "",
    "Weather": "Climate researchers analyzing decades of records report that heat waves, droughts and extreme rainfall events are becoming more frequent. Forecast models show that shifts in the jet stream can cause weather patterns to stall, leading to prolonged heat or days of flooding in the same region.",
    "Ozone Holes": "Scientists monitoring the Antarctic ozone hole report that chlorofluorocarbons banned under the Montreal Protocol are slowly declining, allowing the stratospheric ozone layer to recover and reducing harmful ultraviolet radiation reaching the surface.",
    "Agriculture & Food": "A new study shows that wheat and rice farmers can increase crop yields by using drought-tolerant varieties and improved fertilizer management, which could help improve food security and reduce hunger in developing countries.",
    "Pollution": "Researchers measured fine particulate matter and nitrogen dioxide in city air and found that traffic emissions and industrial smoke are the main sources of air pollution linked to asthma and heart disease.",
}


def clear_input():
    st.session_state["article"] = ""
    st.session_state["example"] = "(choose an example)"
    st.session_state.pop("result", None)


def load_example():
    text = EXAMPLES.get(st.session_state.get("example"), "")
    if text:
        st.session_state["article"] = text
        st.session_state.pop("result", None)


def main():
    st.set_page_config(page_title="AI-Based Climate Change Article Classification System",
                       page_icon="🌍", layout="wide")
    st.markdown(
        f"""<style>
        .block-container {{padding-top: 2rem; max-width: 1100px;}}
        .final-box {{background: {PRIMARY}; color: white; padding: 1.1rem 1.4rem; border-radius: 12px;
                     text-align: center; font-size: 1.6rem; font-weight: 700; letter-spacing: .03em;}}
        .final-sub {{font-size: .95rem; font-weight: 400; opacity: .9;}}
        </style>""", unsafe_allow_html=True)

    st.title("AI-Based Climate Change Article Classification System")
    st.caption("An NLP and Machine Learning system that classifies English climate-related articles into six categories.")

    # ---- Load model and reports (never retrain) ----
    model, model_error = None, None
    try:
        model = load_model()
    except FileNotFoundError as exc:
        model_error = f"{exc}. Run the notebook first (or add the model file to the repository)."
    except Exception as exc:  # corrupted file, version mismatch, wrong object...
        model_error = f"The model file could not be loaded: {exc}"
    metrics, stats, model_name, report_problems = load_reports()

    with st.sidebar:
        st.header("About")
        st.write("Paste an English article or paragraph. The saved TF-IDF + classifier pipeline predicts "
                 "which of six categories it most resembles.")
        st.write("**Categories:** " + ", ".join(CATEGORIES))
        st.warning("**Limitations.** The model knows only these six categories and always picks one of them, "
                   "even for unrelated text. The highest probability is not proof that an article belongs to "
                   "that category. It is designed for climate-related and environmental text.")

    # ---- Dashboard statistics (loaded from reports/) ----
    st.subheader("Model dashboard")
    cols = st.columns(6)
    cols[0].metric("Categories", fmt(stats.get("number_of_selected_categories")))
    cols[1].metric("Training records", fmt(stats.get("training_records")))
    cols[2].metric("Testing records", fmt(stats.get("testing_records")))
    cols[3].metric("Best model", model_name or "N/A")
    cols[4].metric("Test accuracy", fmt(metrics.get("test_accuracy"), "float"))
    cols[5].metric("Test Macro F1", fmt(metrics.get("test_macro_f1"), "float"))
    for problem in report_problems:
        st.warning(problem + " Statistics shown as N/A.")

    st.divider()
    if model_error:
        st.error(model_error)

    # ---- Input ----
    st.selectbox("Load an example article (optional)", list(EXAMPLES), key="example", on_change=load_example)
    st.text_area("Enter Your Article or Paragraph", key="article", height=240,
                 placeholder="Paste an English article or paragraph here...")
    b1, b2, _ = st.columns([1, 1, 4])
    classify = b1.button("Classify Article", type="primary", use_container_width=True)
    b2.button("Clear Input", on_click=clear_input, use_container_width=True)

    if classify:
        text = st.session_state.get("article", "")
        n_words = len(text.split())
        st.session_state.pop("result", None)
        if model is None:
            st.error("Cannot classify because the model is not available (see the error above).")
        elif not text.strip():
            st.error("Please enter an article or paragraph first. The input is empty.")
        elif n_words < MIN_WORDS_HARD:
            st.error(f"The text is too short ({n_words} words). Please enter at least {MIN_WORDS_HARD} words "
                     f"(a full paragraph works best).")
        else:
            try:
                result = classify_text(model, text)
                st.session_state["result"] = {"table": result, "words": n_words}
            except Exception as exc:
                st.error(f"An unexpected error occurred while classifying the text: {exc}")

    # ---- Results ----
    saved = st.session_state.get("result")
    if saved:
        result, n_words = saved["table"], saved["words"]
        if n_words < MIN_WORDS_SOFT:
            st.warning(f"Short input ({n_words} words): predictions for very short texts are less reliable.")
        top = result.loc[0]
        st.markdown(
            f'<div class="final-box"><div class="final-sub">FINAL PREDICTION</div>{top["Category"].upper()}'
            f'<div class="final-sub">{top["Probability (%)"]:.2f}% predicted probability</div></div>',
            unsafe_allow_html=True)
        st.write("")
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Category probabilities")
            for _, row in result.iterrows():
                pct = float(row["Probability (%)"])
                st.write(f"**{row['Category']}** - {pct:.2f}%")
                st.progress(min(max(pct / 100.0, 0.0), 1.0))
        with right:
            st.subheader("Probability comparison")
            fig = probability_chart(result)
            st.pyplot(fig)
            plt.close(fig)
        st.caption(f"Total of the six percentages: {result['Probability (%)'].sum():.2f}%. "
                   "The system always chooses one of its six categories; a high percentage does not prove the "
                   "article truly belongs to that category.")


if __name__ == "__main__":
    main()
