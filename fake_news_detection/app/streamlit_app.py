
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# Allow `from src import ...` when run from anywhere
APP_DIR = Path(__file__).resolve().parent
PKG_DIR = APP_DIR.parent
sys.path.insert(0, str(PKG_DIR))

from src.preprocessing import TextCleaner                       # noqa: E402
from src.models import HuggingFaceClassifier, AVAILABLE_HF_MODELS  # noqa: E402
from src.cross_reference import CredibleSourceMatcher           # noqa: E402
from src.ocr import extract_text_from_image                     # noqa: E402


DATA_DIR = PKG_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Fake News Detection — Research Prototype",
    page_icon="🕵️",
    layout="wide",
)

st.title("🕵️ Fake News Detection — Research Prototype")
st.caption(
    "Systematic Review of Fake News, Propaganda, and Disinformation "
    "(Plikynas et al., 2025 / JOAIR). Paste a caption or upload a screenshot; "
    "the system classifies it and cross-references verified Indonesian sources."
)

with st.sidebar:
    st.header("⚙️ Configuration")

    model_label = st.selectbox(
        "Classifier",
        options=list(AVAILABLE_HF_MODELS.keys()),
        index=list(AVAILABLE_HF_MODELS.keys()).index("roberta-fake-news"),
        help="Pick a Hugging Face model. First run downloads weights.",
    )
    model_id = AVAILABLE_HF_MODELS[model_label]
    st.code(model_id, language="text")

    top_k = st.slider("Cross-reference top-K", 1, 10, 5)
    build_corpus = st.button("🔄 Refresh verified-source corpus")
    use_embeddings = st.checkbox("Use sentence embeddings (slower, better)", value=False)

    st.divider()
    st.markdown("**Verified sources**")
    for k, v in CredibleSourceMatcher().sources.items():
        st.markdown(f"- [{v['name']}]({v['url']})")

@st.cache_resource(show_spinner="Loading classifier…")
def load_classifier(mid: str) -> HuggingFaceClassifier:
    return HuggingFaceClassifier(model_id=mid)


@st.cache_resource(show_spinner="Building verified-source corpus…")
def load_matcher(use_emb: bool):
    m = CredibleSourceMatcher(use_embeddings=use_emb)
    m.build_corpus()
    return m


cleaner = TextCleaner(language="id", strip_emoji=True, remove_stopwords=False)
clf = load_classifier(model_id)

if build_corpus:
    load_matcher.clear()
matcher = load_matcher(use_embeddings)

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📝 Input")
    tab_text, tab_image, tab_csv = st.tabs(["Text caption", "Image upload", "CSV batch"])

    query_text: str = ""
    source_mode: str = ""

    with tab_text:
        query_text = st.text_area(
            "Paste a social-media caption",
            height=180,
            placeholder="Masukkan caption Instagram / TikTok / X di sini…",
        )
        if query_text:
            source_mode = "text"

    with tab_image:
        img_file = st.file_uploader(
            "Upload a social-media screenshot (PNG / JPG)",
            type=["png", "jpg", "jpeg", "webp"],
            key="img",
        )
        if img_file is not None:
            st.image(img_file, use_container_width=True)
            with st.spinner("Running OCR…"):
                query_text = extract_text_from_image(img_file.getvalue())
            st.text_area("Extracted text (editable)", value=query_text, height=160, key="ocr_out")
            query_text = st.session_state.get("ocr_out", query_text)
            source_mode = "image"

    with tab_csv:
        csv_file = st.file_uploader(
            "Upload a CSV produced by social-media-scrapper",
            type=["csv"],
            key="csv",
        )
        if csv_file is not None:
            df = pd.read_csv(csv_file)
            st.dataframe(df.head(), use_container_width=True)
            col = st.selectbox("Text column", options=list(df.columns),
                               index=(list(df.columns).index("caption")
                                      if "caption" in df.columns else 0))
            if st.button("▶️ Classify entire CSV"):
                with st.spinner(f"Classifying {len(df)} rows…"):
                    cleaned = [cleaner(t) for t in df[col].fillna("").astype(str).tolist()]
                    preds = clf.predict(cleaned)
                df_out = df.copy()
                df_out["prediction"] = preds
                st.success("Done.")
                st.dataframe(df_out, use_container_width=True)
                st.download_button(
                    "📥 Download predictions CSV",
                    data=df_out.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"predictions_{datetime.now():%Y%m%d_%H%M%S}.csv",
                )

with col2:
    st.subheader("🎯 Result")

    if not query_text:
        st.info("Enter a caption or upload an image on the left.")
    else:
        cleaned_text = cleaner(query_text)
        with st.spinner("Classifying…"):
            proba = clf.predict_proba([cleaned_text])[0]
            label_map = clf._label_map or {i: str(i) for i in range(len(proba))}
            top = int(proba.argmax())
            pred_label = label_map[top]
            confidence = float(proba[top])

        st.metric("Prediction", pred_label, f"confidence {confidence*100:.1f}%")

        dist_df = pd.DataFrame({
            "label": [label_map[i] for i in range(len(proba))],
            "score": [float(p) for p in proba],
        }).sort_values("score", ascending=False)
        st.bar_chart(dist_df.set_index("label"))

        metrics_path = DATA_DIR / f"metrics_{model_label}.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                m = json.load(f)
            mcols = st.columns(4)
            mcols[0].metric("Accuracy", f"{m['accuracy']:.3f}")
            mcols[1].metric("Precision", f"{m['precision']:.3f}")
            mcols[2].metric("Recall", f"{m['recall']:.3f}")
            mcols[3].metric("F1-score", f"{m['f1']:.3f}")
            st.caption(f"Loaded from `{metrics_path.name}`")
        else:
            st.caption(
                "No stored validation metrics yet — run the notebook evaluation cell "
                f"and save to `{metrics_path.name}`."
            )
            
st.divider()
st.subheader("🔎 Cross-reference against verified sources")

if query_text:
    with st.spinner("Searching verified sources…"):
        matches = matcher.match(cleaner(query_text), top_k=top_k)

    if not matches:
        st.warning("No matches found.")
    else:
        rows = []
        for hit in matches:
            rows.append({
                "similarity": round(hit.similarity, 3),
                "source": hit.article.source_name,
                "title": hit.article.title,
                "url": hit.article.url,
            })
        mdf = pd.DataFrame(rows)

        st.dataframe(
            mdf,
            use_container_width=True,
            column_config={
                "url": st.column_config.LinkColumn("url", display_text="open"),
                "similarity": st.column_config.ProgressColumn(
                    "similarity", format="%.3f", min_value=0.0, max_value=1.0,
                ),
            },
            hide_index=True,
        )

        top_sim = matches[0].similarity
        if top_sim >= 0.45:
            st.success(
                f"High-similarity match found ({top_sim:.2f}) — "
                "content likely corresponds to a verified article."
            )
        elif top_sim >= 0.20:
            st.warning(
                f"Weak similarity ({top_sim:.2f}) — partial overlap with verified sources. "
                "Manual review recommended."
            )
        else:
            st.error(
                f"No verified source matches this content (best sim {top_sim:.2f}). "
                "Treat as unverified."
            )
else:
    st.caption("Cross-reference runs automatically once you provide input.")
