import streamlit as st
from transformers import pipeline
import pandas as pd

st.set_page_config(page_title="NLP Toolkit", page_icon="🧠", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    color: #f8fafc;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 NLP Toolkit")
st.markdown("Explore core NLP tasks powered by HuggingFace Transformers.")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["💬 Sentiment Analysis", "📝 Summarization", "🔍 Named Entity Recognition"])

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_ner_model():
    return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

ENTITY_COLORS = {
    "PER": "#a8d8ea",
    "ORG": "#ffd3b6",
    "LOC": "#d4edda",
    "MISC": "#e2d5f1"
}

def render_ner_html(text, entities):
    html = ""
    prev = 0
    for ent in entities:
        start, end = ent['start'], ent['end']
        label = ent['entity_group']
        color = ENTITY_COLORS.get(label, "#eeeeee")
        html += text[prev:start]
        html += f'<mark style="background:{color};padding:2px 6px;border-radius:4px;font-weight:600;">{text[start:end]} <sup style="font-size:0.65rem;">{label}</sup></mark>'
        prev = end
    html += text[prev:]
    return html

with tab1:
    st.subheader("💬 Sentiment Analysis")
    st.markdown("Detects whether a piece of text carries a **positive** or **negative** sentiment using DistilBERT.")
    user_input = st.text_area("Enter text", placeholder="e.g. This product is absolutely amazing!", height=150)
    if st.button("Analyze Sentiment", use_container_width=True):
        if user_input.strip():
            with st.spinner("Running model..."):
                classifier = load_sentiment_model()
                result = classifier(user_input)[0]
            label = result['label']
            score = result['score']
            if label == "POSITIVE":
                st.success(f"✅ **{label}** — Confidence: {score:.2%}")
            else:
                st.error(f"❌ **{label}** — Confidence: {score:.2%}")
        else:
            st.warning("Please enter some text first.")

with tab2:
    st.subheader("📝 Text Summarization")
    st.markdown("Condenses long articles or paragraphs into a short summary using **BART** (Facebook).")
    text_to_summarize = st.text_area("Paste your text here", placeholder="Paste a long article, paragraph, or document...", height=200, key="sum_input")
    col1, col2 = st.columns(2)
    with col1:
        min_len = st.slider("Min summary length (words)", 30, 100, 50)
    with col2:
        max_len = st.slider("Max summary length (words)", 100, 300, 150)
    if st.button("Summarize", use_container_width=True):
        if text_to_summarize.strip():
            if len(text_to_summarize.split()) < 30:
                st.warning("Text is too short to summarize. Try pasting a longer paragraph.")
            else:
                with st.spinner("Summarizing..."):
                    summarizer = load_summarizer()
                    summary = summarizer(text_to_summarize, max_length=max_len, min_length=min_len, do_sample=False)
                st.subheader("📌 Summary")
                st.write(summary[0]['summary_text'])
        else:
            st.warning("Please paste some text first.")

with tab3:
    st.subheader("🔍 Named Entity Recognition")
    st.markdown("Identifies and classifies **people, organizations, locations, and misc entities** in text using BERT-NER.")
    st.markdown("""
    <div style='display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap;'>
        <span style='background:#a8d8ea;padding:3px 10px;border-radius:4px;font-weight:600;'>PER &nbsp;Person</span>
        <span style='background:#ffd3b6;padding:3px 10px;border-radius:4px;font-weight:600;'>ORG &nbsp;Organization</span>
        <span style='background:#d4edda;padding:3px 10px;border-radius:4px;font-weight:600;'>LOC &nbsp;Location</span>
        <span style='background:#e2d5f1;padding:3px 10px;border-radius:4px;font-weight:600;'>MISC &nbsp;Miscellaneous</span>
    </div>
    """, unsafe_allow_html=True)
    ner_input = st.text_area("Enter text", placeholder="e.g. Elon Musk founded SpaceX in Hawthorne, California.", height=150, key="ner_input")
    if st.button("Extract Entities", use_container_width=True):
        if ner_input.strip():
            with st.spinner("Identifying entities..."):
                ner_model = load_ner_model()
                entities = ner_model(ner_input)
            if entities:
                st.subheader("Annotated Text")
                html = render_ner_html(ner_input, entities)
                st.markdown(f"<div style='line-height:2.2;font-size:1.05rem;padding:12px;background:#f9f9f9;border-radius:8px;'>{html}</div>", unsafe_allow_html=True)
                st.subheader("Entities Found")
                df = pd.DataFrame([{
                    "Entity": e['word'],
                    "Type": e['entity_group'],
                    "Confidence": f"{e['score']:.2%}"
                } for e in entities])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No named entities found in the text.")
        else:
            st.warning("Please enter some text first.")

# Sidebar
with st.sidebar:
    st.header("📖 About")
    st.markdown("""
    This toolkit runs **three transformer-based NLP models** locally via HuggingFace Pipelines.
    No API keys needed.
    """)
    st.markdown("---")
    st.subheader("🤖 Models Used")
    st.markdown("""
    - **Sentiment** — `distilbert-base-uncased-finetuned-sst-2-english`
    - **Summarizer** — `facebook/bart-large-cnn`
    - **NER** — `dslim/bert-base-NER`
    """)
    st.markdown("---")
    st.subheader("⚡ Tech Stack")
    st.markdown("""
    - HuggingFace `transformers`
    - PyTorch (inference backend)
    - Streamlit
    """)
    st.markdown("---")
    st.markdown("💡 Models are cached after first load — subsequent runs are instant.")
