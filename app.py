import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="NLP Toolkit", page_icon="🧠", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
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
    st.info("⏳ Coming soon...")
