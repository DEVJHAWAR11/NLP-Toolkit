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
    st.info("⏳ Coming soon...")

with tab3:
    st.subheader("🔍 Named Entity Recognition")
    st.info("⏳ Coming soon...")
