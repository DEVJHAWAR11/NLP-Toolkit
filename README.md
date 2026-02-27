# NLP Toolkit 🧠

A multi-task NLP web app built with HuggingFace Transformers and Streamlit. Runs three transformer-based models entirely locally — no API keys required.

---

## Features

| Task | Model | What it does |
|---|---|---|
| 💬 Sentiment Analysis | `distilbert-base-uncased-finetuned-sst-2-english` | Classifies text as positive or negative with confidence score |
| 📝 Text Summarization | `facebook/bart-large-cnn` | Condenses long documents into concise summaries |
| 🔍 Named Entity Recognition | `dslim/bert-base-NER` | Extracts and classifies people, orgs, locations from text |

---

## Run Locally

```bash
git clone https://github.com/DEVJHAWAR11/NLP-Toolkit.git
cd NLP-Toolkit
pip install -r requirements.txt
streamlit run app.py
```

> Models are downloaded from HuggingFace Hub on first run and cached automatically.

---

## Tech Stack

- HuggingFace `transformers`
- PyTorch (CPU inference)
- Streamlit

---

## Author

**Dev Jhawar** — [GitHub](https://github.com/DEVJHAWAR11) | [LinkedIn](https://linkedin.com/in/dev-jhawar11)
