import re

import joblib
import nltk
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

st.set_page_config(page_title="Kindle Review Sentiment", page_icon="📚")


@st.cache_resource
def load_artifacts():
    nltk.download("stopwords", quiet=True)
    wv = KeyedVectors.load("w2v.kvmodel")
    clf = joblib.load("rf_model.joblib")
    stops = set(stopwords.words("english"))
    return wv, clf, stops


wv, clf, STOPS = load_artifacts()


def embed(text: str) -> np.ndarray:
    """Same preprocessing as the training notebook, then average word vectors."""
    text = text.lower()
    text = re.sub(r"[^a-z A-Z 0-9]", "", text)
    text = " ".join(w for w in text.split() if w not in STOPS)
    text = BeautifulSoup(text, "html.parser").get_text()
    tokens = simple_preprocess(text)
    vecs = [wv[w] for w in tokens if w in wv.key_to_index]
    if not vecs:
        return np.zeros(wv.vector_size)
    return np.mean(vecs, axis=0)


st.title("📚 Kindle Review Sentiment")
st.write("Paste a book review and the model will predict whether it is **positive** or **negative**.")

review = st.text_area("Your review", height=160, placeholder="This book was a gripping read from start to finish...")

if st.button("Predict", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        vec = embed(review).reshape(1, -1)
        proba = clf.predict_proba(vec)[0]
        pos = float(proba[list(clf.classes_).index(1)])
        label = "Positive 😊" if pos >= 0.5 else "Negative 😞"
        confidence = pos if pos >= 0.5 else 1 - pos

        if pos >= 0.5:
            st.success(f"**{label}**  ·  confidence {confidence:.0%}")
        else:
            st.error(f"**{label}**  ·  confidence {confidence:.0%}")
        st.progress(pos, text=f"Positive probability: {pos:.0%}")

st.caption("Word2Vec (200-d, averaged) + Random Forest · ~79% test accuracy · reviews rated 4-5 stars count as positive.")
