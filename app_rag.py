import streamlit as st
import joblib
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(
    page_title="AI Support Ticket Copilot - RAG",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Support Ticket Copilot")

st.subheader("ML + RAG Version")

st.write(
    "This version combines ticket classification, "
    "urgency prediction, semantic retrieval, and "
    "LLM-generated support responses."
)

st.success("RAG application environment loaded successfully!")
