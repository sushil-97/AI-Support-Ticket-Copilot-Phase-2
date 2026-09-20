import streamlit as st
import joblib
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

support_documents = [
    """
    Billing - Duplicate Charge

    If a customer reports being charged twice for the same transaction,
    verify the transaction history and payment records.

    If a duplicate charge is confirmed, the billing team should initiate
    a refund for the duplicate transaction.

    Refunds may take 5-7 business days to appear in the customer's account.
    """,

    """
    Billing - Failed Payment

    If a customer's payment fails, ask them to verify their payment
    information and ensure sufficient funds are available.

    The customer may retry the payment or use a different payment method.

    If the problem continues, the ticket should be routed to the
    Billing & Payments team.
    """,

    """
    Technical Support - Login Issue

    If a customer cannot log in, first ask them to verify their email
    address and password.

    The customer should try resetting their password using the
    Forgot Password option.

    If the issue continues after a password reset, the ticket should
    be escalated to Technical Support.
    """,

    """
    Technical Support - Website Outage

    If the production website is unavailable for multiple customers,
    treat the issue as a high-priority technical incident.

    The technical team should investigate service availability,
    server health, and recent deployments.

    Widespread outages should be escalated immediately.
    """,

    """
    Refund - Product Return

    Customers requesting a product return should provide their order
    number and reason for the return.

    Eligible products can be returned within 30 days of purchase.

    After the returned product is received and inspected, the refund
    will be processed to the original payment method.
    """,

    """
    Refund - Refund Status

    Customers asking about an existing refund should provide their
    order number or refund reference.

    Approved refunds normally take 5-7 business days to appear in
    the original payment method.

    If the refund has not appeared after this period, the case should
    be reviewed by the Returns & Exchanges team.
    """
]

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()

document_embeddings = embedding_model.encode(
    support_documents
)

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
