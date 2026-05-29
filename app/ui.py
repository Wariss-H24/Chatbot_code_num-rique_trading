import streamlit as st
import os

from chargement import charger_pdf, decouper_texte
from indexation import indexer_pdf
from chatbot import search, generate

st.set_page_config(page_title="RAG - Code du Numérique", page_icon="📘")

st.title("📘 Chatbot IA - Code du Numérique")
st.write("Pose tes questions sur le document officiel")

# =========================
# UPLOAD PDF
# =========================
uploaded_file = st.file_uploader("📄 Charger le PDF", type="pdf")

if uploaded_file:
    pdf_path = f"../documents/{uploaded_file.name}"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF chargé avec succès ✔️")

    if st.button("🚀 Indexer le document"):
        with st.spinner("Indexation en cours..."):
            indexer_pdf(pdf_path)
        st.success("Indexation terminée ✔️")

# =========================
# CHAT
# =========================
st.markdown("---")

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

question = st.text_input("💬 Pose ta question")

if question and question != st.session_state.last_question:
    st.session_state.last_question = question
    with st.spinner("Recherche dans le document..."):
        docs, metas = search(question)
        answer = generate(question, docs)

    st.subheader("🧠 Réponse")
    st.write(answer)

    st.subheader("📚 Sources")
    for i, m in enumerate(metas):
        st.write(f"- {m.get('source', 'document')}")
