import streamlit as st

from report_analyzer import extract_text
from vector_store import create_vector_store

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.vectorstores import FAISS

from prompt_templates import financial_prompt

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):

    def __init__(self):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


embeddings = LocalEmbeddings()

st.set_page_config(
    page_title="AI Financial Report Analyzer",
    layout="wide"
)

st.title("📊 AI Financial Report Analyzer")

uploaded_file = st.file_uploader(
    "Upload Annual Report PDF",
    type=["pdf"]
)

api_key = st.text_input(
    "Gemini API Key",
    type="password"
)

if uploaded_file and api_key:

    with st.spinner("Extracting report data..."):

        text = extract_text(uploaded_file)

        create_vector_store(
    text,
    api_key
)

    st.success("✅ Report Processed Successfully")

    question = st.text_input(
        "Ask Anything About Report"
    )

if question:

    db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

    docs = db.similarity_search(
            question,
            k=4
        )

    context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

    final_prompt = financial_prompt.format(
            context=context,
            question=question
        )

    model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )

    response = model.invoke(final_prompt)

    st.subheader("📈 Analysis")

    st.write(response.content)