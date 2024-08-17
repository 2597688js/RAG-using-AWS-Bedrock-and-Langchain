"""
Author : Janarddan Sarkar
file_name : app.py
date : 18-08-2024
description :
"""

import boto3
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import Bedrock

from dotenv import find_dotenv, load_dotenv

# --- Data Ingestion ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# --- Vector Embedding And Vector Store ---
from langchain_community.vectorstores import FAISS

# --- LLm Models ---
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv(find_dotenv(), override=True)

# --- Bedrock Clients ---
bedrock = boto3.client(service_name="bedrock-runtime", region_name='ap-south-1')
embeddings = OpenAIEmbeddings()

# --- Data ingestion ---
def data_ingestion():
    # Reading pdfs from the data folder
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,
                                                   chunk_overlap=1000)

    docs = text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    try:
        vectorstore_faiss = FAISS.from_documents(
            docs,
            embeddings
        )
        vectorstore_faiss.save_local("faiss_index")
        return vectorstore_faiss
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

# LLM models
def get_mistral_llm():
    # Mistral model
    # create the Mistral Model
    llm = Bedrock(model_id="mistral.mistral-large-2402-v1:0", client=bedrock)

    return llm


def get_llama2_llm():
    # LLAMA 2 llm
    # create the LLAMA2 Model
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock,
                  model_kwargs={'max_gen_len': 512})

    return llm


prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}  # top-k
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']


def main():
    st.set_page_config("Chat PDF")

    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Mistral Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            llm = get_mistral_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            llm = get_llama2_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")


if __name__ == "__main__":
    main()
