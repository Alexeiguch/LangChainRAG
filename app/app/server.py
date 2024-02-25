#!/usr/bin/env python
"""Example LangChain server exposes a retriever."""
from fastapi import FastAPI
from langchain_core.runnables import RunnableLambda
from langchain.document_loaders.url_playwright import PlaywrightURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ollama import RAGRetriever
from langserve import add_routes

from langchain.chat_models.openai import ChatOpenAI

app = FastAPI(
    title="LangChain Server - LLama2 Retriever",
    version="1.0",
    description="Run up a simple api server using Langchain's Runnable interfaces that instantiates a Llama2 Retriever Model to answer questions about Promptior.",
)

playwright = PlaywrightURLLoader([
    "https://www.promptior.ai/", 
    "https://www.promptior.ai/about/"], headless=False)

text_splitter = RecursiveCharacterTextSplitter()

model = RAGRetriever(text_splitter, playwright)
docs = model.load_context()
vectors = model.vectorize(docs)

prompt = """Answer the following question based only on the provided context:

            <context>
            {context}
            </context>

            Question: {input}"""
model.create_retrieval_chain(vectors, prompt)

def invoke(input: str):
    return model.invoke(input)

retriever = RunnableLambda(invoke)

# Adds routes to the app for using the retriever under:
# /invoke
# /batch
# /stream
add_routes(app, retriever, path="/llama")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)