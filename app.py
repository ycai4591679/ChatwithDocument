#Manual Q&A bot powered by chatGPT plus langchain.ipynb
#
#reference:
#https://blog.devgenius.io/chat-with-document-s-using-openai-chatgpt-api-and-text-embedding-6a0ce3dc8bc8
#Sung Kim Mar/09/2023
#
#Chat with Document(s) using OpenAI ChatGPT API and Text Embedding
#How to chat with any documents, PDFs, and books using OpenAI ChatGPT API and Text Embedding

# Import Python Packages
import os
import platform
import openai
import chromadb
import langchain
import magic
import os
import nltk

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain, VectorDBQA
from langchain.document_loaders import GutenbergLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import OnlinePDFLoader
from langchain.document_loaders import PDFMinerLoader
from langchain.document_loaders import UnstructuredFileLoader

disclaimer = ("""
#### Disclaimer and Precautions
######  Use of this tool is at your own risk. The creator cannot be held responsible for any harm caused by this tool.
######  The creator is not responsible for any damages caused to users or third parties as a result of such content.
""")

# Configure Chroma
persist_directory="/data/"

#Convert local file to Embedding (txt file is better)
loader = UnstructuredFileLoader('data/ChatGPT_S_ND.txt')
paldoc = loader.load()

openai.api_key = os.environ.get("OPENAI_API_KEY")

#OPENAI_API_KEY = os.getenv('openai_api_key')
#OPENAI_API_KEY = "openai_api_key"

#local file
text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=30)
paldocs = text_splitter.split_documents(paldoc)

embeddings = OpenAIEmbeddings()

vStore = Chroma.from_documents(paldocs, embeddings)

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# RetrievalQA
model = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo-16k", max_tokens = 256), 
    chain_type="stuff",
    retriever=vStore.as_retriever()
)

#open as a demo Web app using Gradio
import gradio as gr
# 関数を定義
def askandanswer(question, Language): return model.run("Please create a simple answer to the question from  in " + Language + ". [Question] " + question) 
# Webアプリを作成
app = gr.Interface(fn=askandanswer,
                   inputs= [gr.Textbox(placeholder="Please input query "),
                            gr.Dropdown(["中文 Chinese", "英語 English"], label="言語 Language")],
                   outputs="text",
                  title="Chat with Document(s) using OpenAI ChatGPT API and Text Embedding",
    description= "How to chat with any documents, PDFs, and books using OpenAI ChatGPT API and Text Embedding",
    article= (disclaimer),
)
# Webアプリを起動
app.launch()