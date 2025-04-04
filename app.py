import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryByteStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

# Initialize components
llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()
kv_store = InMemoryByteStore()
vectorstore = Chroma(embedding_function=embeddings)

# Load documents
def load_docs(directory):
    loader = DirectoryLoader(directory)
    return loader.load()

doc_folder = "/Users/Manas Goel/Documents/RAG/docs"
dir_docs = load_docs(doc_folder)

# Configure text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Create parent document retriever
parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    document_compressor=text_splitter,
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000),
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
    byte_store=kv_store,
    search_kwargs={"k": 5}
)

# Initialize retriever
docs = parent_retriever.add_documents(dir_docs)
retriever = parent_retriever

# Create prompt template
template = """You are a chatbot that answers questions about G19 Studio based only on the following context. If you do not know the answer, state that the user should contact a G19 Studio team member.
{context}

Question: {input}
"""

prompt = ChatPromptTemplate.from_template(template)
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Core function to get response
def get_response(prompt: str):
    try:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": prompt})
        print("Response time: ", time.process_time() - start)
        return {
            "answer": response["answer"],
            "context": [doc.page_content for doc in response["context"]]
        }
    except Exception as e:
        raise Exception(str(e))