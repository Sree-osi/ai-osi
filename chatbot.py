import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOllama
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the PDF data
src_dir = "C:/Users/Sreeja_R1/osi/story.pdf"
loader = PyPDFLoader(src_dir)
data = loader.load()
logger.info(f"Number of files loaded: {len(data)}")

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
all_splits = text_splitter.split_documents(documents=data)
logger.info(f"Number of documents: {len(all_splits)}")

# Set up embeddings and retrievers
model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceBgeEmbeddings(model_name=model_name)

bm25_retriever = BM25Retriever.from_documents(all_splits)
bm25_retriever.k = 2  # Retrieve top 1 result
db = FAISS.from_documents(documents=all_splits, embedding=embeddings)
semantic_retriever = db.as_retriever(search_kwargs={"k": 2})
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, semantic_retriever], weights=[0.5, 0.5])

# Initialize the LLM
llm = ChatOllama(model="llama3", temperature=0)

# Set up the prompt template
prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You have to answer the question, with respect to the context provided in it, start your answer with keyword 'ANSWERING'<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {question} 
Context: {context}
Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id> 
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type="stuff",
    retriever=ensemble_retriever,
    chain_type_kwargs={"prompt": PROMPT}
)

# Streamlit app
st.title("PDF Q&A Chatbot")
st.write("Ask questions based on the content of the PDF document.")

# Input box for user query
user_query = st.text_input("Enter your question:")

if st.button("Submit"):
    if user_query:
        with st.spinner("Fetching answer..."):
            # Start the timer
            start_time = time.time()

            # Use caching for retrieval
            @st.cache_data
            def get_answer(query):
                docs = ensemble_retriever.get_relevant_documents(query)
                if docs:
                    context = "\n".join([doc.page_content for doc in docs])
                    answer = qa_chain.run(query)  # Use the qa_chain to get the answer
                    return context, answer
                return None, "No relevant documents found."

            context, answer = get_answer(user_query)

            # Stop the timer
            end_time = time.time()
            elapsed_time = end_time - start_time

            st.write("**Context Retrieved:**")
            st.write(context)
            st.write("**Answer:**")
            st.write(answer)
            st.write(f"**Response Time:** {elapsed_time:.2f} seconds")
    else:
        st.write("Please enter a question.")
