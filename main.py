import streamlit as st  # Streamlit is a Python library for building interactive web applications.
from PyPDF2 import PdfReader  # PyPDF2 is a pure-python library built as a PDF toolkit. It can be used for extracting text from PDF files.
from langchain.text_splitter import RecursiveCharacterTextSplitter  # LangChain is a framework for building applications with large language models (LLMs). The RecursiveCharacterTextSplitter is used to split the extracted PDF text into smaller, manageable chunks.
import os  # The os module provides a way to interact with the operating system.
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # LangChain extension for interacting with Google's Generative AI models.
import google.generativeai as genai  # Google's library for accessing their generative AI models.
from langchain.vectorstores import FAISS  # FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.
from langchain_google_genai import ChatGoogleGenerativeAI  # LangChain extension for using Google's Generative AI models in a conversational manner.
from langchain.chains.question_answering import load_qa_chain  # LangChain module for loading a pre-configured question-answering chain.
from langchain.prompts import PromptTemplate  # LangChain module for defining custom prompts for language models.
from dotenv import load_dotenv  # The python-dotenv library is used to load environment variables from a .env file.

# Load environment variables from the .env file
load_dotenv()

# Get the Google API key from the environment variables
os.getenv("GOOGLE_KEY")

# Configure the Google Generative AI with the API key
genai.configure(api_key=os.getenv("GOOGLE_KEY"))

def get_pdf_text(pdf_docs):
    """
    This function extracts text from the uploaded PDF files.

    Args:
    pdf_docs (list): A list of uploaded PDF files.

    Returns:
    str: The combined text from all the PDF files.
    """
    text = ""
    for pdf in pdf_docs:
        # Read the PDF file using PyPDF2
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            # Extract text from each page and append to the text string
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    This function splits the extracted text into chunks.

    Args:
    text (str): The combined text from all the PDF files.

    Returns:
    list: A list of text chunks.
    """
    # Initialize the text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    # Split the text into chunks
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    This function creates a vector store from the text chunks using Google Generative AI embeddings.

    Args:
    text_chunks (list): A list of text chunks.

    Returns:
    None
    """
    # Initialize the embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Create a vector store from the text chunks using FAISS
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # Save the vector store locally
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """
    This function sets up a conversational chain using Google Generative AI model.

    Returns:
    chain: The configured conversational chain.
    """
    # Define the prompt template for the conversational chain
    prompt_template = """
**Act as an Expert:**
You are an expert in answering questions based on the provided context. Your goal is to provide detailed and accurate answers.

**Task:**
Your task is to answer the question as thoroughly as possible using only the information provided in the context. If the answer is not available in the context, please state "answer is not available in the context" and do not provide any incorrect information.

**Context:**
The context for your answer is as follows:
{context}

**Question:**
The question you need to answer is:
{question}

**Answer:**
Please provide your answer in a clear and concise manner, ensuring all relevant details from the context are included.

**Additional Guidelines:**
- Be specific and detailed in your answer.
- Avoid providing any information not present in the context.
- Use a professional and clear tone.
- If any part of the question is unclear, request clarification before providing an answer.

Answer:
"""
    # Initialize the Google Generative AI model
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.3)
    # Create a prompt template object
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Load the question-answering chain with the model and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    """
    This function processes the user's question and retrieves the answer from the PDF content.

    Args:
    user_question (str): The user's question.

    Returns:
    None
    """
    # Initialize the embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Load the vector store from the local file
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # Perform a similarity search to find relevant documents
    docs = new_db.similarity_search(user_question)
    # Get the conversational chain
    chain = get_conversational_chain()
    # Generate the response using the chain
    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)
    # Print the response (for debugging purposes)
    print(response)
    # Display the response in the Streamlit app
    st.write("Reply: ", response["output_text"])

def main():
    """
    This is the main function that runs the Streamlit app.
    It handles the user interface, file uploads, and question answering.
    """
    # Set the page configuration for the Streamlit app
    st.set_page_config("Chat PDF")
    # Display the header of the app
    st.header("Talk with PDF📄")

    # Create a text input widget for the user to ask questions
    user_question = st.text_input("Ask a Question from the PDF Files")

    # If a question is entered, process the user input
    if user_question:
        user_input(user_question)

    # Create a sidebar menu for file uploads and processing
    with st.sidebar:
        st.title("Menu:")
        # Create a file uploader widget for PDF files
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        # If the submit button is clicked, process the PDF files
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Extract text from the uploaded PDF files
                raw_text = get_pdf_text(pdf_docs)
                # Split the extracted text into chunks
                text_chunks = get_text_chunks(raw_text)
                # Create a vector store from the text chunks
                get_vector_store(text_chunks)
                # Display a success message
                st.success("Done")

# This block ensures that the main function runs when the script is executed
if __name__ == "__main__":
    main()