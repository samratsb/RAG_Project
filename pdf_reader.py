from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import os
from dotenv import load_dotenv

# Load Hugging Face API token and file path from .env file
load_dotenv()

file = os.getenv('file')
token = os.getenv('HF_API_TOKEN')

if not token:
    raise ValueError("Hugging Face API token is missing. Please set it in the .env file.")
if not file:
    raise ValueError("File path is missing. Please set it in the .env file.")
if not os.path.isfile(file):
    raise ValueError(f"File does not exist: {file}")

login(token)

def get_pdf_content(documents):
    """
    Extract text content from a list of PDF documents.
    """
    raw_text = ""
    for document in documents:
        pdf_reader = PdfReader(document)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    return raw_text

def get_chunks(text):
    """
    Split text into manageable chunks for processing.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=100,
        chunk_overlap=20,
        length_function=len
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks

def get_embeddings(chunks):
    """
    Generate embeddings for text chunks using a pre-trained model from Hugging Face,
    and store them using FAISS.
    """
    model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
    hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    vector_storage = FAISS.from_texts(chunks, embedding=hf_embeddings)
    
    return vector_storage, hf_embeddings

def setup_qa_chain(vector_storage, model_name="gpt2"):
    """
    Set up a question-answering chain using a Hugging Face model and FAISS vector store.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,  # Adjust this value as needed
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    retriever = vector_storage.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    return qa_chain

# Example usage
documents = [file]
raw_text = get_pdf_content(documents)
chunks = get_chunks(raw_text)
vector_storage, embeddings = get_embeddings(chunks)
qa_chain = setup_qa_chain(vector_storage)

print("AI: Hello! I'm ready to answer questions about the resume. What would you like to know?")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("AI: Goodbye! Have a great day!")
        break
    
    try:
        result = qa_chain.invoke({"query": user_input})
        answer = result['result']
        print(f"AI: {answer}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("AI: I'm sorry, I couldn't process your request. Could you please try again?")
