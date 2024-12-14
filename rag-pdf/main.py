import os
from fastapi import FastAPI, Query
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GooglePalm
import google.generativeai as genai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()


# Initialize FastAPI
app = FastAPI(
    title="CNC chat-bot Server",
    version="1.0",
    decsription="A simple chat-bot Server"
)
api_key = os.getenv("GOOGLE_GEMINI_KEY")
genai.configure(api_key=api_key)
os.environ["GOOGLE_API_KEY"] = api_key

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Load PDF file
loader = PyPDFLoader("CNC_Medical_Clinic_Info.pdf")
docs = loader.load()

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(docs)

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.from_documents(documents[:30], embeddings)

print("FAISS index created successfully.")



@app.get("/chat")
async def chat(query: str = Query(..., description="Ask anything")):
    # Perform similarity search
    query_results = db.similarity_search(query)

    if query_results:
        # Extract context from search results
        context = " ".join([doc.page_content for doc in query_results[:3]])  # Use top 3 results
        
        # Define a prompt template for the chain
        prompt_template = """
        You are a CNC clinic chatbot. Answer the user query clearly and concisely.
        Use the following context to answer the user's query:
        
        Context: {context}
        Question: {query}
        """
        
        # Create a prompt
        prompt = PromptTemplate(
            input_variables=["context", "query"],  # These are placeholders in the template
            template=prompt_template
        )
        
        # Initialize LLMChain
        llm = GooglePalm(model="gemini-pro")  # Use free Gemini model
        chain = LLMChain(llm=llm, prompt=prompt)

        # Run the chain
        response = chain.run({"context": context, "query": query})
        print(f"response:{response}")
        return {"bot": response}
    else:
        return {"bot": "Sorry, I couldn't find any relevant information. Feel free to ask another question."}
