from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings , HuggingFaceInferenceAPIEmbeddings,HuggingFaceInstructEmbeddings,HuggingFaceHubEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
#from gensim.models import Doc2Vec
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
#import chromadb
import faiss
import numpy as np
import pandas as pd
import glob2

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import ConversationalRetrievalChain
import pickle

from langchain import HuggingFaceHub



#load_dotenv()
#os.getenv("GOOGLE_API_KEY")
#genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

HF_key = ""	
openAI_key = "nn"


#texts = ["How do I get a replacement Medicare card?","How do I terminate my Medicare Part B (medical insurance)?"]

    

def select_embeddings_model(pdf_chunks):
    """Connect to the embeddings API endpoint by specifying 
    the name of the embedding model."""
    persist_directory = 'D://MLProject//ZeePT//chromDB'
    
    
    embeddings = HuggingFaceInstructEmbeddings(
            #api_key=HF_key, 
            model_name="hkunlp/instructor-base",  #https://huggingface.co/hkunlp
            #HuggingFaceInferenceAPIEmbeddings #model_name="sentence-transformers/all-MiniLM-L6-v2",
            
        )
    
    for chunks in pdf_chunks:
         vectordb = Chroma.from_documents(documents=chunks, 
                                embedding=embeddings,
                                  persist_directory=persist_directory)
         
 
        #This also works - vectordb = FAISS.from_documents(documents=chunks, embedding=embeddings) 
    vectordb.persist()
    
           
   # vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
                
    print("Vector Stored successfully")
        
    return vectordb
    
    
  