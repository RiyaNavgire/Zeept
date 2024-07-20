from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings , HuggingFaceInferenceAPIEmbeddings,HuggingFaceInstructEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
#from gensim.models import Doc2Vec
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
#from langchain.vectorstores import Chroma
import chromadb
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

google_api_key = 'your_google_api_key'
openai_api_key = "openai_api_key"
HF_key = "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"	



#texts = ["How do I get a replacement Medicare card?","How do I terminate my Medicare Part B (medical insurance)?"]

    

def select_embeddings_model(LLM_service,documents_texts):
    """Connect to the embeddings API endpoint by specifying 
    the name of the embedding model."""
    
    
    #embeddings = []
    
    # if LLM_service == "Document":
    #     # Create and train Doc2Vec model (adjust parameters as needed)
    #     Doc2Vec(vector_size=100, window=5, min_count=2)
    #     model.build_vocab(pdf_chunks)
    #     model.train(pdf_chunks, total_examples=model.corpus_count, epochs=10)
    #     for chunks in pdf_chunks:
    #         for chunk in chunks:
    #             embedding = model.encode(chunk)
    #     embeddings.append(embedding)
    
    
    if LLM_service == "OpenAI":
        embeddings = OpenAIEmbeddings(
            model='text-embedding-ada-002',
            api_key=openai_api_key)

    if LLM_service == "Google":
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
    if LLM_service == "HuggingFace":        
        embeddings = HuggingFaceInstructEmbeddings(
            #api_key=HF_key, 
            model_name="hkunlp/instructor-base",  #https://huggingface.co/hkunlp
            #HuggingFaceInferenceAPIEmbeddings #model_name="sentence-transformers/all-MiniLM-L6-v2",
            
        )
        
           
        #vector_store = FAISS.from_documents(documents = texts, embedding = embeddings)
        #metadatas = [{"page": i} for i in range(len(documents))]
        #for multiple documents create list of pdf text and run below FAISS in loop
        vector_store = FAISS.from_texts(texts = documents_texts, embedding = embeddings)
        
        # Storing vector index create in local
        # file_path="D://MLProject//ZeePT//data//hugging_docs//vector_index.pkl"
        # with open(file_path, "wb") as f:
        #     pickle.dump(vector_store, f)
        vector_store.save_local("D://MLProject//ZeePT//data//faiss_index_hugging") 
             
        print("Vector Stored successfully")
        
    
    #*************FAISS***********************
    
    # for chunks in pdf_chunks:  
    #     for chunk in chunks:
    #         embeddings = [model.encode(chunk) ]#  for chunk in chunks] 
    #         # vector_dimension = embeddings[0].shape[1]
    #          #index = FAISS.IndexFlatL2(vector_dimension)
    #          #FAISS.normalize_L2(embeddings)
    #          #index.add(embeddings)
    #         vector_store = FAISS.from_texts(chunk,embedding = np.array(embeddings_list))
    #         vector_store.save_local("faiss_index")
    #text_embedding_pairs = zip(pdf_chunks, embeddings_list)
           
        # vector_store = FAISS.from_texts(texts = texts, embedding = model) #throws error 'SentenceTransformer' object has no attribute 'embed_documents'
        # vector_store.save_local("D://testing_space//faiss_index_sentence")       
    
    #**************CHROMADB****************
    # client = chromadb.Client()
    # if client.list_collections():
    #     consent_collection = client.create_collection("consent_collection")
    # else:
    #     print("Collection already exists")
                    
    # for sentence, embedding in zip(pdf_chunks, embeddings_list):
    #   document = {"text": sentence, "embedding": embedding}
    #   client.insert(consent_collection, document)
                    
    return vector_store
    
  