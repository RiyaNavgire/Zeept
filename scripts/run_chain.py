import langchain
import streamlit as st
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import pickle
import os



def trigger_chain(query,chain):
    
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
    
    
    new_db = FAISS.load_local("D://MLProject//ZeePT//data//faiss_index_hugging//", embeddings)
    
    docs = new_db.similarity_search(query)  #Similarity matches question vs embeddings created using embedding model which was used to create emebdding
    try:
        
        langchain.debug=True
        response = chain({"input_documents":docs,"question": query}, return_only_outputs=True )     
        st.write("Answer : ",response["output_text"])   
        print("Answers printed successfully")
    finally: print ("Code completed successfully!")
    return response




