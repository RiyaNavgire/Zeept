import langchain
import streamlit as st
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import pickle
import os



def trigger_chain(query,chain):
    # query = "what are the main features of punch iCNG?"
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
    
    
    new_db = FAISS.load_local("D://MLProject//ZeePT//data//faiss_index_hugging//", embeddings)
    
    docs = new_db.similarity_search(query)
    try:
        
        langchain.debug=True
        response = chain({"input_documents":docs,"question": query}, return_only_outputs=True )     
        st.write("Answer : ",response["output_text"])   
        print("Answers printed successfully")
    finally: print ("Code completed successfully!")
    return response

