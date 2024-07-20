import langchain
import streamlit as st
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import OpenAIEmbeddings

from langchain.vectorstores import FAISS
import pickle
import os



def trigger_chain(question,chain):
    
    
    # print("Triggering")
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
    
    # new_db = FAISS.load_local("D://MLProject//ZeePT//data//multiVectorStores//", embedding)
    
   # docs = vector_db.similarity_search(question)  #Similarity matches question vs embeddings created using embedding model which was used to create emebdding
    try:
        
        langchain.debug=True
        llm_response = chain.invoke({"input":question})   
        print(llm_response['answer'])
        print('\n\nSources:')
        for source in llm_response["source_documents"]:
            print(source.metadata['source'])
            
        st.write("Answer : ",llm_response["source_documents"])   
        print("Answers printed successfully")
    finally: print ("Code completed successfully!")
    return llm_response