from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
import os
import pickle
from langchain.llms import LlamaCpp
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers #it is used for binding locally downloaded LLM models
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
import langchain


#from langchain_community.llms import Ollama

#But to be clear, the answers it gives you are super short . But it WORKS. HF for not making the Pro requirement clearer.

def doc_retrieve(vector_db):
    # Initialise LLAMA2 LLM with required params
    # llm = Llama(
    # streaming = True,
    # model_path="/content/drive/MyDrive/Model/mistral-7b-instruct-v0.1.Q1_K_M.gguf",
    # temperature=0.75,
    # top_p=1,
    # verbose=True,
    # n_ctx=4096  )
      
       
    #**********OLLAMA********
    # llm=Ollama(model="gemma")
    # output_parser=StrOutputParser()
    # chain=prompt|llm|output_parser
    
    
    #********* Hugging Face Models using API Key ********   
    HF_key = ""
    model_id = "google/flan-t5-base" # "stabilityai/stablelm-tuned-alpha-3b"    
    llm = HuggingFaceHub(huggingfacehub_api_token=HF_key,repo_id=model_id, model_kwargs={"temperature":1, "max_length":500})
    prompt_template="""
        Use the following piece of context to answer the question asked.
        Please try to provide the answer only based on the context

        {context}
        Question:{question}
        Helpful Answers:
        """
    
    #********* Hugging Face Models LLAMA download locally and use CTransformers ********   
    # llm = llm=CTransformers(model='D://MLProject//ZeePT//model//llama-2-7b-chat.ggmlv3.q8_0.bin',
    #                    model_type='llama', config={"temperature":0.5, "max_new_tokens":500})
    
    
    
    # file_path =  "D://MLProject//ZeePT//data//hugging_docs//vector_index.pkl"
    # if os.path.exists(file_path):
    #     with open(file_path, "rb") as f:
    #         vectorIndex = pickle.load(f)
    
    
    
    #memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    #chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vectorIndex.as_retriever(search_kwargs={"k": 2}),memory = memory)  #ensure vectorIndex object type is FAISS
    # prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    # document_chain=create_stuff_documents_chain(llm,prompt)
    # chain=create_retrieval_chain(vector_db.as_retriever(search_kwargs={"k": 3}),document_chain)
    # #chain = load_qa_chain(llm=llm,prompt = prompt)
    
    # working code - 
    #chain = RetrievalQAWithSourcesChain.from_chain_type(llm="http://127.0.0.1:11434/",chain_type="map_reduce", retriever=vector_db.as_retriever(search_kwargs={"k": 3}),return_source_documents=True)  #ensure vectorIndex object type is FAISS
    chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,chain_type="map_reduce", retriever=vector_db.as_retriever(search_kwargs={"k": 3}),return_source_documents=True)  #ensure vectorIndex object type is FAISS
    print(chain)
    
    
    
    return chain



#ConversationalRetrievalChain - requires chat_history as key
#RetrievalQAWithSourcesChain - requires arguments sources chain
#RetrievalQA.from_chain_type