from langchain.chains import RetrievalQAWithSourcesChain
from LLMHosted import CustomLLM
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from transformers import  AutoTokenizer
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate

# Example setup, adjust as necessary
def custom_llm_embedretrieve(documents_texts):
   # api_url = "https://your-llm-server/api/v1/generate"

    embeddings = HuggingFaceInstructEmbeddings(
            #api_key=HF_key, 
            model_name="hkunlp/instructor-base",  #https://huggingface.co/hkunlp
            #HuggingFaceInferenceAPIEmbeddings #model_name="sentence-transformers/all-MiniLM-L6-v2",
            
        )

    #custom_llm = CustomLLM(api_url=api_url)
    
    HF_key = "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"
    model_id = "google/flan-t5-base" # "stabilityai/stablelm-tuned-alpha-3b"    
    llm = HuggingFaceHub(huggingfacehub_api_token=HF_key,repo_id=model_id, model_kwargs={"temperature":1, "max_length":500})
    
    
    

# Assuming you have set up a vector store `vector_db` with Chroma
    vector_db = FAISS.from_texts(texts = documents_texts, embedding = embeddings)
    vector_db.save_local("D://MLProject//ZeePT//data//faiss_index_hugging") 
    #retriever_vec = vector_db.as_retriever(search_k=5)  # Adjust search_k as necessary

    # qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    #     llm=llm,
    #     chain_type="map_reduce",
    #     retriever=retriever_vec,
    #     return_source_documents=True
    # )
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(llm=llm,prompt = prompt)
    
    new_db = FAISS.load_local("D://MLProject//ZeePT//data//faiss_index_hugging", embeddings)
    question = "Explain the feature selection"
    docs = new_db.similarity_search(question) 
# Now you can use `qa_chain` to run queries
    
    #result = qa_chain(query)
    #print(result)
    
    
    response = chain({"input_documents":docs,"question": question}, return_only_outputs=True )   
     
    print(response["output_text"])
   