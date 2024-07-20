import glob2
from PyPDF2 import PdfReader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader,TextLoader
import os,re
import pdfplumber
import customSplitter as cs
from langchain.vectorstores import Chroma
    

def preprocess_pdf():
    pdf_files = glob2.glob("D:\\MLProject\\ZeePT\\data\\pdfs\\*.pdf")
    
    for pdfs in pdf_files:
        filename = os.path.basename(pdfs).split(".")[0]
        
        with pdfplumber.open(pdfs) as pdf:  # open document
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
                text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                text = re.sub(r'[^A-Za-z0-9\s.,;:!?()-]', '', text)  # Remove special characters  
                with open("D:\\MLProject\\ZeePT\\data\\audit\\" + filename + ".txt", 'w', encoding='utf-8') as f:
                    f.write(text + "\n\n")
               
            
        
def get_text_chunks():
    doc_files = glob2.glob("D:\\MLProject\\ZeePT\\data\\audit\\*.txt")
    text_chunks = []
    for doc in doc_files:  
        segmented_text = {}   
        
        with open(doc, 'r', encoding='utf-8') as file:          
          text = file.read()
       # Regex pattern for matching sections
          sections = re.split(r'(Opinion|Basis for Opinion|Key Audit Matters|How the matter was addressed in our audit|Other Information|Information Other|Responsibilities of Management|Auditor’s Responsibilities)', text)
                                          
          for i in range(1, len(sections), 2):
             section_name = sections[i].strip().lower()
             section_content = sections[i + 1].strip()                    
                    # Append the new content to the existing content if the section already exists
             if section_name in segmented_text:
                segmented_text[section_name] += "\n" + section_content
             else:
                segmented_text[section_name] = section_content
               
      
      # Create a dictionary from matched sections
        splitter = cs.customTextSplitter()
        
        for section, content in segmented_text.items():
            chunks_with_metadata = splitter.split_and_package(content, doc ,section)
            text_chunks.append(chunks_with_metadata)
            
       # final_chunks.append(text_chunks)                
    for i, chunk in enumerate(text_chunks):
        print(f"Chunk {i+1}:\n{chunk}\n")    
    return text_chunks
   


def llm_embedretrieve(chunks):
    # api_url = "https://your-llm-server/api/v1/generate"
    HF_key = ""
    model_id = "google/flan-t5-base" # "stabilityai/stablelm-tuned-alpha-3b"    
   
    embeddings = HuggingFaceInstructEmbeddings(
            
            model_name="hkunlp/instructor-base",  #https://huggingface.co/hkunlp
            
            
        )
    persist_directory = 'D://MLProject//ZeePT//data//audit//chromaDB'
    # Assuming you have set up a vector store `vector_db` with Chroma
    for chunk in chunks:
         vectordb = Chroma.from_documents(documents=chunk, 
                                embedding=embeddings,
                                  persist_directory=persist_directory)
         
 
        #This also works - vectordb = FAISS.from_documents(documents=chunks, embedding=embeddings) 
    vectordb.persist()
    
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
    llm = HuggingFaceHub(huggingfacehub_api_token=HF_key,repo_id=model_id, model_kwargs={"temperature":1, "max_length":500})
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(llm=llm,prompt = prompt)
    
    new_db = Chroma(persist_directory,embedding=embeddings,)
    question = "Explain the feature selection"
    docs = new_db.similarity_search(question) 
# Now you can use `qa_chain` to run queries
    
    #result = qa_chain(query)
    #print(result)
    
    
    response = chain({"input_documents":docs,"question": question}, return_only_outputs=True )   
     
    print(response["output_text"])
   
   
   
if __name__ == "__main__":
   #streamlit run D:\MLProject\ZeePT\scripts\app.py
    preprocess_pdf()
    text_chunks = get_text_chunks()
    chain = llm_embedretrieve(text_chunks)  
   