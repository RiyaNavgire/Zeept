from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain import HuggingFaceHub

def get_text_chunks(loaders):
    
    
   
    
    #for pdf in pdfs:             
    text_splitter = RecursiveCharacterTextSplitter(
      separators = ["\n\n", "\n", " "],  # List of separators based on requirement (defaults to ["\n\n", "\n", " "])
      chunk_size = 2000,  # size of each chunk created
      chunk_overlap  = 200,  # size of  overlap between chunks in order to maintain the context
      length_function = len  # Function to calculate size, currently we are using "len" which denotes length of string however you can pass any token counter)
    )
    pdf_chunks = []      
    for loader in loaders:      
      chunk = text_splitter.split_documents(loader.load())
      pdf_chunks.append(chunk)
    
    
    #print("\nPDF Chunk length - ",len(pdf_chunks))
    #print(pdf_chunks[3])
    
    return pdf_chunks