import glob2
from langchain_community.document_loaders import PyPDFLoader

from langchain.document_loaders import UnstructuredPDFLoader
import os

def get_pdf_text():
    pdf_files = glob2.glob("D:\MLProject\ZeePT\data\multipdfs\\*.pdf")
    pdf_readers = []
    
    for pdf_file in pdf_files:
       loader = PyPDFLoader(pdf_file)      
       pdf_readers.append(loader)
    
    
    print("\nTotal PDFs processed:", {len(pdf_readers)})
    
    
    
    return pdf_readers
