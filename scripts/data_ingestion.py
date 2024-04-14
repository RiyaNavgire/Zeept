import glob2
from PyPDF2 import PdfReader 

def get_pdf_text():
    pdf_texts = ""
    text = ""
    pdf_files = glob2.glob("D:\MLProject\ZeePT\data\pdfs\\*.pdf")
    pdf_readers = []
    for pdf_file in pdf_files:
      pdf_readers.append(PdfReader(pdf_file))

    for pdf_reader in pdf_readers:
       for page in pdf_reader.pages:
           text += page.extract_text()
           #print(text)
       #pdf_texts.append(text)
       
    print("\nTotal PDFs processed:", {len(pdf_texts)})
    #Perform further processing or analysis on each text snippet
    #print(f"**************Sample text from a PDF******************:\n{pdf_texts[2]}:--")  
    
    return text
