from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_text_chunks(pdfs):
    
    #pdf_chunks = ""
    
    #for pdf in pdfs:             
    text_splitter = RecursiveCharacterTextSplitter(
      separators = ["\n\n", "\n", " "],  # List of separators based on requirement (defaults to ["\n\n", "\n", " "])
      chunk_size = 2000,  # size of each chunk created
      chunk_overlap  = 200,  # size of  overlap between chunks in order to maintain the context
      length_function = len  # Function to calculate size, currently we are using "len" which denotes length of string however you can pass any token counter)
    )
    chunks = text_splitter.split_text(pdfs)
        #pdf_chunks.append(chunks)
    
    
    #print("\nPDF Chunk length - ",len(pdf_chunks))
    #print(pdf_chunks[3])
    
    return chunks