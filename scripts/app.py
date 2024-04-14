import streamlit as st
import data_ingestion as di
import data_chunks as dc
import doc_retrieval as dr
import run_chain as rc
import create_embedding as create_embed



def user_input(user_question,chain):
    # embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")    
    # new_db = FAISS.load_local("faiss_index", embeddings)
    # docs = new_db.similarity_search(user_question)

    answer = rc.trigger_chain(user_question, chain)
    st.header("Answer:")
    st.write(answer["answer"])
    
    #st.write("Reply: ", answer["output_text"])
    print("Answers printed!!!")

    
    
if __name__ == "__main__":
   #
    # pdf_list = di.get_pdf_text()
    # doc_texts = dc.get_text_chunks(pdf_list)
    # create_embed.select_embeddings_model("HuggingFace",doc_texts)
    # chain = dr.doc_retrieve()
    # rc.trigger_chain("What is monthly premium", chain)
    
    #*********STREAMLIT*************
    st.set_page_config(page_title="Chat PDF",page_icon=":books:",  # Optional: Set a page icon (emoji or URL)
    layout="wide",  # Optional: Set the layout ("centered" or "wide")
    initial_sidebar_state="collapsed"  # Optional: Set the initial sidebar state)
    )
    st.header("Zeept BOT using Llama - Ask me Anything , any Documents💁")
    user_question = st.text_input("Please ask your Question?")

    

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # pdf_list = di.get_pdf_text()
                 # doc_texts = dc.get_text_chunks(pdf_list)
                # create_embed.select_embeddings_model("HuggingFace",doc_texts)
                st.success("Done")
        if user_question:
            pdf_list = di.get_pdf_text()
            doc_texts = dc.get_text_chunks(pdf_list)
            create_embed.select_embeddings_model("HuggingFace",doc_texts)
            chain = dr.doc_retrieve()
            rc.trigger_chain(user_question, chain)
            
                
    
        