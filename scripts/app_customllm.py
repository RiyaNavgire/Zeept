import streamlit as st
import data_ingestion as di
import data_chunks as dc
import doc_retrieval_customLLM as dr
import doc_retrieval_multi as drm
import run_chain as rc
import create_embedding as create_embed
import data_ingestion_multi as di_multi
import data_chunks_multi as dc_multi
import create_embedding_multi as create_embed_multi
import run_chain_multi as run_multi



def user_input(user_question,chain):
   

    answer = rc.trigger_chain(user_question, chain)
    st.header("Answer:")
    st.write(answer["answer"])
    
    #st.write("Reply: ", answer["output_text"])
    print("Answers printed!!!")

    
    
if __name__ == "__main__":
   #streamlit run D:\MLProject\ZeePT\scripts\app.py
    pdf_list = di.get_pdf_text()
    doc_texts = dc.get_text_chunks(pdf_list)
    chain = dr.custom_llm_embedretrieve(doc_texts)  
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
            print("starting streamlit")
            #SINGLE
            pdf_list = di.get_pdf_text()
            doc_texts = dc.get_text_chunks(pdf_list)
            chain = dr.custom_llm_embedretrieve(doc_texts)
            #rc.trigger_chain(user_question, chain)
        