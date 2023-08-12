import streamlit as st
import os
import openai
import pickle

from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

openai.key = st.secrets["OPENAI_API_KEY"]

def main():

    st.image("img/bupa_logo.png", width=60)
    st.image("img/background.png")
    st.markdown("<h4 style='text-align: center; '>What can we help you with today? </h4>", unsafe_allow_html=True)

    # upload a PDF file
    #pdf = st.file_uploader("Upload your PDF", type='pdf')

    # if pdf is not None:
    #     pdf_reader=PdfReader(pdf)

    #     text=""
    #     for page in pdf_reader.pages:
    #         text += page.extract_text()

    #     # split text since llms have context windows
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=1000,
    #         chunk_overlap=200,
    #         length_function=len
    #         )
        
    #     chunks = text_splitter.split_text(text=text)

    # create embeddings from custom document
    # if already exists, load
    store_name="embeddings/BBY1124JAN23-BBY-Policy-Benefits-and-Terms.pkl"
    #store_name = pdf.name[:-4]
    #st.write('embeddings/'+f'{store_name}'+'.pkl')

    if os.path.exists(f"{store_name}"):
        with open(f"{store_name}", "rb") as f:
            VectorStore = pickle.load(f)
        # st.write('Embeddings Loaded from the Disk')s
    else:
        pass
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        # with open(f"{store_name}", "wb") as f:
        #     pickle.dump(VectorStore, f)

    # Accept user questions/query
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    query = st.text_input(
        "Ask me a question about our policy documentation.",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled
    )

    if query:
        docs = VectorStore.similarity_search(query=query, k=3)

        llm = OpenAI() # model_name='gpt-3.5-turbo'
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            print(cb)
        
        with st.chat_message(name='assistant', avatar='img/bupa_logo.png'):
            st.write(response)

    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown('''For more information, see [Bupa-By-You Policy Benefits and Terms](https://www.bupa.co.uk/~/media/Files/UserDefined/BBY/BBY1124JAN23-BBY-Policy-Benefits-and-Terms.pdf) or [Bupa.co.uk](https://www.bupa.co.uk/)''')
    st.markdown('''Made by Ben Turner as a **proof-of-concept only**.''')

if __name__ == '__main__':
    main()