import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import openai

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        st.write(type(pdf),pdf)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_data):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len        
    )
    chunks = text_splitter.split_text(raw_data)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl") # takes forever.
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm =llm,
        retriever=vector_store.as_retriever(),
        memory = memory
    )
    return conversation_chain

def handle_user_question(user_question):
    response = st.session_state.conversation({'question' : user_question})
    st.write(response)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")

    if "conversation" not in st.session_state:
            st.session_state.conversation = None

    st.header("Chat with multiple PDFs :books : ")
    user_question = st.text_input("Ask question about document : ")

    if user_question:
        handle_user_question(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'",accept_multiple_files=True)
        if st.button("process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)               # gets pdf text.
                text_chunks = get_text_chunks(raw_text)         # gets text chunks.
                vector_store = get_vector_store(text_chunks)
                # st.write(vector_store)
                st.session_state.conversation = get_conversation_chain(vector_store)

if __name__=='__main__':
    main()