import streamlit as st

def main():
    print('Hello...')
    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")
    st.header("Chat with multiple PDFs :books : ")
    st.text_input("Ask question about document : ")
    with st.sidebar:
        st.subheader("Your Documents")
        st.file_uploader("Upload your PDFs here and click on 'Process'")

if __name__=='__main__':
    main()