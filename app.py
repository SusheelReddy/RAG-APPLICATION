import streamlit as st
from utilis.retrieval import load_faiss_index, retrieve_chunks
from utilis.prompt import build_prompt
from utilis.Completion import generate_completion


st.title("RAG APP- Founder story")
st.write("Ask questions grounded ")

query = st.text_input("Enter your question here")

if query:
    index , chunk_mapping = load_faiss_index()
    top_chunks = retrieve_chunks(query,index,chunk_mapping)
    prompt=build_prompt(top_chunks,query)
    response=generate_completion(prompt)

    st.subheader("Answer")
    st.write(response)
     
    with st.expander("Retrieved Chunks"):
        for chunk in top_chunks:
            st.markdown(f"-{chunk}")