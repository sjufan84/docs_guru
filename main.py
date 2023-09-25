import streamlit as st
import PyPDF2
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import openai
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

if "db" not in st.session_state:
    st.session_state.db = None
if "similar_text" not in st.session_state:  
    st.session_state.similar_text = None
if "answer" not in st.session_state:
    st.session_state.answer = None

embeddings = OpenAIEmbeddings()

def convert_pdf_to_text(file):
    pdf_reader = PyPDF2.PdfReader(file)
    return " ".join([pdf_reader.pages[i].extract_text() for i in range(0, len(pdf_reader.pages))])

def convert_csv_to_text(file):
    df = pd.read_csv(file)
    return " ".join(df.apply(lambda row: ' '.join(row.astype(str)), axis=1))

def convert_xlsx_to_text(file):
    df = pd.read_excel(file)
    return " ".join(df.apply(lambda row: ' '.join(row.astype(str)), axis=1))

@st.cache_resource
def create_vector_store(uploaded_files):
    # Step 2: File to Text Conversion
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.name.split('.')[-1]
        
        if file_type == 'pdf':
            texts.append(convert_pdf_to_text(uploaded_file))
        elif file_type == 'csv':
            texts.append(convert_csv_to_text(uploaded_file))
        elif file_type in ['xlsx', 'xls']:
            texts.append(convert_xlsx_to_text(uploaded_file))
        elif file_type == 'txt':
            texts.append(uploaded_file.read().decode("utf-8"))

    # Step 3: Text Chunking
    # Create overlapping chunks of text 1000 characters long
    # with a 25 character overlap
    chunk_size = 1000
    overlap = 25
    chunks = []
    for text in texts:
        for i in range(0, len(text), chunk_size-overlap):
            chunks.append(text[i:i+chunk_size])

    # Step 4: Text Embedding
    # Create a vector store of the text chunks
    db = Chroma.from_texts(chunks, embeddings)

    return db

# Create a container to hold the file uploader and the start button
container = st.container()
# Step 1: Multi-file Upload
with container:
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=['txt', 'pdf', 'csv', 'xlsx', 'xls'])

    texts = []

    # Create a button to start the process
    start_button = st.button("Start upload")
    if start_button and len(uploaded_files) == 0:
        st.error("Please upload at least one file.")
    elif start_button and len(uploaded_files) > 0:
        with st.spinner("Uploading files..."):
            st.session_state.db = create_vector_store(uploaded_files)
            # If the vector store is successfully created, display a success message
            container.success("Vector store successfully created!")

def get_similar_text(user_query):
    """ Search the vector store for similar text """
    # Search the vector store for similar text
    vectorstore = st.session_state.db
    results = vectorstore.similarity_search(user_query, k=4)

    # Create a list from the results
    similar_text = []
    for result in results:
        similar_text.append(result.page_content)

    return similar_text

def get_llm_response(user_question:str):
    """ Get the response from the language model
    with the context from the vector store """
    context = get_similar_text(user_question)

    # Create the prompt
    messages = [
        {"role": "system", "content" : f"""You are a helpful assistant
        that answers a user's questions about documents that they have uploaded.
        The user's question is {user_question}, and the relevant context is: {context}.
        If the context is not relevant to the question, do your best to answer the 
        question as a subject matter expert, while letting the user know that the
        context is not relevant.
        """
        },
    ]
    
    # List of models to use
    models = ["gpt-4", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo"]

        # Loop through the models and try to generate the recipe
    for model in models:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.9,
                max_tokens=500,
                top_p=1,
                frequency_penalty=0.5,
                presence_penalty=0.5,
            )

            answer = response.choices[0].message.content
            return answer
        except:
            continue

st.success("DocsGuru is ready to answer your questions!\
           Upload any docs of types .txt, .pdf, .csv, .xlsx, or .xls, and\
           DocsGuru can speak intelligently about whatever you upload.")

if st.session_state.db:
    question = st.text_input("Enter your question")
    if st.button("Ask"):
        with st.spinner("DocsGuru is thinking..."):
            st.session_state.similar_text = get_similar_text(question)
            st.session_state.answer = get_llm_response(question)
            st.write(st.session_state.answer)   