from langchain.llms import HuggingFaceHub
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import video_to_audio, audio_to_text, video_to_text
import os
import nest_asyncio
import streamlit as st

# Convert video to transcript
video_path = "assets/Máy học là gì.mp4"
audio_path = "audio.wav"
text = video_to_text(video_path, audio_path, model_name="base")
print(text)
docs = text


# Login Huggingface and create token "write"
os.environ["HUGGINGFACEHUB_API_TOKEN"] 


# Data/document 
nest_asyncio.apply()

# Text Splitting - Chunking
docs = [Document(page_content=text)]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0)
chunks = text_splitter.split_documents(docs)
print("Number of sub - documents : ", len(chunks))

# Check the contents of the chucks
print(chunks[0])
print(chunks[1])
print(chunks[2])


# Embeddings
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=["HUGGINGFACEHUB_API_TOKEN"], model_name="BAAI/bge-base-en-v1.5"
)

# Vector Store - ChromaDB
vectorstore = Chroma.from_documents(chunks, embeddings)
query = "Machine Learning ra đời khi nào?"
search = vectorstore.similarity_search(query)

search[0].page_content

## Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr", #similarity
    search_kwargs={'k': 4}
)

retriever.get_relevant_documents(query)

# Large Language Model - Open Source
llm = HuggingFaceHub(
    repo_id="huggingfaceh4/zephyr-7b-alpha",
    model_kwargs={"temperature": 0.5, "max_length": 64,"max_new_tokens":512}
)

# Prompt Template and User Input
query = "Tại sao Machine Learning quan trọng?"

prompt = f"""
 <|System|>
Tôi là trợ lý AI mời bạn đặt câu hỏi về Machine Learning
</s>
 <|User|>
 {query}
 </s>
 <|Assistant|>
"""

# RAG RetrievalQA chain
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever)

response = qa.run(prompt)

print(response)

# Chain
template = """
 <|System|>
Tôi là trợ lý AI mời bạn đặt câu hỏi về Machine Learning
</s>
 <|User|>
 {query}
 </s>
 <|Assistant|>
"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever,  "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Predict
response = rag_chain.invoke("Máy học là ")
print(response)

# Launch Streamlit App
st.title("Chatbot hỏi đáp từ video")

# User enters question
user_input = st.text_area("Hãy đặt câu hỏi của bạn:")

# If the user has entered a query, execute the query
if user_input:
    response = rag_chain.invoke(user_input)
    st.write(response)
else:
    st.write("Vui lòng nhập câu hỏi.")
