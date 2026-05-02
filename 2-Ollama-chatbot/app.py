import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

load_dotenv()

# langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Ollama"

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant.Pls respond to the user questions"),
        ("user", "Question: {question}"),
    ]
)


def generate_response(question, engine, temperature, max_tokens):

    llm = Ollama(model=engine)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer


# Title of the app
st.title("Enhanced Q&A Chatbot with Ollama")


# drop down to select various Ollama models
engine = st.sidebar.selectbox("Select an Ollama model", ["llama2", "phi3"])

# adjust response paramaeter
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("max_tokens", min_value=50, max_value=300, value=150)

# main interface for user input
st.write("Go ahead and ask your question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, engine, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide your query")
