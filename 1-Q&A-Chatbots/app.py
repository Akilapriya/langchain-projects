import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with openAI"

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant.Pls respond to the user questions"),
        ("user", "Question: {question}"),
    ]
)


def generate_response(question, api_key, llm, temperature, max_tokens):
    openai_api_key = api_key
    llm = ChatOpenAI(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer


# Title of the app
st.title("Enhanced Q&A Chatbot with OPENAI")

# sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OPENAI API KEY:", type="password")

# drop down to select various OPENAI models
llm = st.sidebar.selectbox("Select an OPENAI model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])

# adjust response paramaeter
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("max_tokens", min_value=50, max_value=300, value=150)

# main interface for user input
st.write("Go ahead and ask your question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide your query")
