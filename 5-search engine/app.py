import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

#arxiv and wikipedia tools
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name="Search")

#title 
st.title("Langchain - Chat with Search")

"""
In this exampple we are using 'StreamlitCallbackHandler' to display the thoughts and actions of an agent in an interactive streamlit app
"""

#sidebar settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[{"role": "assistant", "content": "Hi I'm a chatbot who can search the web. How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:=st.chat_input(placeholder ="whats machine learning"):
    st.session_state.messages.append({"role":"user", "content": prompt})

    llm= ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")
    tools=[search, arxiv, wiki]
    search_agent= initialize_agent(tools, llm, AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response= search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
    
