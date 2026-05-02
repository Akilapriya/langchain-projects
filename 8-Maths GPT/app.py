import sys
import streamlit as st

st.write(f"Python Executable: {sys.executable}")
st.write(f"Looking for packages in: {sys.path}")
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents import Tool, initialize_agent
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_classic.chains import LLMChain, LLMMathChain

# strealit app
st.set_page_config(
    page_title="Text To Math Problem Solver and Data Search Assistant", page_icon="📚"
)
st.title("Text To Math Problem Solver Using Google Gemma 2")


groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key")
    st.stop()

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

# Initializing Tools
wikipedia_wrapper = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet to find various info on the topics mentioned",
)
# initialize math tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math related questions. Only input mathematical expression needs to be provided",
)

prompt = """
Your a agent tasked for solving users math question. Provide detailed explanation and display it point wise for the question below
Question:{question}
Answer:
"""
prompt_template = PromptTemplate(input_variables=["question"], template=prompt)

# combine all tools into chain
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic based and reasoning questions",
)
# initialize the agent
assistant_agent = initialize_agent(
    tools=[wiki_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handling_parsing_erros=True,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I'm a Math Chatbot who can answer all your maths questions",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# start the interaction
question = st.text_area(
    "Enter your question",
    "I have 5 apple and 7 bananas. i eat 2bananas and give away 3 apples, I buy a dozen apples and 2 pack of cherry.each cherry pack contains 25 cherries. How many total pieces of fruits do I have at the end",
)
if st.button("find my answer"):
    if question:
        with st.spinner("Generate response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write("Response:")
            st.success(response)
    else:
        st.warning("Please enter the question")
