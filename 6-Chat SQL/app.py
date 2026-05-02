import streamlit as st
from pathlib import Path
from langchain_classic.agents import create_sql_agent
from langchain_classic.sql_database import SQLDatabase
from langchain_classic.agents.agent_types import AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_classic.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

st.set_page_config(page_title="Langchain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 Langchain: Chat with SQL DB")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["Use SQLite3 Database-student.db", "Connect to MYSQL Database"]

selected_opt = st.sidebar.radio(
    label="Choose the DB which you want to chat", options=radio_opt
)

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provide MY SQL Host")
    mysql_user = st.sidebar.text_input("MY SQL User")
    mysql_password = st.sidebar.text_input("MY SQL Password", type="password")
    mysql_db = st.sidebar.text_input("MY SQL Database")
else:
    db_uri = LOCALDB

api_key = st.sidebar.text_input(label="Groq API key", type="password")

if not db_uri:
    st.info("Please enter database infr and uri")
if not api_key:
    st.info("Please add the Groq api key")
    st.stop()

# llm model
llm = ChatGroq(
    groq_api_key=api_key, model_name="llama-3.3-70b-versatile", streaming=True
)


@st.cache_resource(ttl="2h")
def configure_db(
    db_uri, mysql=None, host=None, mysql_user=None, mysql_password=None, mysql_db=None
):
    if db_uri == LOCALDB:
        dbfilepath = (Path(__file__).parent / "student.db").absolute()
        print(dbfilepath)
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == mysql:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("please provide all musql connection detail")
            st.stop()
        return SQLDatabase(
            create_engine(
                f"mysql+mysql/connector://{mysql_user}:{mysql_password}@{mysql_host}/ {mysql_db}"
            )
        )


if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri)


# toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
)

if "messages" not in st.session_state or st.sidebar.button("clear message history"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
user_query = st.chat_input(placeholder="Ask anything from the database")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        streamlit_cb = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[streamlit_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
