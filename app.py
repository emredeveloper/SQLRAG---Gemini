import streamlit as st
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine, inspect
import pandas as pd
import sqlite3
import os
import re

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

# --- 1. Enhanced Injection Warning ---
INJECTION_WARNING = """
    **Security Notice:**
    SQL agents can be vulnerable to prompt injection attacks. For maximum safety, use a read-only database user and ensure you understand the risks before connecting to a production database.

    Learn more: https://python.langchain.com/docs/security
    """

LOCALDB = "USE_LOCALDB"

# --- 2. User Inputs with Database Preview ---
radio_opt = ["Use sample database - Chinook.db", "Connect to your SQL database"]
selected_opt = st.sidebar.radio(label="Choose suitable option", options=radio_opt)

if radio_opt.index(selected_opt) == 1:
    st.sidebar.warning(INJECTION_WARNING, icon="⚠️")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Database Connection:")
    db_uri = st.sidebar.text_input(
        label="Enter Database URI",
        help="Enter the URI of your SQL database (e.g., mysql://user:pass@hostname:port/db)",
    )
else:
    db_uri = LOCALDB

google_api_key = st.sidebar.text_input(
    label="Google API Key",
    type="password",
)

llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, streaming=True)

# Check user inputs and provide database preview
if not db_uri and selected_opt == "Connect to your SQL database":
    st.info("Please enter database URI to connect to your database.")
    st.stop()

if not google_api_key:
    st.info("Please add your Google API key to continue.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key 

# --- 3. Improved Database Connection and Preview ---
@st.cache_resource(ttl="2h")
def configure_db(db_uri):
    try:
        if db_uri == LOCALDB:
            db_filepath = (Path(__file__).parent / "Chinook.db").absolute()
            creator = lambda: sqlite3.connect(f"file:{db_filepath}?mode=ro", uri=True)
            engine = create_engine("sqlite:///", creator=creator)
        else:
            # Validate the database URI format
            if not re.match(r'^[a-zA-Z]+://', db_uri):
                st.error("Invalid database URI format. Please ensure it starts with a valid scheme (e.g., mysql://, postgresql://).")
                st.stop()
            engine = create_engine(db_uri)
            
        with engine.connect() as connection:
            inspector = inspect(engine)
            table_names = inspector.get_table_names()
            
            st.sidebar.subheader("Database Tables:")
            if table_names:
                for table in table_names:
                    st.sidebar.write(f"- {table}")
            else:
                st.sidebar.write("No tables found in the database.")
    
        return SQLDatabase(engine)  
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        st.stop()

db = configure_db(db_uri)

# --- 4. Enhanced Chatbot Instructions ---
DEFAULT_SYSTEM_MESSAGE = """I am here to assist you with the database.
Feel free to ask me anything related to the database.
If your query involves accessing data, I'll use SQL queries to retrieve the information.
If I cannot answer your question or if it's beyond the scope of the database, I'll let you know.
You can also request to view the database structure or ask for examples of queries.

For instance, you can ask me to create a new SQL table, modify existing tables, or retrieve specific information from the database.
Here are some example queries you can try:
- "Show me all customers."
- "What is the total sales amount?"
- "List all albums in the database."
"""

# --- 5. Create SQL Agent ---
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# --- 6. Chat Interface with Improved Formatting and Feedback ---
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [
        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
        {"role": "assistant", "content": "How can I help you with the database?"}
    ]

for msg in st.session_state.messages:
    if msg["role"] == "system":
        st.info(msg["content"])
    elif msg["role"] == "assistant":
        st.text(msg["content"])

# User query input
user_query = st.text_area("Ask me anything!", key="user_query")
if st.button("Send"):
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.text("You: " + user_query)

        with st.spinner("Thinking..."):
            response = agent.run(user_query)
            st.session_state.messages.append({"role": "assistant", "content": response})

            if isinstance(response, pd.DataFrame):
                st.subheader("Query Result:")
                st.table(response)  # Display as DataFrame if applicable
            else:
                st.info(response)
    else:
        st.warning("Please enter a query before sending.")

# Button to toggle table view
if "tabulate_output" not in st.session_state:
    st.session_state["tabulate_output"] = False

if st.button("Toggle Table View"):
    st.session_state["tabulate_output"] = not st.session_state["tabulate_output"]
