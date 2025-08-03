import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "test-play-langgraph"

llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# Import tools after LLM initialization to avoid circular imports
from .tools import tools
llm_with_tools = llm.bind_tools(tools)
