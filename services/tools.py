import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()
tavily_api_key=os.getenv('TAVILY_API_KEY')

os.environ['TAVILY_API_KEY']=tavily_api_key

search_tool = TavilySearch(max_results=2)
tools = [search_tool]