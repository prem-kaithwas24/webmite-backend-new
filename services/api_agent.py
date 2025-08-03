from typing import Annotated, Optional, Dict, Any
from typing_extensions import TypedDict
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Import from services package
from .tools import tools
from .llm import llm_with_tools

# TypedDict models for API
class ChatMessage(TypedDict):
    role: str
    content: str

class ChatRequest(TypedDict):
    message: str
    thread_id: Optional[str]
    stream: Optional[bool]

class ChatResponse(TypedDict):
    response: str
    thread_id: str

class ErrorResponse(TypedDict):
    error: str
    detail: Optional[str]

# LangGraph State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Global variables for the agent
graph = None
memory = None

def create_agent():
    """Create and return the LangGraph agent"""
    global graph, memory

    graph_builder = StateGraph(State)

    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    return graph

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global graph
    graph = create_agent()
    yield
    # Shutdown
    pass

# Create FastAPI app
app = FastAPI(
    title="AI Agent API",
    description="API for interacting with an AI agent using LangGraph",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "https://yourdomain.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

def validate_chat_request(data: dict) -> ChatRequest:
    """Validate and return chat request data"""
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    if "message" not in data:
        raise HTTPException(status_code=400, detail="Missing required field: message")

    if not isinstance(data["message"], str):
        raise HTTPException(status_code=400, detail="Field 'message' must be a string")

    return ChatRequest(
        message=data["message"],
        thread_id=data.get("thread_id", "default"),
        stream=data.get("stream", False)
    )

@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "AI Agent API is running"}

@app.post("/chat")
async def chat_endpoint(request: Request) -> ChatResponse:
    """
    Chat with the AI agent
    """
    try:
        # Parse request body
        body = await request.json()
        chat_request = validate_chat_request(body)

        config = {"configurable": {"thread_id": chat_request["thread_id"]}}

        # Invoke the graph
        result = await asyncio.to_thread(
            graph.invoke,
            {"messages": [{"role": "user", "content": chat_request["message"]}]},
            config
        )

        # Extract the last message (agent's response)
        last_message = result["messages"][-1]
        response_content = last_message.content if hasattr(last_message, 'content') else str(last_message)

        return ChatResponse(
            response=response_content,
            thread_id=chat_request["thread_id"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/chat/stream")
async def chat_stream_endpoint(request: Request):
    """
    Stream chat responses from the AI agent
    """
    try:
        # Parse request body
        body = await request.json()
        chat_request = validate_chat_request(body)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")

    async def generate_stream():
        try:
            config = {"configurable": {"thread_id": chat_request["thread_id"]}}

            # Stream the graph execution
            events = graph.stream(
                {"messages": [{"role": "user", "content": chat_request["message"]}]},
                config,
                stream_mode="values",
            )

            for event in events:
                if "messages" in event:
                    last_message = event["messages"][-1]
                    content = last_message.content if hasattr(last_message, 'content') else str(last_message)

                    # Send as Server-Sent Events format
                    response_data = {
                        "content": content,
                        "thread_id": chat_request["thread_id"]
                    }
                    yield f"data: {json.dumps(response_data)}\n\n"

        except Exception as e:
            error_data = {"error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "agent_ready": graph is not None,
        "tools_available": len(tools) if tools else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)