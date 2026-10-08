import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from core.config import GROQ_API_KEY, GROQ_MODEL

print(f"Groq API Key set: {bool(GROQ_API_KEY)}")
print(f"Model: {GROQ_MODEL}")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
)

@tool
def profiler_agent_tool(goal: str) -> str:
    """Dispatch the autonomous Data Profiler agent. Returns JSON with profile_path."""
    return '{"profile_path": "test.json"}'

@tool
def sttm_agent_tool(goal: str) -> str:
    """Generate a Bronze STTM CSV. Returns JSON with sttm_path."""
    return '{"sttm_path": "test.csv"}'

print("Creating agent...")
agent = create_agent(llm, [profiler_agent_tool, sttm_agent_tool])
print("Invoking agent...")
result = agent.invoke({"messages": [HumanMessage(content="Just say hello, do not call any tools.")]})
print("SUCCESS:", result["messages"][-1].content[:200])
