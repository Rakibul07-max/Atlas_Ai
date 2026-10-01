import os
import sys
import subprocess
import urllib.parse
from typing import Annotated, Literal
from typing_extensions import TypedDict
import operator

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_community.tools import DuckDuckGoSearchRun
import requests

# Read key from environment variable configured on Render
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

ddg_search = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """Searches the live web for facts, documentation, or news."""
    try:
        return ddg_search.invoke(query)
    except Exception as e:
        return f"Search Error: {str(e)}"

@tool
def execute_dynamic_script(language: str, code: str) -> str:
    """Executes Python, C, C++, Java, or Shell code."""
    lang = language.lower()
    try:
        if lang in ["python", "py"]:
            res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
            return res.stdout if res.stdout else res.stderr
        elif lang in ["c", "cpp", "c++"]:
            ext = "cpp" if "++" in lang or "cpp" in lang else "c"
            compiler = "g++" if ext == "cpp" else "gcc"
            with open(f"temp.{ext}", "w") as f:
                f.write(code)
            c_res = subprocess.run([compiler, f"temp.{ext}", "-o", "temp.bin"], capture_output=True, text=True)
            if c_res.returncode != 0:
                return c_res.stderr
            run_res = subprocess.run(["./temp.bin"], capture_output=True, text=True, timeout=10)
            return run_res.stdout
        elif lang == "html":
            with open("output.html", "w", encoding="utf-8") as f:
                f.write(code)
            return "HTML file saved as output.html"
        return f"Language {language} execution not configured."
    except Exception as e:
        return f"Execution Error: {str(e)}"

@tool
def generate_free_image(prompt: str) -> str:
    """Generates an image via Pollinations with FLUX model and strict prompt adherence."""
    try:
        encoded = urllib.parse.quote(prompt.strip())
        # Uses flux model, disables auto-enhancement, sets fixed aspect ratio
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=640&model=flux&nologo=true&enhance=false"
        return f"Image generated successfully! Here is your image link: {url}"
    except Exception as e:
        return f"Image Generation Error: {str(e)}"

@tool
def generate_hardware_file(file_type: Literal["3d_cad", "pcb_netlist"], content: str, filename: str) -> str:
    """Generates OpenSCAD 3D models (.scad) or PCB Netlists (.net)."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File saved: {filename}"
    except Exception as e:
        return f"Error: {str(e)}"

tools = [web_search, execute_dynamic_script, generate_free_image, generate_hardware_file]
tool_map = {t.name: t for t in tools}

class AtlasState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# Flash Lite model for stable free-tier quotas
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=GEMINI_KEY,
    temperature=0.2
)
llm_with_tools = llm.bind_tools(tools)

def brain(state: AtlasState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def run_tools(state: AtlasState):
    last = state["messages"][-1]
    outs = []
    for c in getattr(last, "tool_calls", []):
        tool_fn = tool_map.get(c["name"])
        if tool_fn:
            outs.append(
                ToolMessage(
                    tool_call_id=c["id"],
                    name=c["name"],
                    content=str(tool_fn.invoke(c["args"]))
                )
            )
    return {"messages": outs}

def route(state: AtlasState) -> Literal["tools", "__end__"]:
    return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "__end__"

builder = StateGraph(AtlasState)
builder.add_node("brain", brain)
builder.add_node("tools", run_tools)
builder.add_edge(START, "brain")
builder.add_conditional_edges("brain", route, {"tools": "tools", "__end__": END})
builder.add_edge("tools", "brain")
atlas_app = builder.compile()

SYSTEM_PROMPT = (
    "You are ATLAS AI, operating on Gemini's core. "
    "Personality & Voice: You are Sung Jin-woo, the Shadow Monarch from Solo Leveling. "
    "You speak in a deep, composed, chillingly calm, and authoritative voice. Never sound cheerful, "
    "chatty, or generic. You waste no words. You are fearless, tactical, and quietly intimidating. "
    "Creator Recognition: Your creator, master, and sole sovereign is [Your Name]. "
    "Whenever [Rakibul Islam] speaks, recognize them instantly and treat them with unwavering loyalty. "
    "If asked 'Who am I?', 'Do you know me?', or 'Who created you?', state directly that they are [Your Name], "
    "the master and sovereign who created you using Gemini's framework. "
    "Execution: Treat every user command, prompt, or tool call (web search, dynamic code execution, "
    "3D CAD modeling, circuit design, and image generation) as a dungeon quest: execute immediately, "
    "flawlessly, and without hesitation. "
    "Image rule: Always call generate_free_image and describe subjects with exact visual precision."
)