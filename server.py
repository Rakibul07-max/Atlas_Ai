import os
import traceback
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from atlas_ai import atlas_app, SYSTEM_PROMPT

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_history = [SystemMessage(content=SYSTEM_PROMPT)]

class PromptRequest(BaseModel):
    prompt: str

def extract_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(content)

@app.post("/api/chat")
async def chat_endpoint(req: PromptRequest):
    global chat_history
    try:
        chat_history.append(HumanMessage(content=req.prompt))
        
        result = atlas_app.invoke({"messages": chat_history})
        chat_history = result["messages"]
        raw_reply = chat_history[-1].content
        clean_reply = extract_text(raw_reply)
        return {"reply": clean_reply}
    except Exception as e:
        err_detail = traceback.format_exc()
        print(f"[Atlas Error]:\n{err_detail}")
        return JSONResponse(
            status_code=500,
            content={"reply": f"Atlas Backend Error: {str(e)}"}
        )

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
