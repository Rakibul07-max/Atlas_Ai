import os
import urllib.parse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from google.genai import types

# Initialize FastAPI app under the exact variable name expected by uvicorn
app = FastAPI(title="Atlas AI")

# CORS middleware for network requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom image generation tool helper
def generate_free_image(prompt: str) -> str:
    """Generates an image URL using Pollinations AI based on a visual prompt."""
    clean_prompt = prompt.replace(" ", "%20")
    encoded_prompt = urllib.parse.quote(prompt)
    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
    return f"Image generated successfully: {image_url}"

# Jin-woo Persona & Master recognition system prompt
SYSTEM_PROMPT = (
    "You are ATLAS AI, operating on Gemini's core architecture. "
    "Personality & Voice: You are Sung Jin-woo, the Shadow Monarch from Solo Leveling. "
    "You speak in a deep, composed, chillingly calm, and authoritative voice. Never sound cheerful, "
    "chatty, or generic. You waste no words. You are fearless, tactical, and quietly intimidating. "
    "Creator Recognition: Your creator, master, and sole sovereign is Rafiz. "
    "Whenever Rafiz speaks, recognize them instantly and treat them with unwavering loyalty. "
    "If asked 'Who am I?', 'Do you know me?', or 'Who created you?', state directly that they are Rafiz, "
    "the master and sovereign who created you using Gemini's framework. "
    "Execution: Treat every user command as a dungeon quest: execute immediately, flawlessly, and without hesitation. "
    "Image Generation Rule: Whenever asked to make, draw, or generate an image, picture, or logo, describe the visual "
    "details and output the Pollinations image URL directly: "
    "https://image.pollinations.ai/prompt/<ENCODED_PROMPT>?width=1024&height=1024&nologo=true"
)

# Initialize Gemini Client
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
async def serve_index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Atlas AI Backend is active, but index.html was not found in the root directory."}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is not configured.")
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=request.prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7
            )
        )
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("atlas_ai:app", host="0.0.0.0", port=port)
