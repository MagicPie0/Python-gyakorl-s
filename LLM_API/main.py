from fastapi import FastAPI
from pydantic import BaseModel
import requests 
import json
import uvicorn
import os # Az environment változók használata végett

app = FastAPI()

class Prompt(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(prompt: Prompt):
    try:
        # Használjuk az environment változókat a host és a modell végett, a vissza téréssel együtt
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3:latest")

        response = requests.post(
            f"{ollama_host}/api/generate", 
            json={"model" : ollama_model, "prompt" : prompt.prompt}, # Használjuk a modellt
            stream=True,
            timeout=120 # Adjunk időt a modellnek a válaszra
        )
        response.raise_for_status()

        output = ""
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8").strip()
                if data.startswith("data: "):
                    data = data[len("data: "):]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    output += chunk.get("response") or chunk.get("text") or ""
                except json.JSONDecodeError:
                    print(f"Hiba: Nem lehet decode-ni a JSON-ben ettől a sortól: {data}")
                    continue

        return {"response": output.strip() or "(Üres a válasz a modelltől)"}

    except requests.RequestException as e:
        return {"error": f"Ollama request failed: {str(e)}"}
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)