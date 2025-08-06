# === main.py ===
import os
import uuid
import whisper
import ffmpeg
import torch
import asyncio
import subprocess
from gtts import gTTS
from io import BytesIO

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState

# ─── Configuration ─────────────────────────────
AUDIO_DIR = "../audio_chunks"
LLAMA_EXE_PATH = "C:/Users/ASIM YASH/Voice_chatbot/llama.cpp/build/bin/Release/llama-run.exe"
MODEL_PATH = "file://C:/Users/ASIM YASH/Voice_chatbot/models/mistral-7b-instruct-v0.1.Q2_K.gguf"
os.makedirs(AUDIO_DIR, exist_ok=True)

# ─── FastAPI App ──────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ─── Model Initialization ─────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")
whisper_model = whisper.load_model("base")

# ─── Utility Function ─────────────────────────
def get_llama_response(prompt: str) -> str:
    command = [LLAMA_EXE_PATH, MODEL_PATH, prompt]
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        output = result.stdout.strip() or result.stderr.strip() or "❌ No output from LLM"
        return (
            output.replace("<|im_end|>", "")
                  .replace("<|im_sep|>", "")
                  .replace("<|im_start|>", "")
                  .replace("Assistant:", "")
                  .replace("assistant:", "")
                  .replace("\x1b[0m", "")
                  .strip()
        )
    
        if "assistant" in output:
            output = output.split("assistant")[-1].strip(": \n")
        elif "Assistant" in output:
            output = output.split("Assistant")[-1].strip(": \n")

        return output.strip()
    except Exception as e:
        return f"❌ LLM error: {str(e)}"

# ─── WebSocket Audio Stream Handler ───────────
@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connected")

    try:
        while True:
            audio_id = str(uuid.uuid4())
            webm_path = os.path.join(AUDIO_DIR, f"{audio_id}.webm")
            wav_path = os.path.join(AUDIO_DIR, f"{audio_id}.wav")

            try:
                chunk = await asyncio.wait_for(websocket.receive_bytes(), timeout=30.0)
                with open(webm_path, "wb") as f:
                    f.write(chunk)
            except asyncio.TimeoutError:
                print("⏱️ Timeout waiting for audio, skipping...")
                continue
            except WebSocketDisconnect:
                print("❌ WebSocket disconnected")
                break

            print(f"🔊 Processing audio chunk {audio_id}")

            try:
                ffmpeg.input(webm_path).output(wav_path, ac=1, ar="16000").run(overwrite_output=True)
            except ffmpeg.Error as e:
                print("❌ FFmpeg error:", e)
                continue

            result = whisper_model.transcribe(wav_path)
            transcript = result["text"].strip()

            if not transcript:
                print("⚠️ Empty transcription, skipping...")
                continue

            print(f"🎧 You: {transcript}")
            response = get_llama_response(f"You: {transcript}")
            print(f"🤖 AI: {response}")

            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    # Send response text to frontend
                    await websocket.send_text(
                        f"<div class='chat'><b>You:</b> {transcript}</div>"
                        f"<div class='chat bot'><b>AI:</b> {response}</div>"
                    )

                    # Generate TTS response
                    tts = gTTS(text=response, lang="en")
                    mp3_fp = BytesIO()
                    tts.write_to_fp(mp3_fp)
                    mp3_fp.seek(0)

                    # Send audio response to frontend
                    await websocket.send_bytes(mp3_fp.read())
            except Exception as e:
                print("⚠️ Send error:", e)
                break

    except Exception as e:
        print("🔥 Unexpected error:", e)
    finally:
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
        except RuntimeError:
            pass
        print("🔒 WebSocket closed")
