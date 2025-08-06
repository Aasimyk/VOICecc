import subprocess

LLAMA_EXE_PATH = "C:/Users/ASIM YASH/Voice_chatbot/llama.cpp/build/bin/Release/llama-run.exe"
MODEL_PATH = "file://C:/Users/ASIM YASH/Voice_chatbot/models/mistral-7b-instruct-v0.1.Q2_K.gguf"

def get_llama_response(prompt: str) -> str:
    try:
        result = subprocess.run(
            [LLAMA_EXE_PATH, MODEL_PATH, prompt],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        return result.stdout.strip() or f"❌ LLM error: {result.stderr.strip() or 'Empty output'}"
    except Exception as e:
        return f"❌ LLM error: {str(e)}"
