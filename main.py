import os
import gradio as gr
from google import genai
from dotenv import load_dotenv
from google.genai import types
import json

load_dotenv()

client = genai.Client()

with open("instructions.json","r", encoding="utf-8") as file:
    dados_instrucao = json.load(file)

system_instructions = json.dumps(dados_instrucao,indent=2,ensure_ascii=False)

def generate_response(user_message,history):
    try:
        chat = client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_instructions,
            temperature=1.7,
            top_p=0.9,
            top_k=50,
            max_output_tokens=2048,
        )
    )
        response = chat.send_message(user_message)
        return response.text
    except Exception as e:
        print(e)

demo = gr.ChatInterface(
    fn=generate_response,
    title='Sherlock',
    description='Sherlock é o assistente detetive e você é o elementar caro Watson'
)


if __name__ == "__main__":
    demo.launch(share=True)