import os
import gradio as gr
from google import genai
from dotenv import load_dotenv
from google.genai import types
import json
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

load_dotenv()

client = genai.Client()

server_params = StdioServerParameters(
    command="python",
    args= ["mcp_server.py"],
    env=os.environ
)

with open("instructions.json","r", encoding="utf-8") as file:
    dados_instrucao = json.load(file)

system_instructions = json.dumps(dados_instrucao,indent=2,ensure_ascii=False)

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

async def consultar_servidor_mcp(pergunta:str):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read,write) as session:
            
            await session.initialize()

            result = await session.call_tool(
                "consultar_documentacao",
                arguments={"pergunta":pergunta}
            )

            if result.content and len(result.content) > 0:
                return result.content[0].text
            return "Nenhuma informação encontrada"

async def generate_response(user_message,history):
    try:
        contexto_encontrado = await consultar_servidor_mcp(user_message)
        
        prompt = f"""
        Mensagem usuário: 
        {user_message}
        
        Contexto relevante:
        {contexto_encontrado}

        Responda à mensagem do usuário ultilizando o contexto encontrado, se necessário"""

        response = chat.send_message(prompt)
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