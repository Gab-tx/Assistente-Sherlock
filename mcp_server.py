import os
import sys
from dotenv import load_dotenv
from rag_utils import RAGengine
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("Servidor MCP Sherlock","0.1")

arquivos_pdf = [
    r"pdfs\797.pdf",
    r"pdfs\Humanidades digitais e o mundo lusófono_ebook final_sobre autores atualizado.pdf"
]

print("Carregando RAG Engine", file=sys.stderr)

rag_engine = RAGengine(arquivos_pdf)
print("RAG Engine carregado com sucesso",file=sys.stderr)

@mcp.tool()
def consultar_documentacao(pergunta:str) -> str:
    try:
        contexto = rag_engine.buscar_contexto(pergunta)
        if not contexto:
            return "Desculpe não consegui encontrar informações relevante nos documentos"
        return contexto
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    mcp.run()