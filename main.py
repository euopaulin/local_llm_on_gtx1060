from langchain_community.llms import Ollama 
import time
from typing import Tuple

MODELO = "mistral" 
llm = None

print(f"Conectando com modelo local: {MODELO}...")

try:
    llm = Ollama(model=MODELO)
    
    llm.invoke("Teste de conexão rápido.") 
    
    print(f"Modelo '{MODELO}' conectado com sucesso!")
    
except Exception as e:
    print(f"Falha CRÍTICA ao conectar o modelo. Verifique o Docker: {e}")
    exit()

def executa_e_mede(prompt: str) -> Tuple[str, float, int]:

    print(f"\nEU: {prompt}")

    start_time = time.time()

    response = llm.invoke(prompt)
    
    tempo_total = time.time() - start_time
    numero_tokens = len(response.split())

    print(f"\nIA: {response}")

    return response, tempo_total, numero_tokens


def primeira():
    prompt = "Me chamo Paulo!"

    _, tempo, tokens = executa_e_mede(prompt) 
    
    print("\n--- MÉTRICAS DA PRIMEIRA CHAMADA (Performance) ---")
    print(f"Tokens Gerados: {tokens}")
    print(f"Latência Total: {tempo:.2f} segundos")
    print(f"Throughput: {tokens / tempo:.2f} T/s")

def segunda():
    prompt = "Qual é o jogo de video game mais bem avalidado da história?"

    _, tempo, tokens = executa_e_mede(prompt)
    
    print("\n--- MÉTRICAS DA SEGUNDA CHAMADA (Performance) ---")
    print(f"Tokens Gerados: {tokens}")
    print(f"Latência Total: {tempo:.2f} segundos")
    print(f"Throughput: {tokens / tempo:.2f} T/s")

primeira()
segunda()