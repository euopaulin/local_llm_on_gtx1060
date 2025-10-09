# 🤖 Projeto LLM Local: Ollama + Docker + Agente de IA + Open-WebUI na GTX 1060 (6GB)

> implantação 100% local rodando em Linux Ubuntu 24.04, persistente, com GPU, Agent (ReAct) e RAG. Ideal para laboratório, PoC e estudo de MLOps/Engenharia de Prompt.

## 📌 Sumário
- [Visão Geral](#-visão-geral)
- [Arquitetura](#-arquitetura)
- [Requisitos](#-requisitos)
- [1) Infraestrutura & Setup (DevOps)](#1-infraestrutura--setup-devops)
  - [1.1) Configuração Essencial](#11-configuração-essencial)
  - [1.2) Verificação do Serviço](#12-verificação-do-serviço)
- [2) Projeto Python (Engineering & Agents)](#2-projeto-python-engineering--agents)
  - [2.1) Ambiente Virtual & Dependências](#21-ambiente-virtual--dependências)
  - [2.2) Medição de Desempenho (Tokens/s)](#22-medição-de-desempenho-tokenss)
- [3) Agente ReAct com Busca no Google](#3-agente-react-com-busca-no-google)
  - [3.1) Variáveis de Ambiente](#31-variáveis-de-ambiente)
  - [3.2) Código do Agent (ReAct Estrito)](#32-código-do-agent-react-estrito)
- [4) RAG Local com ChromaDB](#4-rag-local-com-chromadb)
- [5) Docker Compose (opcional)](#5-docker-compose-opcional)
- [6) Troubleshooting (Boss Fights)](#6-troubleshooting-boss-fights)
- [7) Segurança & Boas Práticas](#7-segurança--boas-práticas)
- [8) Cheatsheet](#8-cheatsheet)
- [Licença](#licença)

---

## 🧠 Visão Geral
Este guia documenta a configuração do **LLM open-source Mistral 7B** rodando **100% local** via **Ollama** dentro de **Docker**, acelerado por **GPU NVIDIA GTX 1060 (6GB)**. 
A pilha inclui:
- **Ollama** para servir o modelo (com quantização que cabe em 6 GB VRAM);
- **Agent (ReAct)** para usar **Google Search como Tool** e contornar _knowledge cutoff_;
- **RAG** local com **ChromaDB** + **Sentence-Transformers** para buscar conhecimento nos seus PDFs/textos.
- Implementação do LLM com a interface gráfica **Open-WebUI** para que fique com uma aparência semelhante aos famosos chats de genIA que conhecemos.

---

## Configuração da máquina usada para o laboratorio
- **Processador:** Ryzen 5 1600 6/12
- **Memória:** 16GB memória ram DDR4 2666mhz XPG
- **GPU:** GTX 1060 6GB Gitabyte G1 Gaming
- **PSU:** Fonte 600w

## 🏗️ Arquitetura
```text

                                           +------------------+
    +------------------------------------> |    Open-WebUI    |
    |                                      +------------------+
    |                                               ^
    |                                               |
    |                                               |
    |       +-------------------+         +------------------+          +--------------------------+
    |       |  Cliente (CLI/py) | <-----> |   Ollama (API)   | <------> |  GPU NVIDIA (GTX 1060)   |
    |       +-------------------+         +------------------+          +--------------------------+
    |               |                               ^
    |               v                               |
    |       +-------------------+                   |
    +-------|   Agent (ReAct)   | ----(Google)------+    +-----------------------+
            |  + RAG (Chroma)   | -----------------------|  Base de Conhecimento |
            +-------------------+                        |  (PDFs / .txt / etc)  |
                                                         +-----------------------+
```

---

## ✅ Requisitos
- **Linux** com Docker Engine ativo;
- **Driver NVIDIA** + **NVIDIA Container Toolkit** instalados;
- **GPU**: GTX 1060 **6GB VRAM** (funciona com quantização `q4_x` do Mistral);
- Acesso à internet **apenas** para baixar imagens/modelos e, se desejar, usar a Tool de busca;
- **Python 3.10+** (sugerido).

> Dica: teste a GPU com `nvidia-smi` e o suporte no Docker com `docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi`.

---

## 1) Infraestrutura & Setup (DevOps)

### 1.1) Configuração Essencial
**Pré-requisitos**: NVIDIA Driver + Container Toolkit e Docker OK.

**Permissão e Limpeza do Ollama nativo (evitar conflito):**
```bash
# Adicionar o usuário ao grupo docker (requer logout/login)
sudo usermod -aG docker $USER

# Parar serviços em conflito
sudo systemctl stop ollama || true
docker stop ollama || true
docker rm ollama || true

# Limpeza de cache local do ollama (nativo)
rm -rf ~/.ollama
```

**Subir o servidor Ollama com GPU (persistente):**
```bash
docker run -d --restart always --gpus=all   -v ollama:/root/.ollama   -p 11434:11434   --name ollama   ollama/ollama
```

> **Observação**: O contêiner baixa os modelos sob demanda. Você pode executar `docker exec -it ollama ollama pull mistral` para antecipar o download. Em placas com 6GB, priorize **quantizações** como `mistral:instruct-q4_K_M` para estabilidade.

### 1.2) Verificação do Serviço
```bash
docker ps
# Deve mostrar o contêiner 'ollama' com STATUS: Up X seconds/minutes
```

Teste rápido via CLI (baixa o modelo se necessário):
```bash
docker exec -it ollama ollama run mistral "Escreva um haicai sobre DevOps."
```

---

## 2) Projeto Python (Engineering & Agents)

### 2.1) Ambiente Virtual & Dependências
```bash
# Criar e Ativar Ambiente Virtual
python3 -m venv .venv
source .venv/bin/activate

# Bibliotecas essenciais (Agents + RAG)
pip install --upgrade pip
pip install langchain-community google-search-results
pip install chromadb sentence-transformers pypdf
```

> **Modelos de embedding**: por padrão, usaremos `sentence-transformers/all-MiniLM-L6-v2` (leve e eficiente).

### 2.2) Medição de Desempenho (Tokens/s)
> Métrica: **Throughput (Tokens/s)** e **Latência**. O exemplo abaixo estima T/s por contagem de “palavras” de saída (aproximação simples).

```python
import time
from langchain_community.llms import Ollama

MODELO = "mistral"  # ou "mistral:instruct-q4_K_M" se preferir instruções/quantização
llm = Ollama(model=MODELO, temperature=0.3)

def executa_e_mede(prompt: str):
    start_time = time.time()
    response = llm.invoke(prompt)
    elapsed = time.time() - start_time
    tokens_aprox = len(response.split())  # aproximação simples

    print(f"Latência Total: {elapsed:.2f} s")
    print(f"Throughput (T/s) ~ {tokens_aprox / elapsed:.2f}")
    print("Resposta:
", response[:500], "...")

executa_e_mede("Explique o que é DevOps em 3 parágrafos.")
# Em testes reais com GTX 1060 6GB, ~9–10 T/s é um valor típico com quantização adequada.
```

> Para medições mais precisas, considere **streaming** e contagem de *tokens* no servidor, ou comparar tempos em prompts/repostas fixas.

---

## 3) Agente ReAct com Busca no Google
Objetivo: **contornar knowledge cutoff** usando uma **Tool** de busca. O Agent decide quando consultar a internet.

### 3.1) Variáveis de Ambiente
```bash
export GOOGLE_API_KEY="SUA_CHAVE_AQUI"
export GOOGLE_CSE_ID="SEU_CX_ID_AQUI"
```

### 3.2) Código do Agent (ReAct Estrito)
```python
import os
from langchain_community.llms import Ollama
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.tools import GoogleSearchRun
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

# LLM (Ollama)
llm = Ollama(model="mistral", temperature=0.3)

# Tools: Google Search
search = GoogleSearchAPIWrapper()
google_tool = GoogleSearchRun(api_wrapper=search, name="GoogleSearch")
tools = [google_tool]

# Prompt ReAct: formato rígido para reduzir parsing errors
prompt_template = PromptTemplate.from_template(
    """
Você é um assistente de IA inteligente. Use as ferramentas fornecidas quando necessário.
Ferramentas disponíveis: {tools}

Siga rigorosamente o formato abaixo (português):
Question: a pergunta do usuário
Thought: seu raciocínio breve sobre o próximo passo
Action: a ação a executar, escolha UMA dentro de [{tool_names}]
Action Input: o input para a ação (texto)
Observation: o resultado da ação
... (este ciclo Thought/Action/Action Input/Observation pode repetir N vezes) ...
Thought: concluiu? Se sim, responda.
Final Answer: sua resposta final e objetiva ao usuário

Comece!
Question: {input}
{agent_scratchpad}
"""
)

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True  # autocorreção de formatação
)

pergunta = "Qual é o próximo jogo do São Paulo Futebol Clube e quando será?"
resultado = agent_executor.invoke({"input": pergunta})
print(resultado["output"])
```

> **Nota**: O Agent **pode** fazer chamadas externas. Se desejar **somente local**, remova a Tool e use RAG (seção abaixo).

---

## 4) RAG Local com ChromaDB
Exemplo mínimo para **perguntar sobre seus PDFs** localmente (sem internet):
```python
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA

# 1) Carregar documentos
loader = PyPDFLoader("meu_arquivo.pdf")
docs = loader.load()

# 2) Embeddings + VectorStore
emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(docs, embedding=emb, persist_directory=".ragdb")
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# 3) LLM + Cadeia de QA
llm = Ollama(model="mistral:instruct-q4_K_M", temperature=0.2)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# 4) Perguntar
resposta = qa.invoke({"query": "Resuma os pontos principais do documento."})
print(resposta["result"])
```

> Persistência: o Chroma em `persist_directory` guarda o índice entre execuções.

---

## 5) Docker Compose (opcional)
Se preferir **compor** serviços (ex.: Ollama + sua API/Frontend), um exemplo simples:
```yaml
# docker-compose.yml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    restart: always
    ports:
      - "11434:11434"
    volumes:
      - ollama:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: ["gpu"]  # requer docker compose plugin com suporte a GPU
volumes:
  ollama:
```

> Subir: `docker compose up -d`

---

## 6) Troubleshooting (Boss Fights)

- **`permission denied` ao usar Docker**  
  Adicione o usuário ao grupo `docker` e faça **logout/login**:
  ```bash
  sudo usermod -aG docker $USER
  ```

- **GPU não detectada (`no CUDA-capable device`)**  
  Verifique `nvidia-smi` no host e suporte no container (`nvidia-container-toolkit`).

- **Porta 11434 em uso**  
  Ajuste `-p 11434:11434` para outra porta livre (ex.: `-p 11435:11434`).

- **VRAM insuficiente / OOM**  
  Use quantizações **q4_K_M** ou similares: `mistral:instruct-q4_K_M`.  
  Evite prompts/respostas gigantes; reduza `temperature`/comprimento.

- **Modelos sumindo após reboot**  
  Garanta `--restart always` e **volume nomeado** `-v ollama:/root/.ollama`.

- **Agent quebrando com parsing**  
  Mantenha o **prompt ReAct rígido** e `handle_parsing_errors=True`.

---

## 7) Segurança & Boas Práticas
- **API Keys** em variáveis de ambiente ou `.env` (não commitar);
- **Rede**: exponha a API do Ollama apenas na LAN/localhost quando possível;
- **Controle de versão**: versionar somente **código**, não índices (`.ragdb`) nem caches;
- **Observabilidade**: logar tempos/respostas para avaliar **latência** e **T/s** ao longo do tempo.

---

## 8) Cheatsheet
```bash
# Parar/remover serviço antigo
sudo systemctl stop ollama || true
docker stop ollama || true && docker rm ollama || true

# Subir Ollama com GPU
docker run -d --restart always --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# Baixar modelo (quantizado) e testar
docker exec -it ollama ollama pull mistral:instruct-q4_K_M
docker exec -it ollama ollama run mistral:instruct-q4_K_M "Diga 'GG DevOps!'"
```

---

## Licença
Uso educacional/laboratorial. Verifique licenças dos modelos e bibliotecas antes de uso comercial.

---

**GG!** Você tem um **LLM local** com **GPU**, **Agent ReAct** e **RAG** prontos para missão. Se quiser, posso gerar um **Dockerfile** de API (FastAPI) para servir seu RAG via HTTP e integrar com frontend. 🎮
