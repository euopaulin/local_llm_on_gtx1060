import os
from langchain_community.llms import Ollama
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.tools import GoogleSearchRun
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

# --- 1. CONFIGURAÇÃO DA INFRA E LLM ---
MODELO = "mistral"
try:
    llm = Ollama(model=MODELO, temperature=0.3)
except Exception as e:
    print("ERRO: Falha ao conectar ao Ollama. Verifique o Docker.")
    exit()

search = GoogleSearchAPIWrapper()
google_tool = GoogleSearchRun(api_wrapper=search, name="GoogleSearch")
tools = [google_tool]

prompt_template = PromptTemplate.from_template("""
    You are an intelligent AI assistant. Use the tools provided to answer the user's question.
    You have access to the following tools:
    {tools}
    
    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    {agent_scratchpad}
    """)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template
)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True # Permite ao agente tentar corrigir o próprio erro!
)

# --- 6. EXECUÇÃO ---
pergunta = "Qual o presidente atual do Estados Unidos?"

print(f"--- AGENT INICIADO ---\n")
print(f"PERGUNTA: {pergunta}")

# A flag 'verbose=True' do AgentExecutor mostrará o raciocínio da IA (Chain of Thought)!
resultado = agent_executor.invoke({"input": pergunta})

print("\n--- RESPOSTA FINAL DA IA (COM BUSCA NA WEB) ---")
print(resultado['output'])