import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate

MODELO = "Mistral"
ARQUIV_PDF = ["pdf/Teste.pdf", "pdf/turing.pdf"]  # Coloque seu arquivo PDF na pasta 'pdf'

#Chamando o arquivo PDF
caminho_pdf = os.path.join(os.path.dirname(__file__), ARQUIV_PDF[1])

print (f"Carregando o arquivo PDF: {caminho_pdf}...")
try:
    loader = PyPDFLoader(caminho_pdf)
    documentos = loader.load()
    print (f"Arquivo PDF carregado com sucesso!")
except Exception as e:
    print (f"Falha ao carregar o arquivo PDF: {e}")
    exit()


#Dividindo o documento em partes menores
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documentos_divididos = text_splitter.split_documents(documentos)
print (f"Documentos divididos em {len(documentos_divididos)} partes.")

#Criando os embeddings
print (f"Criando os embeddings(Convertendo texto em vetores)...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2", # Modelo leve e eficiente
    model_kwargs={'device': 'cpu'} # Forçamos CPU para não sobrecarregar sua GPU/VRAM
)

print("Indexando vetores no ChromaDB...")
vector_store = Chroma.from_documents(
    documents=documentos_divididos, 
    embedding=embeddings_model, 
    persist_directory="./chroma_db"
)

# O retriever é o componente que busca os chunks mais relevantes
retriever = vector_store.as_retriever()

# Define o LLM local no seu contêiner Ollama
llm = Ollama(model=MODELO) 

# Template de prompt: Instruímos o LLM a usar o contexto fornecido
prompt_template = ChatPromptTemplate.from_template(
    """
    Responda à pergunta do usuário APENAS com base no contexto fornecido. 
    Se a resposta não estiver no contexto, diga que você não tem informações sobre o assunto.
    
    CONTEXTO:
    {context}
    
    PERGUNTA DO USUÁRIO: {input}
    """
)

# Cria a chain que combina os documentos recuperados com o LLM (stuff documents)
document_chain = create_stuff_documents_chain(llm, prompt_template)

# Cria a chain final que orquestra a busca (retriever) e a geração (document_chain)
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# --- 5. INVOCAR A PERGUNTA ---
pergunta = "As ideias de Turing ainda são presentes nos dias atuais?" # Mude sua pergunta aqui!

print(f"\n--- PERGUNTA RAG: {pergunta} ---")

# Executa a chain RAG
resposta_completa = retrieval_chain.invoke({"input": pergunta})

# A resposta final está na chave 'answer'
print("\n--- RESPOSTA DO LLM BASEADA NO PDF ---\n")
print(resposta_completa["answer"])

# Exibe a fonte (chunks) que o LLM utilizou para responder
print("\n--- FONTES UTILIZADAS (Para conferência) ---")
for doc in resposta_completa["context"]:
    print(f"- Fonte: {doc.metadata.get('source')} (Pág: {doc.metadata.get('page')})")