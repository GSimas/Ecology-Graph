import streamlit as st
import pandas as pd
import json
from groq import Groq

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Chatbot Ecológico | PPGEGC",
    page_icon="🤖",
    layout="wide"
)

# Estilização para o Modo Escuro
st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; }
    .stChatMessage { background-color: #2C3E50; border-radius: 10px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- INICIALIZAÇÃO DA API (AUTOMÁTICA VIA SECRETS) ---
# Tenta obter a chave dos secrets. Se não encontrar, avisa o administrador.
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=GROQ_API_KEY)
except Exception:
    st.error("ERRO: A GROQ_API_KEY não foi configurada nos Secrets do Streamlit.")
    st.stop()

# --- FUNÇÕES DE DADOS ---
@st.cache_data
def carregar_contexto_base():
    try:
        with open('base_ppgegc.json', 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        # Estrutura um resumo compacto para fornecer contexto à IA
        # Enviamos os metadados principais dos documentos para o "cérebro" do Chatbot
        resumo = []
        for d in dados:
            resumo.append(
                f"Doc: {d['titulo']} | Nível: {d['nivel_academico']} | "
                f"Orientador: {d['orientador']} | Ano: {d['ano']} | "
                f"Conceitos: {', '.join(d['palavras_chave'])}"
            )
        return "\n".join(resumo)
    except:
        return "Erro ao aceder à base de dados base_ppgegc.json."

# --- INTERFACE DO CHAT ---
st.title("🤖 Assistente de Inteligência Ecológica")
st.markdown("""
Este chatbot analisa em tempo real a base de dados do PPGEGC para responder a questões complexas sobre a estrutura do conhecimento no programa.
""")

# Inicializa o histórico do chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe mensagens anteriores do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Lógica de interação
if prompt := st.chat_input("Perquise a ecologia do conhecimento..."):
    # Adiciona a pergunta ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepara o prompt do sistema com a base de dados completa
    contexto_dados = carregar_contexto_base()
    
    system_prompt = f"""
    Você é um consultor especializado na Ecologia do Conhecimento do PPGEGC/UFSC.
    Sua missão é analisar os dados fornecidos e responder de forma precisa, acadêmica e perspicaz.
    
    BASE DE CONHECIMENTO DISPONÍVEL:
    {contexto_dados}
    
    INSTRUÇÕES:
    1. Baseie-se exclusivamente nos dados acima para responder sobre pessoas, anos e temas.
    2. Se identificar tendências (ex: aumento de temas sobre IA), mencione-as.
    3. Seja cortês, mas direto ao ponto.
    4. Responda sempre em Português.
    """

    # Chamada para a LLM (Llama 3 70B para maior profundidade analítica)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            completion = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                ],
                temperature=0.4,
                max_tokens=1500,
                stream=True
            )

            # Efeito de streaming para a resposta
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Houve uma falha na comunicação com o motor de IA: {e}")
