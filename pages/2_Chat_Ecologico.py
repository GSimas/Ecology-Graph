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

# --- FUNÇÕES DE DADOS ---
@st.cache_data
def carregar_contexto_base():
    try:
        with open('base_ppgegc.json', 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        # Criamos um resumo textual para a IA não se perder em milhares de linhas
        # Selecionamos os campos essenciais para economizar "tokens"
        resumo = []
        for d in dados[:150]: # Enviamos os 150 registros mais recentes para contexto
            resumo.append(
                f"Título: {d['titulo']} | Nível: {d['nivel_academico']} | "
                f"Autor: {', '.join(d['autores'])} | Orientador: {d['orientador']} | "
                f"Ano: {d['ano']} | Conceitos: {', '.join(d['palavras_chave'])}"
            )
        return "\n".join(resumo)
    except:
        return "Erro ao carregar base de dados."

# --- INTERFACE DO CHAT ---
st.title("🤖 Chatbot da Ecologia do Conhecimento")
st.markdown("""
Este assistente utiliza Inteligência Artificial (Llama 3 via Groq) para analisar a base de dados do PPGEGC. 
Você pode perguntar coisas como: *'Quem mais orienta sobre Gestão do Conhecimento?'* ou *'Qual a evolução das teses sobre IA?'*
""")

# Configuração da API Key (Segurança)
# Dica: No Streamlit Cloud, você pode colocar isso em 'Secrets'
with st.sidebar:
    st.header("Configuração")
    api_key = st.text_input("Insira sua Groq API Key:", type="password")
    st.info("Obtenha sua chave grátis em console.groq.com")

if not api_key:
    st.warning("Por favor, insira a API Key na barra lateral para conversar.")
    st.stop()

client = Groq(api_key=api_key)

# Inicializa o histórico do chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Lógica do Chat
if prompt := st.chat_input("Pergunte algo sobre a base do PPGEGC..."):
    # Adiciona pergunta do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Preparar Contexto para a IA
    contexto_dados = carregar_contexto_base()
    
    system_prompt = f"""
    Você é um especialista em Ecologia do Conhecimento e consultor do PPGEGC/UFSC.
    Seu objetivo é responder perguntas baseadas estritamente nos dados abaixo.
    
    DADOS DA BASE:
    {contexto_dados}
    
    REGRAS:
    1. Se a informação não estiver nos dados, diga que não possui esse registro específico.
    2. Seja conciso, acadêmico e útil.
    3. Se o usuário perguntar sobre tendências, analise os anos e os conceitos fornecidos.
    4. Você fala Português do Brasil.
    """

    # Chamada para a LLM
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
                temperature=0.5,
                max_tokens=1024,
                stream=True
            )

            for chunk in completion:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            # Adiciona resposta da IA ao histórico
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Erro na comunicação com a IA: {e}")
