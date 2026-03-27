import streamlit as st
import pandas as pd
import json
from groq import Groq

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Chatbot Ecológico | PPGEGC",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização para o Modo Escuro
st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; }
    .stChatMessage { background-color: #2C3E50; border-radius: 10px; margin-bottom: 10px; }
    h1 { color: #F39C12; }
    </style>
""", unsafe_allow_html=True)

# --- INICIALIZAÇÃO DA API ---
# O Streamlit busca automaticamente o segredo configurado no dashboard (Settings > Secrets)
try:
    if "GROQ_API_KEY" in st.secrets:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
        client = Groq(api_key=GROQ_API_KEY)
    else:
        st.error("ERRO: A chave 'GROQ_API_KEY' não foi encontrada nos Secrets do Streamlit.")
        st.stop()
except Exception as e:
    st.error(f"Erro ao carregar as chaves de segurança: {e}")
    st.stop()

# --- FUNÇÕES DE DADOS ---
@st.cache_data
def carregar_contexto_base():
    try:
        with open('base_ppgegc.json', 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        # Estrutura um resumo para o contexto da IA (Metadados essenciais)
        resumo = []
        for d in dados:
            resumo.append(
                f"Doc: {d['titulo']} | Nível: {d['nivel_academico']} | "
                f"Orientador: {d['orientador']} | Ano: {d['ano']} | "
                f"Conceitos: {', '.join(d['palavras_chave'])}"
            )
        return "\n".join(resumo)
    except Exception as e:
        return f"Erro ao acessar a base de dados: {e}"

# --- INTERFACE DO CHAT ---
st.title("🤖 Assistente de Inteligência Ecológica")
st.markdown("""
Analise a base de dados do PPGEGC em linguagem natural. 
Este chatbot utiliza o modelo **Llama 3.3 70B** para interpretação profunda dos dados.
""")

# Inicializa o histórico do chat na sessão
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe o histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Lógica de interação principal
if prompt := st.chat_input("Perquise a ecologia do conhecimento..."):
    # 1. Adiciona a pergunta do usuário ao histórico e exibe
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Carrega a base de conhecimento dinâmica
    contexto_dados = carregar_contexto_base()
    
    # 3. Define as instruções de comportamento da IA
    system_prompt = f"""
    Você é um consultor especializado na Ecologia do Conhecimento do PPGEGC/UFSC.
    Sua missão é analisar os dados fornecidos e responder de forma precisa e acadêmica.
    
    BASE DE CONHECIMENTO ATUALIZADA:
    {contexto_dados}
    
    REGRAS DE OURO:
    - Baseie-se apenas nos dados fornecidos para falar de pessoas, anos e temas.
    - Se não encontrar um dado específico, admita que não consta na base atual.
    - Identifique conexões entre orientadores e conceitos quando solicitado.
    - Use Português do Brasil com tom profissional.
    """

    # 4. Chamada para o motor de IA (Llama 3.3 via Groq)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Modelo atualizado e estável: llama-3.3-70b-versatile
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                ],
                temperature=0.4,
                max_tokens=2048,
                stream=True
            )

            # Efeito de streaming (texto aparecendo aos poucos)
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            # Finaliza a exibição da resposta
            message_placeholder.markdown(full_response)
            
            # 5. Salva a resposta da IA no histórico
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Houve uma falha na comunicação com o motor de IA: {e}")
