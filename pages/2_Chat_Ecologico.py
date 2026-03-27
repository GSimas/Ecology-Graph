import streamlit as st
import pandas as pd
import json
from groq import Groq

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Chatbot Ecológico | PPGEGC", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; }
    .stChatMessage { background-color: #2C3E50; border-radius: 10px; margin-bottom: 10px; }
    h1 { color: #F39C12; }
    </style>
""", unsafe_allow_html=True)

# --- INICIALIZAÇÃO DA API ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=GROQ_API_KEY)
except:
    st.error("Erro: Configure a 'GROQ_API_KEY' nos Secrets do Streamlit.")
    st.stop()

# --- FUNÇÕES DE DADOS COM FILTRO INTELIGENTE ---
@st.cache_data
def buscar_contexto_relevante(query, limite=25):
    """
    Em vez de enviar tudo, procura os 25 documentos mais relacionados 
    à pergunta do usuário para não estourar o limite de tokens.
    """
    try:
        with open('base_ppgegc.json', 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        # Transforma a pergunta em palavras-chave (minúsculas)
        termos_busca = query.lower().split()
        
        resultados = []
        for d in dados:
            # Cria uma "sopa de texto" do documento para busca
            texto_doc = f"{d['titulo']} {d['orientador']} {' '.join(d['palavras_chave'])}".lower()
            
            # Calcula uma pontuação simples de relevância
            score = sum(1 for termo in termos_busca if termo in texto_doc)
            
            # Se houver match ou se a base for pequena, adiciona
            if score > 0:
                resultados.append((score, d))
        
        # Ordena pelos mais relevantes
        resultados.sort(key=lambda x: x[0], reverse=True)
        
        # Se a busca não retornar nada, pega os 25 mais recentes (pelo ano)
        if not resultados:
            dados_ordenados = sorted(dados, key=lambda x: str(x.get('ano', '0')), reverse=True)
            selecionados = dados_ordenados[:limite]
        else:
            selecionados = [r[1] for r in resultados[:limite]]
            
        # Formata o resumo para a IA
        resumo = []
        for s in selecionados:
            resumo.append(f"- Doc: {s['titulo']} | Ori: {s['orientador']} | Ano: {s['ano']} | Temas: {', '.join(s['palavras_chave'])}")
        
        return "\n".join(resumo)
    except Exception as e:
        return f"Erro ao processar base: {e}"

# --- INTERFACE ---
st.title("🤖 Assistente de Inteligência Ecológica")
st.info("Otimizado: Agora o chat filtra apenas os dados relevantes para sua pergunta, respeitando os limites da API.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Perquise a ecologia do conhecimento..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # BUSCA APENAS O CONTEXTO NECESSÁRIO
    with st.spinner("Consultando arquivos do PPGEGC..."):
        contexto_selecionado = buscar_contexto_relevante(prompt)
    
    system_prompt = f"""
    Você é um consultor do PPGEGC/UFSC. Responda com base nestes dados selecionados:
    
    DADOS RELEVANTES:
    {contexto_selecionado}
    
    REGRAS:
    1. Responda apenas com base nos dados acima.
    2. Se a pergunta for geral (ex: 'Olá'), ignore os dados e seja cortês.
    3. Responda em Português do Brasil.
    """

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Usando o modelo mais atual e estável
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                ],
                temperature=0.3,
                stream=True
            )

            for chunk in completion:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Erro de comunicação: {e}")
