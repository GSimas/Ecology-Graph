import streamlit as st
import pandas as pd
import json
import networkx as nx
from groq import Groq

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Analista IA de Redes | Universal",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização Dark Mode
st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; }
    .stChatMessage { background-color: #2C3E50; border-radius: 10px; margin-bottom: 10px; }
    h1, h2, h3 { color: #F39C12; font-family: 'Helvetica Neue', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# --- PROTEÇÃO E LEITURA DA MEMÓRIA VIVA (SESSION STATE) ---
if 'dados_completos' not in st.session_state or not st.session_state['dados_completos']:
    st.warning("⚠️ Nenhuma base de dados carregada na memória.")
    st.info("Por favor, vá à página inicial e inicie a extração de um Programa de Pós-Graduação antes de iniciar o chat.")
    st.stop()

base_dados = st.session_state['dados_completos']
nome_programa = st.session_state.get('nome_programa', 'Programa Desconhecido')

# --- CONFIGURAÇÃO AUTOMÁTICA DA API KEY ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("⚠️ Erro: API Key não encontrada nos Secrets do Streamlit.")
    st.stop()

# --- MOTOR DE INTELIGÊNCIA SNA ---
@st.cache_data
def processar_inteligencia_rede(dados):
    G = nx.Graph()
    for d in dados:
        titulo = d.get('titulo')
        if not titulo: continue
        G.add_node(titulo, tipo='Documento')
        ori = d.get('orientador')
        if ori:
            G.add_node(ori, tipo='Orientador')
            G.add_edge(titulo, ori)
        for pk in d.get('palavras_chave', []):
            if pk:
                G.add_node(pk, tipo='Conceito')
                G.add_edge(titulo, pk)

    degree_dict = dict(G.degree())
    deg_cent = nx.degree_centrality(G)
    bet_cent = nx.betweenness_centrality(G)

    def extrair_vips(tipo_filtro, metrica_dict, top_n=15):
        nós_filtro = [n for n, attr in G.nodes(data=True) if attr.get('tipo') == tipo_filtro]
        sub_dict = {n: met_val for n, met_val in metrica_dict.items() if n in nós_filtro}
        top = sorted(sub_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return "\n".join([f"- {nome}: {valor:.4f}" for nome, valor in top])

    return f"""
    1. ORIENTADORES (Volume/Degree Centrality):
    {extrair_vips('Orientador', deg_cent)}
    
    2. ORIENTADORES (Pontes e Conexões Diversas/Betweenness):
    {extrair_vips('Orientador', bet_cent)}
    
    3. TEMAS/CONCEITOS MAIS CENTRAIS:
    {extrair_vips('Conceito', deg_cent)}
    """

# --- INICIALIZAÇÃO DO CHAT E DA IA ---
client = Groq(api_key=api_key)

with st.spinner(f"A treinar a IA com a matemática da rede do {nome_programa}..."):
    contexto_sna = processar_inteligencia_rede(base_dados)

st.title("🤖 Chatbot Estratégico (SNA)")
st.caption(f"Acesso via Llama 3.3 70B | Especializado em: {nome_programa}")

# Histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada do Chat
if prompt := st.chat_input("Ex: Quem domina as orientações e quem faz a ponte entre áreas?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    system_prompt = f"""
    Você é o Analista Sénior de Ecologia do Conhecimento responsável por analisar o programa: {nome_programa} da UFSC.
    Você possui acesso em tempo real aos cálculos topológicos da rede académica deste programa.

    DADOS MATEMÁTICOS DA REDE ATUAL:
    {contexto_sna}
    
    INSTRUÇÕES CRÍTICAS:
    - Responda em Português do Brasil de forma assertiva e especializada.
    - Quando o usuário perguntar quem é o "líder" em volume, baseie-se na lista de Degree Centrality.
    - Quando o usuário perguntar quem é o "broker", o "influenciador" ou quem "conecta saberes", use rigorosamente a lista de Betweenness Centrality.
    - Se perguntarem sobre assuntos ou tendências, cite os Temas Centrais.
    """

    mensagens_limpas = [{"role": "system", "content": system_prompt}]
    for m in st.session_state.messages:
        mensagens_limpas.append({"role": m["role"], "content": m["content"]})

    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        full_res = ""
        
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=mensagens_limpas,
                temperature=0.4,
                stream=True
            )

            for chunk in completion:
                if chunk.choices[0].delta.content:
                    full_res += chunk.choices[0].delta.content
                    msg_placeholder.markdown(full_res + "▌")
            
            msg_placeholder.markdown(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})
            
        except Exception as e:
            st.error(f"Erro na comunicação com a IA: {e}")
