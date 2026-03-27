import streamlit as st
import pandas as pd
import json
import networkx as nx
from groq import Groq

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Analista IA de Redes | PPGEGC",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização Dark Mode
st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; }
    .stChatMessage { background-color: #2C3E50; border-radius: 10px; margin-bottom: 10px; }
    h1, h2, h3 { color: #F39C12; }
    </style>
""", unsafe_allow_html=True)

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
        titulo = d['titulo']
        G.add_node(titulo, tipo='Documento')
        ori = d.get('orientador')
        if ori:
            G.add_node(ori, tipo='Orientador')
            G.add_edge(titulo, ori)
        for pk in d.get('palavras_chave', []):
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
    1. ORIENTADORES (Volume/Degree):
    {extrair_vips('Orientador', degree_dict)}
    2. ORIENTADORES (Pontes/Betweenness):
    {extrair_vips('Orientador', bet_cent)}
    3. TEMAS (Centralidade):
    {extrair_vips('Conceito', deg_cent)}
    """

# --- INICIALIZAÇÃO ---
client = Groq(api_key=api_key)

try:
    with open('base_ppgegc.json', 'r', encoding='utf-8') as f:
        base_dados = json.load(f)
except Exception as e:
    st.error(f"Erro ao ler base_ppgegc.json: {e}")
    st.stop()

with st.spinner("Sincronizando inteligência da rede..."):
    contexto_sna = processar_inteligencia_rede(base_dados)

st.title("🤖 Chatbot Ecológico Estratégico")
st.caption("Acesso via Llama 3.3 70B & Groq Cloud")

# Histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada do Chat
if prompt := st.chat_input("Pergunte algo sobre a ecologia do PPGEGC..."):
    # Adiciona a mensagem do usuário ao estado
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Preparação do Prompt de Sistema
    system_prompt = f"""
    Você é o Analista Sénior de Ecologia do Conhecimento do PPGEGC/UFSC.
    DADOS MATEMÁTICOS DA REDE:
    {contexto_sna}
    
    INSTRUÇÕES:
    - Responda em Português do Brasil.
    - Use os dados SNA acima para justificar quem é mais influente (Betweenness) ou tem mais volume (Degree).
    - Seja acadêmico, mas direto.
    """

    # Construção Segura da Lista de Mensagens (Garante que só enviamos role e content)
    mensagens_limpas = [{"role": "system", "content": system_prompt}]
    for m in st.session_state.messages:
        mensagens_limpas.append({"role": m["role"], "content": m["content"]})

    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        full_res = ""
        
        try:
            # USANDO O MODELO LLAMA-3.3-70B-VERSATILE (Mais estável na Groq atualmente)
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
            st.error(f"Erro na API da Groq: {e}")
            st.info("Dica: Verifique se o modelo 'llama-3.3-70b-versatile' está disponível ou tente 'llama-3.1-8b-instant'.")
