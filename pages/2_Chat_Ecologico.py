import streamlit as st
import pandas as pd
import json
import networkx as nx
from groq import Groq

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Analista IA de Redes | PPGEGC",
    page_icon="🤖",
    layout="wide"
)

# Estilização para manter a identidade visual do projeto
st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; }
    .stChatMessage { background-color: #2C3E50; border-radius: 10px; margin-bottom: 10px; }
    h1, h2, h3 { color: #F39C12; }
    </style>
""", unsafe_allow_html=True)

# --- MOTOR DE INTELIGÊNCIA SNA (SOCIAL NETWORK ANALYSIS) ---

@st.cache_data
def processar_inteligencia_rede(dados):
    """
    Constrói a rede integral e calcula métricas SNA para todos os nós.
    Retorna um resumo estruturado para o 'cérebro' da IA.
    """
    G = nx.Graph()
    
    # Construção da Topologia
    for d in dados:
        titulo = d['titulo']
        G.add_node(titulo, tipo='Documento', nivel=d.get('nivel_academico', 'N/A'))
        
        # Conexão com Orientador
        ori = d.get('orientador')
        if ori:
            G.add_node(ori, tipo='Orientador')
            G.add_edge(titulo, ori)
            
        # Conexão com Autores
        for autor in d.get('autores', []):
            G.add_node(autor, tipo='Autor')
            G.add_edge(titulo, autor)
            
        # Conexão com Conceitos (Palavras-chave)
        for pk in d.get('palavras_chave', []):
            G.add_node(pk, tipo='Conceito')
            G.add_edge(titulo, pk)

    # CÁLCULOS TOTAIS
    # 1. Degree (Volume de conexões diretas)
    degree_dict = dict(G.degree())
    # 2. Degree Centrality (Importância relativa na rede)
    deg_cent = nx.degree_centrality(G)
    # 3. Betweenness Centrality (Poder de intermediação / Brokers)
    bet_cent = nx.betweenness_centrality(G)

    # Organizar os dados em categorias para a IA entender o contexto
    def extrair_vips(tipo_filtro, metrica_dict, label, top_n=15):
        nós_filtro = [n for n, attr in G.nodes(data=True) if attr.get('tipo') == tipo_filtro]
        sub_dict = {n: met_val for n, met_val in met_dict.items() if n in nós_filtro}
        top = sorted(sub_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return "\n".join([f"- {nome}: {valor:.4f}" for nome, valor in top])

    # Construção do Relatório de Contexto
    relatorio = f"""
    --- RELATÓRIO DE INTELIGÊNCIA DA REDE PPGEGC ---
    Tamanho Total: {G.number_of_nodes()} entidades e {G.number_of_edges()} conexões.

    1. ORIENTADORES COM MAIOR VOLUME (Degree):
    {extrair_vips('Orientador', degree_dict, 'Conexões')}

    2. ORIENTADORES MAIS INFLUENTES/PONTES (Betweenness):
    (Estes professores conectam áreas diferentes do programa)
    {extrair_vips('Orientador', bet_cent, 'Betweenness')}

    3. CONCEITOS MAIS CENTRAIS (TEMAS DOMINANTES):
    {extrair_vips('Conceito', deg_cent, 'Centralidade')}

    4. CONCEITOS 'PONTE' (INTERDISCIPLINARES):
    (Palavras-chave que unem diferentes linhas de pesquisa)
    {extrair_vips('Conceito', bet_cent, 'Betweenness')}
    """
    return relatorio

# --- LOGICA DO CHAT ---

st.title("🤖 Chatbot Ecológico Estratégico")
st.markdown("Este chat tem consciência total da **matemática da rede** (Degree e Betweenness) de todos os membros.")

# API Key via Barra Lateral
with st.sidebar:
    st.header("⚙️ Configuração")
    api_key = st.text_input("Groq API Key:", type="password", help="Obtenha em console.groq.com")
    if not api_key:
        st.info("Aguardando API Key...")
        st.stop()

# Inicialização
client = Groq(api_key=api_key)

try:
    with open('base_ppgegc.json', 'r', encoding='utf-8') as f:
        base_dados = json.load(f)
except:
    st.error("Erro ao carregar base_ppgegc.json")
    st.stop()

# Processamento da "Consciência Matemática"
with st.spinner("A calcular a inteligência da rede..."):
    contexto_sna = processar_inteligencia_rede(base_dados)

# Histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada do utilizador
if prompt := st.chat_input("Ex: Quem é o orientador mais influente na conexão de áreas distintas?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # SYSTEM PROMPT COM DADOS REAIS
    system_prompt = f"""
    Você é o Analista Sénior de Ecologia do Conhecimento do PPGEGC/UFSC.
    Você não apenas lê títulos, mas interpreta a matemática da rede SNA.

    CONTEXTO ESTATÍSTICO DA REDE COMPLETA:
    {contexto_sna}

    SUAS DIRETRIZES:
    1. Quando perguntarem sobre "importância" ou "influência", use os dados de Betweenness.
    2. Quando perguntarem sobre "quem faz mais" ou "temas mais comuns", use os dados de Degree.
    3. Se um nome não estiver no relatório VIP acima, diga que ele faz parte da rede, mas não está no Top 15 atual de centralidade.
    4. Explique que o Betweenness indica o poder de 'Brokerage' (conectar ilhas de conhecimento).
    5. Responda sempre em Português com tom consultivo e acadêmico.
    """

    # Resposta da IA com Streaming
    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        full_res = ""
        
        try:
            completion = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                ],
                temperature=0.4, # Menor temperatura para evitar alucinações matemáticas
                stream=True
            )

            for chunk in completion:
                if chunk.choices[0].delta.content:
                    full_res += chunk.choices[0].delta.content
                    msg_placeholder.markdown(full_res + "▌")
            
            msg_placeholder.markdown(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})
            
        except Exception as e:
            st.error(f"Erro na Groq: {e}")
