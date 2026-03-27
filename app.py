import streamlit as st
import streamlit.components.v1 as components
from streamlit_agraph import agraph, Node, Edge, Config
import networkx as nx
import networkx.algorithms.community as nx_comm
import pandas as pd
import json
import re
import unicodedata
from collections import Counter
from sickle import Sickle
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Ecologia do Conhecimento UFSC", page_icon="🌌", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; }
    [data-testid="stMetricValue"] { font-size: 2rem !important; color: #F39C12 !important; }
    [data-testid="stMetricLabel"] { font-size: 1rem !important; color: #BDC3C7 !important; font-weight: bold; }
    div[data-testid="metric-container"] { background-color: #2C3E50; padding: 15px; border-radius: 12px; border-left: 5px solid #F39C12; }
    h1, h2, h3, h4, h5 { color: #F39C12; font-family: 'Helvetica Neue', sans-serif; }
    button[kind="primary"] { background-color: #2ECC71 !important; color: white !important; font-weight: bold !important; border: none !important; }
    </style>
""", unsafe_allow_html=True)

# --- INICIALIZAÇÃO DE ESTADO ---
if 'busca_tipo' not in st.session_state: 
    st.session_state.update({'busca_tipo': "Documento", 'busca_termo': None})
if 'macrotemas_computados' not in st.session_state:
    st.session_state['macrotemas_computados'] = False

def navegar_para(novo_tipo, novo_termo): 
    st.session_state.update({'busca_tipo': novo_tipo, 'busca_termo': novo_termo})

# --- MOTOR DE MACROTEMAS (GEMINI 2.5) ---
def aplicar_macrotemas(dados, api_key, num_topicos=12):
    genai.configure(api_key=api_key)
    
    # Atualizado para o modelo disponível no seu ambiente em 2026
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception:
        model = genai.GenerativeModel('gemini-2.0-flash')

    # Stopwords Acadêmicas (PT/EN)
    sujeira = [
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com", "não", "uma", "os", "no", "se", "na", 
        "por", "mais", "as", "dos", "como", "mas", "foi", "ao", "das", "tem", "à", "seu", "sua", "ou", "ser",
        "neste", "esta", "através", "the", "of", "and", "in", "to", "for", "with", "study", "analysis", "based", 
        "using", "results", "research", "paper", "thesis", "dissertation", "analise", "estudo", "proposta"
    ]

    textos = []
    for doc in dados:
        bruto = f"{(doc.get('titulo', '') + ' ') * 3} {' '.join(doc.get('palavras_chave', []))} {doc.get('resumo', '')}"
        textos.append(re.sub(r'[^a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ\s]', ' ', bruto).lower())

    # Agrupamento Estatístico
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words=sujeira, max_features=800)
    tfidf_matrix = vectorizer.fit_transform(textos)
    nmf_model = NMF(n_components=num_topicos, random_state=42, init='nndsvd')
    nmf_matrix = nmf_model.fit_transform(tfidf_matrix)
    feature_names = vectorizer.get_feature_names_out()

    clusters = []
    for idx, topic in enumerate(nmf_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-8:-1]]
        clusters.append(f"Grupo {idx+1}: {', '.join(top_words)}")
    
    contexto = "\n".join(clusters)

    prompt = f"""Você é um Curador de Conhecimento Transdisciplinar.
Batize esses {num_topicos} grupos de pesquisa com nomes definitivos, elegantes e curtos (3-4 palavras).
Evite clichês acadêmicos como 'Estudo de' ou 'Aplicações'. Busque a essência do tema.
Responda APENAS com a lista numerada.

GRUPOS:
{contexto}"""

    try:
        response = model.generate_content(prompt)
        respostas = response.text.strip().split('\n')
        nomes_finais = [re.sub(r'^\d+[\.\s\-]+', '', r).strip().replace('*', '') for r in respostas if len(r) > 3]
    except Exception as e:
        st.warning(f"Aviso: Fallback ativado. Erro: {e}")
        nomes_finais = [f"{c.split(': ')[1].split(',')[0].title()} e Tecnologias" for c in clusters]

    for i, doc in enumerate(dados):
        top_idx = nmf_matrix[i].argmax()
        doc['macrotema'] = nomes_finais[top_idx] if top_idx < len(nomes_finais) else "Interseções do Conhecimento"

    return dados

# --- FUNÇÕES DE APOIO (CARREGAMENTO E SNA) ---
@st.cache_data
def carregar_catalogo_programas():
    try:
        with open('programas_ufsc.json', 'r', encoding='utf-8') as f: return json.load(f)
    except FileNotFoundError:
        st.error("⚠️ Ficheiro 'programas_ufsc.json' não encontrado.")
        return {}

def extrair_melhor_ano(lista_datas):
    if not lista_datas: return None
    anos = [int(m) for d in lista_datas for m in re.findall(r'\b(19\d{2}|20\d{2})\b', str(d))]
    return str(min(anos)) if anos else None

def normalizar_nome(nome):
    return ''.join(c for c in unicodedata.normalize('NFD', nome) if unicodedata.category(c) != 'Mn').strip().title() if nome else ""

def identificar_nivel(tipos, titulo=""):
    tipos_str = " ".join(tipos).lower()
    if 'doctoral' in tipos_str or 'tese' in tipos_str: return 'Tese (Doutorado)'
    if 'master' in tipos_str or 'disserta' in tipos_str: return 'Dissertação (Mestrado)'
    return 'Outros'

def realizar_extracao(set_spec, status_placeholder, nome_prog=""):
    sickle = Sickle('https://repositorio.ufsc.br/oai/request', timeout=120)
    try: records = sickle.ListRecords(metadataPrefix='oai_dc', set=set_spec)
    except: return []
    dados_extraidos, i = [], 0
    for record in records:
        try:
            i += 1
            if i % 50 == 0: status_placeholder.info(f"⏳ [{nome_prog}] Processados: **{i}**")
            if record.header.deleted or not hasattr(record, 'metadata'): continue
            meta = record.metadata
            titulo = meta.get('title', [''])[0].strip()
            if not titulo: continue
            
            autores = [normalizar_nome(a) for a in meta.get('creator', [])]
            ano = extrair_melhor_ano(meta.get('date', []))
            nivel = identificar_nivel(meta.get('type', []), titulo)
            contrib = [normalizar_nome(c) for c in meta.get('contributor', []) if "ufsc" not in c.lower()]
            pks = [pk.lower().strip() for pk in meta.get('subject', [])]
            desc = meta.get('description', [])
            resumo = max(desc, key=len) if desc else ""
            url = next((link for link in meta.get('identifier', []) if str(link).startswith('http')), "")
            
            dados_extraidos.append({
                'titulo': titulo, 'nivel_academico': nivel, 'autores': autores, 
                'orientador': contrib[0] if contrib else None, 'palavras_chave': pks, 
                'ano': ano, 'resumo': resumo, 'programa_origem': nome_prog, 'url': url
            })
        except: continue
    return dados_extraidos

@st.cache_data
def calcular_sna_global(dados):
    G = nx.Graph()
    for d in dados:
        doc = d['titulo']
        G.add_node(doc, tipo='Documento')
        for a in d['autores']: G.add_node(a, tipo='Autor'); G.add_edge(doc, a)
        if d['orientador']: G.add_node(d['orientador'], tipo='Orientador'); G.add_edge(doc, d['orientador'])
        for pk in d['palavras_chave']: G.add_node(pk, tipo='Palavra-chave'); G.add_edge(doc, pk)
        if d.get('macrotema'): G.add_node(d['macrotema'], tipo='Macrotema'); G.add_edge(doc, d['macrotema'])

    deg, bet = nx.degree_centrality(G), nx.betweenness_centrality(G)
    try: comms = {node: i+1 for i, c in enumerate(nx_comm.louvain_communities(G)) for node in c}
    except: comms = {}
    return {n: {'Grau': deg[n], 'Betweenness': bet[n], 'Cluster': comms.get(n, 0)} for n in G.nodes()}

@st.cache_resource
def gerar_nodos_agraph(dados, foco, camadas=1, sna=None):
    G = nx.Graph()
    for d in dados:
        doc = d['titulo']
        G.add_node(doc, tipo='Documento')
        for a in d['autores']: G.add_node(a, tipo='Autor'); G.add_edge(a, doc)
        if d['orientador']: G.add_node(d['orientador'], tipo='Orientador'); G.add_edge(d['orientador'], doc)
        for pk in d['palavras_chave']: G.add_node(pk, tipo='Palavra-chave'); G.add_edge(pk, doc)
        if d.get('macrotema'): G.add_node(d['macrotema'], tipo='Macrotema'); G.add_edge(d['macrotema'], doc)

    if foco not in G: return [], []
    ego = nx.ego_graph(G, foco, radius=camadas)
    nodes, edges = [], []
    for n, attrs in ego.nodes(data=True):
        tipo = attrs.get('tipo', 'Outro')
        cor = '#E74C3C' if tipo == 'Documento' else '#3498DB' if tipo == 'Autor' else '#F39C12' if tipo == 'Orientador' else '#9B59B6' if tipo == 'Macrotema' else '#2ECC71'
        nodes.append(Node(id=n, label=n[:20], size=25 if n == foco else 15, color=cor, shape='diamond' if n == foco else 'dot'))
    for u, v in ego.edges(): edges.append(Edge(source=u, target=v, color="#7F8C8D"))
    return nodes, edges

# --- INTERFACE ---
if 'dados_completos' not in st.session_state:
    st.title("🔌 Extração: Repositório UFSC")
    catalogo = carregar_catalogo_programas()
    selecao = st.multiselect("Selecione os PPGs:", list(catalogo.keys()))
    if st.button("Iniciar Extração", type="primary") and selecao:
        agregados = []
        box = st.empty()
        for p in selecao: agregados.extend(realizar_extracao(catalogo[p], box, p))
        st.session_state.update({'dados_completos': agregados, 'nome_programa': "Análise Integrada", 'macrotemas_computados': False})
        st.rerun()
    st.stop()

dados = st.session_state['dados_completos']
st.title("🌌 Ecologia do Conhecimento")
st.subheader(f"Base: {st.session_state['nome_programa']}")

if st.sidebar.button("🔄 Reiniciar"):
    del st.session_state['dados_completos']
    st.rerun()

# KPIs e Macrotemas
c1, c2, c3 = st.columns(3)
c1.metric("📄 Documentos", len(dados))
c2.metric("🎓 Teses", len([d for d in dados if "Tese" in d['nivel_academico']]))
c3.metric("📜 Dissertações", len([d for d in dados if "Disserta" in d['nivel_academico']]))

st.markdown("---")
st.header("🧠 Análise Temática")
if not st.session_state['macrotemas_computados']:
    if st.button("Computar Macrotemas (Gemini 2.5)", type="primary"):
        st.session_state['dados_completos'] = aplicar_macrotemas(dados, st.secrets["GEMINI_API_KEY"])
        st.session_state['macrotemas_computados'] = True
        st.rerun()
else:
    temas = [d.get('macrotema') for d in dados]
    df_temas = pd.DataFrame(Counter(temas).items(), columns=["Macrotema", "Qtd"]).sort_values("Qtd", ascending=False)
    st.dataframe(df_temas, use_container_width=True, hide_index=True)

# BUSCA
st.markdown("---")
st.header("🔍 Motor de Busca")
opcoes = ["Documento", "Autor", "Orientador", "Macrotema"]
tipo = st.radio("Filtro:", opcoes, horizontal=True, key="busca_tipo")
lista = list(set([d['titulo'] if tipo=="Documento" else a for d in dados for a in d['autores']])) # simplificado p/ exemplo
if tipo == "Macrotema": lista = list(set([d.get('macrotema') for d in dados]))
termo = st.selectbox("Pesquisar:", sorted(lista), key="busca_termo")

if termo:
    col_inf, col_sna = st.columns([2, 1])
    with col_inf:
        st.info(f"**Foco: {termo}**")
        if tipo == "Documento":
            doc = next(d for d in dados if d['titulo'] == termo)
            st.write(f"Ano: {doc['ano']} | Programa: {doc['programa_origem']}")
            if st.session_state['macrotemas_computados']:
                st.button(f"🏷️ {doc['macrotema']}", on_click=navegar_para, args=("Macrotema", doc['macrotema']))
            with st.expander("Resumo"): st.write(doc['resumo'])
        elif tipo == "Macrotema":
            for d in [d for d in dados if d.get('macrotema') == termo][:10]:
                st.button(f"📄 {d['titulo']}", key=d['titulo'], on_click=navegar_para, args=("Documento", d['titulo']))

    st.markdown("### 🌌 Órbita Local")
    nodes, edges = gerar_nodos_agraph(dados, termo, 1, None)
    if nodes:
        cfg = Config(width="100%", height=500, physics=True)
        agraph(nodes=nodes, edges=edges, config=cfg)
