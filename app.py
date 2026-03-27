import streamlit as st
import streamlit.components.v1 as components
from streamlit_agraph import agraph, Node, Edge, Config
import networkx as nx
import networkx.algorithms.community as nx_comm
from pyvis.network import Network
import pandas as pd
import json
import re
import unicodedata
from sickle import Sickle
from sickle.oaiexceptions import NoRecordsMatch

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

# --- FUNÇÕES DE BACKEND (EXTRAÇÃO E BUSCA) ---
@st.cache_data
def carregar_catalogo_programas():
    try:
        with open('programas_ufsc.json', 'r', encoding='utf-8') as f: return json.load(f)
    except FileNotFoundError:
        st.error("⚠️ Ficheiro 'programas_ufsc.json' não encontrado na raiz.")
        return {}

def extrair_melhor_ano(lista_datas):
    if not lista_datas: return None
    anos = [int(m) for d in lista_datas for m in re.findall(r'\b(19\d{2}|20\d{2})\b', str(d))]
    return str(min(anos)) if anos else None

def normalizar_nome(nome):
    return ''.join(c for c in unicodedata.normalize('NFD', nome) if unicodedata.category(c) != 'Mn').strip().title() if nome else ""

def normalizar_palavra_chave(pk):
    return ''.join(c for c in unicodedata.normalize('NFD', pk.lower().strip()) if unicodedata.category(c) != 'Mn') if pk else ""

def identificar_nivel(tipos, titulo=""):
    tipos_str = " ".join(tipos).lower()
    if 'doctoral' in tipos_str or 'tese' in tipos_str or 'tese' in titulo.lower(): return 'Tese (Doutorado)'
    if 'master' in tipos_str or 'disserta' in tipos_str or 'disserta' in titulo.lower(): return 'Dissertação (Mestrado)'
    return 'Outros'

def realizar_extracao(set_spec, status_placeholder, nome_prog=""):
    sickle = Sickle('https://repositorio.ufsc.br/oai/request', timeout=120)
    try: records = sickle.ListRecords(metadataPrefix='oai_dc', set=set_spec)
    except: return []
    dados_extraidos, titulos_vistos, iterator, i = [], set(), iter(records), 0
    
    while True:
        try: record = next(iterator)
        except StopIteration: break
        except Exception: break
            
        try:
            i += 1
            if i % 50 == 0: status_placeholder.info(f"⏳ [{nome_prog}] A extrair documentos... Já processados: **{i}**")
            if record.header.deleted or not hasattr(record, 'metadata') or not record.metadata: continue
            meta = record.metadata
            titulo = meta.get('title', [''])[0].strip()
            if not titulo or titulo in titulos_vistos: continue
            titulos_vistos.add(titulo)
            
            autores = [normalizar_nome(a) for a in meta.get('creator', []) if a.strip()]
            ano_real = extrair_melhor_ano(meta.get('date', []))
            nivel = identificar_nivel(meta.get('type', []), titulo)
            contrib = [normalizar_nome(c) for c in meta.get('contributor', []) if "ufsc" not in c.lower() and "universidade" not in c.lower()]
            orientador = contrib[0] if len(contrib) > 0 else None
            co_orientadores = contrib[1:] if len(contrib) > 1 else []
            pks = list(set([normalizar_palavra_chave(pk) for pk in meta.get('subject', []) if pk]))
            descricoes = meta.get('description', [])
            resumo = max(descricoes, key=len) if descricoes else ""
            
            # --- NOVO: Capturar a URL (dc.identifier.uri) ---
            identificadores = meta.get('identifier', [])
            # Procura na lista de identificadores aquele que começa com 'http'
            url_doc = next((link for link in identificadores if str(link).startswith('http')), "")
            
            dados_extraidos.append({
                'titulo': titulo, 
                'nivel_academico': nivel, 
                'autores': autores, 
                'orientador': orientador, 
                'co_orientadores': co_orientadores, 
                'possui_coorientador': len(co_orientadores) > 0, 
                'palavras_chave': pks, 
                'ano': ano_real, 
                'resumo': resumo, 
                'programa_origem': nome_prog,
                'url': url_doc  # --- NOVO CAMPO ADICIONADO ---
            })
        except Exception: continue
    return dados_extraidos

@st.cache_data
def calcular_sna_global(dados):
    G = nx.Graph()
    for d in dados:
        doc = d.get('titulo')
        if not doc: continue
        G.add_node(doc, tipo='Documento')
        for a in d.get('autores', []): G.add_node(a, tipo='Autor'); G.add_edge(doc, a)
        ori = d.get('orientador')
        if ori: G.add_node(ori, tipo='Orientador'); G.add_edge(doc, ori)
        for pk in d.get('palavras_chave', []): G.add_node(pk, tipo='Palavra-chave'); G.add_edge(doc, pk)

    deg_cent, bet_cent, grau_abs = nx.degree_centrality(G), nx.betweenness_centrality(G), dict(G.degree())
    try: mapa_comunidades = {node: i+1 for i, comm in enumerate(nx_comm.louvain_communities(G)) for node in comm}
    except: mapa_comunidades = {}
    rank_bet = {node: rank+1 for rank, (node, _) in enumerate(sorted(bet_cent.items(), key=lambda x: x[1], reverse=True))}

    return {node: {'Grau Absoluto': grau_abs.get(node, 0), 'Degree Centrality': deg_cent.get(node, 0), 'Betweenness': bet_cent.get(node, 0), 'Comunidade': mapa_comunidades.get(node, 'N/A'), 'Ranking Global': rank_bet.get(node, 'N/A')} for node in G.nodes()}

@st.cache_resource
def gerar_nodos_agraph(dados_recorte, termo_foco, grau_separacao=1):
    """Constrói os objetos Node e Edge para o visualizador avançado do streamlit-agraph."""
    G = nx.Graph()
    for tese in dados_recorte:
        doc_id = tese['titulo']
        G.add_node(doc_id, tipo='Documento', ano=tese.get('ano', 'N/A'), nivel=tese.get('nivel_academico', 'N/A'))
        for autor in tese.get('autores', []): G.add_node(autor, tipo='Autor'); G.add_edge(autor, doc_id)
        if tese.get('orientador'): G.add_node(tese['orientador'], tipo='Orientador'); G.add_edge(tese['orientador'], doc_id)
        for co in tese.get('co_orientadores', []): G.add_node(co, tipo='Co-orientador'); G.add_edge(co, doc_id)
        for pk in tese.get('palavras_chave', []): G.add_node(pk, tipo='Conceito'); G.add_edge(pk, doc_id)

    if termo_foco not in G.nodes(): return [], []
    
    # Recorta a órbita do nó
    ego_G = nx.ego_graph(G, termo_foco, radius=grau_separacao)
    graus = dict(ego_G.degree())

    nodes = []
    edges = []

    # 1. Construir os Nós (Nodes)
    for node, attrs in ego_G.nodes(data=True):
        tipo = attrs.get('tipo', 'Desconhecido')
        
        # Lógica de Tamanho e Cor
        tam = 40 if node == termo_foco else 15 + (graus[node] * 1.5)
        cor = '#FFFFFF' if node == termo_foco else ('#E74C3C' if tipo == 'Documento' else '#3498DB' if tipo == 'Autor' else '#F39C12' if tipo == 'Orientador' else '#2ECC71' if tipo == 'Conceito' else '#95A5A6')
        
        # Tooltip (Hover)
        hover = f"Tipo: {tipo}\nConexões locais: {graus[node]}"
        if tipo == 'Documento': hover += f"\nAno: {attrs.get('ano')}\nNível: {attrs.get('nivel')}"
        
        # Formas Geométricas (Agraph suporta dot, star, triangle, square, diamond, etc.)
        formato = 'diamond' if node == termo_foco else ('star' if tipo == 'Orientador' else 'square' if tipo == 'Documento' else 'triangle' if tipo == 'Conceito' else 'dot')
        
        # Rótulo inteligente (encurta títulos de teses muito longos para não poluir o visual)
        rotulo = node[:25] + "..." if len(node) > 25 and tipo == 'Documento' else node

        nodes.append(Node(id=node, label=rotulo, size=tam, color=cor, title=f"{node}\n{hover}", shape=formato))

    # 2. Construir as Arestas (Edges)
    for u, v in ego_G.edges():
        edges.append(Edge(source=u, target=v, color="#7F8C8D", width=1.5))

    return nodes, edges

# --- INTERFACE E EXTRAÇÃO ---
if 'dados_completos' not in st.session_state:
    st.title("🔌 Conexão Direta: Repositório Institucional UFSC")
    colecoes_disponiveis = carregar_catalogo_programas()
    
    if colecoes_disponiveis:
        st.markdown("### Selecione os Programas para Analisar")
        programas_selecionados = st.multiselect("Pode selecionar múltiplos Programas (PPGs):", list(colecoes_disponiveis.keys()))
        
        if programas_selecionados:
            for prog in programas_selecionados:
                set_spec = colecoes_disponiveis[prog]
                url = f"https://repositorio.ufsc.br/handle/{set_spec.split('_')[-1]}" if '_' in set_spec else "https://repositorio.ufsc.br/"
                st.info(f"**{prog}**\n\n🔗 [Aceder à página original na UFSC]({url}) | 🪪 ID Interno OAI: `{set_spec}`")
        
        with st.form("form_extracao"):
            btn_extrair = st.form_submit_button("Iniciar Extração ao Vivo", type="primary")
            
        status_box = st.empty()
        if btn_extrair and programas_selecionados:
            dados_agregados = []
            for prog in programas_selecionados:
                dados_agregados.extend(realizar_extracao(colecoes_disponiveis[prog], status_box, nome_prog=prog))
            
            if dados_agregados:
                status_box.success("✅ Extração Concluída!")
                st.session_state.update({'dados_completos': dados_agregados, 'nome_programa': prog if len(programas_selecionados)==1 else f"Análise Multidisciplinar ({len(programas_selecionados)} Programas)"})
                st.rerun() 
    st.stop()

# --- DASHBOARD PRINCIPAL ---
dados_completos = st.session_state['dados_completos']
st.title("🌌 Ecologia do Conhecimento")
st.subheader(f"Base: {st.session_state['nome_programa']}")

if st.sidebar.button("🔄 Nova Extração", type="primary"):
    del st.session_state['dados_completos']
    st.rerun()

# KPIs
autores_set = set([a for d in dados_completos for a in d.get('autores', [])])
orientadores_set = set([d.get('orientador') for d in dados_completos if d.get('orientador')])
coorientadores_set = set([co for d in dados_completos for co in d.get('co_orientadores', [])])
keywords_set = set([kw for d in dados_completos for kw in d.get('palavras_chave', [])])

c1, c2, c3 = st.columns(3)
c1.metric("📄 Documentos Totais", len(dados_completos))
c2.metric("🎓 Teses (Doutorado)", len([d for d in dados_completos if "Tese" in d.get('nivel_academico', '')]))
c3.metric("📜 Dissertações", len([d for d in dados_completos if "Disserta" in d.get('nivel_academico', '')]))

c4, c5, c6, c7 = st.columns(4)
c4.metric("✍️ Autores Únicos", len(autores_set))
c5.metric("🏫 Orientadores", len(orientadores_set))
c6.metric("🤝 Co-orientadores", len(coorientadores_set))
c7.metric("💡 Conceitos (Keywords)", len(keywords_set))

st.markdown("---")

# --- MOTOR DE BUSCA (EGO-GRAPH) ---
st.header("🔍 Motor de Busca e Dossiê (Search Engine)")

if 'busca_tipo' not in st.session_state: st.session_state.update({'busca_tipo': "Documento", 'busca_termo': None})
def navegar_para(novo_tipo, novo_termo): st.session_state.update({'busca_tipo': novo_tipo, 'busca_termo': novo_termo})

sna_global = calcular_sna_global(dados_completos)
tipo_busca = st.radio("Procurar:", ["Documento", "Autor", "Orientador", "Co-orientador", "Palavra-chave"], horizontal=True, key="busca_tipo")

opcoes = [d['titulo'] for d in dados_completos] if tipo_busca == "Documento" else list(autores_set) if tipo_busca == "Autor" else list(orientadores_set) if tipo_busca == "Orientador" else list(coorientadores_set) if tipo_busca == "Co-orientador" else list(keywords_set)

if st.session_state['busca_termo'] not in opcoes: st.session_state['busca_termo'] = None
termo_selecionado = st.selectbox("Selecione:", sorted(opcoes), index=sorted(opcoes).index(st.session_state['busca_termo']) if st.session_state['busca_termo'] in opcoes else None, placeholder="Pesquise aqui...")

if termo_selecionado != st.session_state['busca_termo']:
    st.session_state['busca_termo'] = termo_selecionado
    st.rerun()

termo_ativo = st.session_state['busca_termo']

if termo_ativo:
    col_info, col_sna = st.columns([2, 1])
    with col_info:
        st.info(f"**{termo_ativo}**")
        if tipo_busca == "Documento":
            doc = next((d for d in dados_completos if d['titulo'] == termo_ativo), {})
            
            # --- NOVO: Exibição dos Detalhes e URL ---
            st.write(f"**Ano:** {doc.get('ano', 'N/A')} | **Nível:** {doc.get('nivel_academico', 'N/A')} | **Programa:** {doc.get('programa_origem', 'N/A')}")
            
            if doc.get('url'):
                st.markdown(f"🔗 **Link Oficial na UFSC:** [{doc['url']}]({doc['url']})")
                st.markdown("<br>", unsafe_allow_html=True)
            
            # Botões interativos
            st.write("**Rede de Autoria e Orientação:**")
            for a in doc.get('autores', []): 
                st.button(f"👤 {a}", on_click=navegar_para, args=("Autor", a))
            if doc.get('orientador'): 
                st.button(f"🏫 {doc['orientador']}", on_click=navegar_para, args=("Orientador", doc['orientador']))
            for co in doc.get('co_orientadores', []):
                st.button(f"🤝 {co}", on_click=navegar_para, args=("Co-orientador", co))
                
            st.write("**Palavras-chave:**")
            for pk in doc.get('palavras_chave', []): 
                st.button(f"💡 {pk}", on_click=navegar_para, args=("Palavra-chave", pk))
                
            with st.expander("Ler Resumo (Abstract)"): 
                st.write(doc.get('resumo', 'Resumo não disponível.'))
        elif tipo_busca == "Autor":
            docs = [d for d in dados_completos if termo_ativo in d.get('autores', [])]
            for d in docs: st.button(f"📄 {d['titulo']}", on_click=navegar_para, args=("Documento", d['titulo']))
        elif tipo_busca == "Orientador":
            docs = [d for d in dados_completos if d.get('orientador') == termo_ativo]
            for d in docs: st.button(f"📄 {d['titulo']}", on_click=navegar_para, args=("Documento", d['titulo']))
        elif tipo_busca == "Palavra-chave":
            docs = [d for d in dados_completos if termo_ativo in d.get('palavras_chave', [])]
            for d in docs: st.button(f"📄 {d['titulo']}", on_click=navegar_para, args=("Documento", d['titulo']))

    with col_sna:
        metricas = sna_global.get(termo_ativo, {})
        if metricas:
            st.success(f"Cluster: {metricas.get('Comunidade')} | Rank: #{metricas.get('Ranking Global')}")
            st.metric("Grau (Conexões)", metricas.get('Grau Absoluto'))
            st.metric("Betweenness", f"{metricas.get('Betweenness', 0):.4f}")

    st.markdown("### 🌌 Órbita de Relacionamentos")
    grau_expansao = st.slider("Expansão do Grafo (Camadas de Profundidade):", 1, 3, 1)
    
    with st.spinner("A mapear o ecossistema local em 3D/2D..."):
        nodes, edges = gerar_nodos_agraph(dados_completos, termo_ativo, grau_expansao)
        
        if nodes and edges:
            # Configuração da física e do visual do motor Agraph
            config = Config(
                width="100%",
                height=600,
                directed=False, 
                physics=True, 
                hierarchical=False,
                nodeHighlightBehavior=True, # Faz brilhar os vizinhos ao passar o rato
                highlightColor="#F1C40F", # Amarelo vibrante no hover
                collapsible=False
            )
            
            # Renderiza o grafo nativo!
            retorno_clique = agraph(nodes=nodes, edges=edges, config=config)
            
            # Se o utilizador clicar num nó dentro do grafo, o Streamlit deteta o ID do nó clicado!
            if retorno_clique and retorno_clique != termo_ativo:
                st.info(f"💡 Você clicou no nó: **{retorno_clique}**. Pesquise por este nome na caixa acima para ver os detalhes completos!")
                
        else:
            st.warning("Não foi possível gerar a órbita visual para este termo.")
