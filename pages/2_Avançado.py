import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
import json
import streamlit.components.v1 as components

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Análises Avançadas | Universal",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização Dark Mode
st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; }
    h1, h2, h3, h4 { color: #F39C12; font-family: 'Helvetica Neue', sans-serif; }
    .stMetric { background-color: #2C3E50; padding: 15px; border-radius: 10px; }
    button[kind="primary"] { background-color: #2ECC71 !important; color: white !important; font-weight: bold !important; border: none !important; }
    </style>
""", unsafe_allow_html=True)

# --- PROTEÇÃO E LEITURA DA MEMÓRIA VIVA (SESSION STATE) ---
if 'dados_completos' not in st.session_state or not st.session_state['dados_completos']:
    st.warning("⚠️ Nenhuma base de dados carregada na memória.")
    st.info("Por favor, vá à página inicial e inicie a extração de um Programa de Pós-Graduação antes de usar as análises avançadas.")
    st.stop()

dados_gerais = st.session_state['dados_completos']
nome_programa = st.session_state.get('nome_programa', 'Programa Desconhecido')

# --- FUNÇÕES DE BACK-END ---

@st.cache_data
def preparar_dataframe(dados_lista):
    df = pd.DataFrame(dados_lista)
    df['Ano'] = pd.to_numeric(df.get('ano'), errors='coerce')
    df = df.dropna(subset=['Ano'])
    df['Ano'] = df['Ano'].astype(int)
    df['nivel_academico'] = df.get('nivel_academico', 'Outros').fillna('Outros')
    return df

# --- LÓGICA 1: BURST DETECTION ---
@st.cache_data
def detetar_explosoes(df_base, min_freq, z_score):
    if df_base.empty or 'palavras_chave' not in df_base.columns: return pd.DataFrame()
    df_f = df_base.explode('palavras_chave').groupby(['Ano', 'palavras_chave']).size().reset_index(name='Frequencia')
    if df_f.empty: return pd.DataFrame()
    
    contagem = df_f.groupby('palavras_chave')['Frequencia'].sum()
    validos = contagem[contagem >= min_freq].index
    df_f = df_f[df_f['palavras_chave'].isin(validos)]
    
    if df_f.empty: return pd.DataFrame()
    
    anos = range(df_f['Ano'].min(), df_f['Ano'].max() + 1)
    grelha = pd.MultiIndex.from_product([anos, validos], names=['Ano', 'palavras_chave']).to_frame(index=False)
    df_c = pd.merge(grelha, df_f, on=['Ano', 'palavras_chave'], how='left').fillna(0).sort_values(['palavras_chave', 'Ano'])
    
    df_c['Media'] = df_c.groupby('palavras_chave')['Frequencia'].transform(lambda x: x.expanding().mean().shift(1).fillna(0))
    df_c['Std'] = df_c.groupby('palavras_chave')['Frequencia'].transform(lambda x: x.expanding().std().shift(1).fillna(0))
    df_c['Em_Explosao'] = (df_c['Frequencia'] > (df_c['Media'] + (z_score * df_c['Std']))) & (df_c['Frequencia'] >= 2)
    return df_c

# --- LÓGICA 2: GENEALOGIA ---
@st.cache_resource
def gerar_grafo_genealogico(dados_lista, orientadores_foco):
    G = nx.DiGraph()
    for d in dados_lista:
        ori = d.get('orientador')
        autores = d.get('autores', [])
        if ori:
            for autor in autores:
                G.add_edge(ori, autor, label=f"{d.get('nivel_academico', '')} ({d.get('ano', '')})")

    if orientadores_foco:
        nós_desc = set()
        for o in orientadores_foco:
            if G.has_node(o):
                nós_desc.update(nx.descendants(G, o))
                nós_desc.add(o)
        G = G.subgraph(nós_desc).copy()

    for node in G.nodes():
        tem_pupilos = G.out_degree(node) > 0
        foi_pupilo = G.in_degree(node) > 0
        if tem_pupilos and foi_pupilo: color, label = '#F39C12', f"🎓 {node}"
        elif tem_pupilos: color, label = '#E74C3C', f"🏛️ {node}"
        else: color, label = '#3498DB', node
        G.nodes[node].update({'color': color, 'label': label, 'size': 25 if tem_pupilos else 15})

    net = Network(height='700px', width='100%', bgcolor='#222222', font_color='white', directed=True, cdn_resources='remote')
    net.from_nx(G)
    net.set_options('{"layout": {"hierarchical": {"enabled": true, "direction": "UD", "sortMethod": "directed"}}, "physics": {"enabled": false}}')
    path = "temp_gen.html"
    net.save_graph(path)
    return path, G.number_of_nodes(), G.number_of_edges()

# --- LÓGICA 3: BURT'S CONSTRAINT ---
@st.cache_data
def calcular_burt(dados_lista):
    G = nx.Graph()
    for d in dados_lista:
        ori = d.get('orientador')
        if ori:
            G.add_node(ori, tipo='Orientador')
            for pk in d.get('palavras_chave', []):
                G.add_node(pk, tipo='Conceito')
                G.add_edge(ori, pk)
    constraint = nx.constraint(G)
    betweenness = nx.betweenness_centrality(G)
    degree = dict(G.degree())
    resumo = []
    for node in G.nodes():
        if G.nodes[node].get('tipo') == 'Orientador':
            resumo.append({
                'Orientador': node,
                'Restrição (Constraint)': constraint.get(node, 0),
                'Intermediação (Betweenness)': betweenness.get(node, 0),
                'Diversidade': degree.get(node, 0)
            })
    return pd.DataFrame(resumo)

# --- LÓGICA 4: SANKEY ---
@st.cache_data
def preparar_sankey(dados_lista, top_n=10):
    df = pd.DataFrame(dados_lista)
    if 'orientador' not in df.columns: return [], [], [], []
    top_orient = df['orientador'].value_counts().head(top_n).index.tolist()
    df_f = df[df['orientador'].isin(top_orient)]
    fluxos = []
    for _, row in df_f.iterrows():
        ori, niv = row.get('orientador'), row.get('nivel_academico', 'Outros')
        if not ori or not pd.notna(ori): continue
        fluxos.append({'src': ori, 'tgt': niv, 'val': 1})
        for pk in row.get('palavras_chave', [])[:2]:
            if pk: fluxos.append({'src': niv, 'tgt': pk, 'val': 1})
            
    if not fluxos: return [], [], [], []
    df_fluxos = pd.DataFrame(fluxos).groupby(['src', 'tgt']).sum().reset_index()
    nodes = list(set(df_fluxos['src']).union(set(df_fluxos['tgt'])))
    mapping = {name: i for i, name in enumerate(nodes)}
    return nodes, df_fluxos['src'].map(mapping), df_fluxos['tgt'].map(mapping), df_fluxos['val']

# --- INTERFACE ---

st.title(f"🧪 Ecologia do Conhecimento: Análises Profundas")
st.subheader(f"Explorando o DNA de: {nome_programa}")
df_geral = preparar_dataframe(dados_gerais)

tab1, tab2, tab3, tab4 = st.tabs(["🔥 Burst", "🌳 Genealogia", "🕳️ Furos (Burt)", "🌊 Fluxos (Sankey)"])

with tab1:
    st.markdown("### Deteção de Explosões (Emergências)")
    with st.form("f_burst"):
        c1, c2, c3 = st.columns(3)
        min_f = c1.number_input("Freq. Mínima (Filtro Ruído):", 2, 50, 5)
        z_s = c2.slider("Sensibilidade (Z-Score):", 1.0, 3.0, 1.5)
        btn_b = st.form_submit_button("Analisar")
    if btn_b:
        df_b = detetar_explosoes(df_geral, min_f, z_s)
        if not df_b.empty and df_b['Em_Explosao'].any():
            st.plotly_chart(px.scatter(df_b[df_b['Em_Explosao']], x="Ano", y="palavras_chave", size="Frequencia", color="palavras_chave", template="plotly_dark", title="Rupturas Epistemológicas"))
        else: st.info("Sem explosões detetadas com os parâmetros atuais.")

with tab2:
    st.markdown("### DNA Académico (Linhagens)")
    todos_ori = sorted(list(set([d.get('orientador') for d in dados_gerais if d.get('orientador')])))
    ori_foco = st.multiselect("Filtrar Dinastia (Patriarcas/Matriarcas):", todos_ori)
    if st.button("Mapear Árvore de Descendência", type="primary"):
        p, n, e = gerar_grafo_genealogico(dados_gerais, ori_foco)
        st.write(f"**Tamanho da Dinastia:** {n} membros conectados através de {e} vínculos de orientação.")
        with open(p, 'r', encoding='utf-8') as f: components.html(f.read(), height=700)

with tab3:
    st.markdown("### Métrica de Burt (Brokers vs Bolhas)")
    df_bt = calcular_burt(dados_gerais)
    if not df_bt.empty:
        st.plotly_chart(px.scatter(df_bt, x="Diversidade", y="Restrição (Constraint)", size="Intermediação (Betweenness)", hover_name="Orientador", color="Restrição (Constraint)", color_continuous_scale="RdYlGn_r", template="plotly_dark", title="Mapa de Furos Estruturais"))
    else: st.warning("Dados insuficientes para calcular a métrica de restrição.")

with tab4:
    st.markdown("### Fluxos de Energia Epistémica (Sankey)")
    n_s = st.slider("Qtd Orientadores Principais:", 5, 30, 10)
    nodes, src, tgt, val = preparar_sankey(dados_gerais, n_s)
    if nodes:
        fig = go.Figure(go.Sankey(node=dict(pad=15, thickness=20, label=nodes, color="#F39C12"), link=dict(source=src, target=tgt, value=val, color="rgba(243, 156, 18, 0.4)")))
        fig.update_layout(template="plotly_dark", height=700)
        st.plotly_chart(fig, use_container_width=True)
    else: st.warning("Não há vínculos suficientes entre orientadores e conceitos.")
