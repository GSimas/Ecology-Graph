import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
import networkx.algorithms.community as nx_comm
from pyvis.network import Network
import json
import re
from collections import Counter
import itertools
import streamlit.components.v1 as components

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="Exploração Global | SNA", page_icon="🔭", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; }
    h1, h2, h3 { color: #F39C12; font-family: 'Helvetica Neue', sans-serif; }
    button[kind="primary"] { background-color: #2ECC71 !important; color: white !important; font-weight: bold !important; border: none !important; }
    </style>
""", unsafe_allow_html=True)

if 'dados_completos' not in st.session_state or not st.session_state['dados_completos']:
    st.warning("⚠️ Nenhuma base de dados carregada na memória.")
    st.info("Por favor, vá à página inicial e inicie a extração de um Programa de Pós-Graduação primeiro.")
    st.stop()

dados_completos = st.session_state['dados_completos']

# --- FUNÇÕES DE BACKEND MOVIDAS DO APP.PY ---
@st.cache_data
def obter_dataframe_metricas(dados_recorte):
    G = nx.Graph()
    for tese in dados_recorte:
        doc_id = tese['titulo']
        G.add_node(doc_id, tipo='Documento')
        for autor in tese.get('autores', []): G.add_node(autor, tipo='Autor'); G.add_edge(autor, doc_id)
        if tese.get('orientador'): G.add_node(tese['orientador'], tipo='Orientador'); G.add_edge(tese['orientador'], doc_id)
        for pk in tese.get('palavras_chave', []): G.add_node(pk, tipo='Conceito'); G.add_edge(doc_id, pk)

    degree_cent, betweenness_cent = nx.degree_centrality(G), nx.betweenness_centrality(G)
    return pd.DataFrame([{'Entidade (Nó)': n, 'Categoria': attrs.get('tipo', 'Outro'), 'Grau Absoluto': G.degree(n), 'Degree Centrality': degree_cent[n], 'Betweenness': betweenness_cent[n]} for n, attrs in G.nodes(data=True)])

@st.cache_resource
def gerar_html_pyvis(dados_recorte, metodo_cor, metodo_tamanho):
    G = nx.Graph()
    for tese in dados_recorte:
        doc_id = tese['titulo']
        G.add_node(doc_id, tipo='Documento', nivel=tese.get('nivel_academico', 'N/A'))
        if tese.get('orientador'): G.add_node(tese['orientador'], tipo='Orientador'); G.add_edge(tese['orientador'], doc_id)
        for pk in tese.get('palavras_chave', []): G.add_node(pk, tipo='Conceito'); G.add_edge(doc_id, pk)

    deg_cent, bet_cent, grau_abs = nx.degree_centrality(G), nx.betweenness_centrality(G), dict(G.degree())
    max_deg, max_bet, max_abs = max(deg_cent.values() or [1]), max(bet_cent.values() or [1]), max(grau_abs.values() or [1])

    legendas = []
    if metodo_cor != "Original (Categoria)":
        comunidades = nx_comm.louvain_communities(G) if metodo_cor == "Comunidades (Louvain)" else nx_comm.greedy_modularity_communities(G)
        paleta = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6']
        for i, comm in enumerate(comunidades):
            cor = paleta[i % len(paleta)]
            legendas.append({"id": i+1, "cor": cor, "tamanho": len(comm)})
            for n in comm: G.nodes[n]['color'] = cor

    for n, attrs in G.nodes(data=True):
        tipo = attrs.get('tipo', 'Desconhecido')
        tam = 10 + (grau_abs[n]/max_abs)*50 if metodo_tamanho == "Grau Absoluto" else (10 + (deg_cent[n]/max_deg)*50 if metodo_tamanho == "Degree Centrality" else (10 + (bet_cent[n]/max_bet)*50 if metodo_tamanho == "Betweenness" else 20))
        attrs.update({'size': tam, 'title': f"{n}\nGrau: {grau_abs[n]}"})
        if metodo_cor == "Original (Categoria)": attrs['color'] = '#E74C3C' if tipo == 'Documento' else '#F39C12' if tipo == 'Orientador' else '#2ECC71'

    net = Network(height='600px', width='100%', bgcolor='#222222', font_color='white', directed=False, cdn_resources='remote')
    net.from_nx(G)
    net.set_options('{"interaction": {"hover": true, "selectConnectedEdges": true}}')
    path = "temp_pyvis_global.html"
    net.save_graph(path)
    return path, G.number_of_nodes(), G.number_of_edges(), legendas

@st.cache_resource
def gerar_html_coocorrencia(dados_recorte, min_coocorrencia=1):
    G = nx.Graph()
    for d in dados_recorte:
        pks = d.get('palavras_chave', [])
        for pk in pks: G.nodes[pk]['count'] = G.nodes.get(pk, {}).get('count', 0) + 1; G.add_node(pk, tipo='Conceito') if not G.has_node(pk) else None
        for pk1, pk2 in itertools.combinations(pks, 2):
            if G.has_edge(pk1, pk2): G[pk1][pk2]['weight'] += 1
            else: G.add_edge(pk1, pk2, weight=1)

    G.remove_edges_from([(u, v) for u, v, attrs in G.edges(data=True) if attrs['weight'] < min_coocorrencia])
    G.remove_nodes_from(list(nx.isolates(G)))

    for n, attrs in G.nodes(data=True): attrs.update({'shape': 'dot', 'size': min(10 + (attrs['count'] * 1.5), 60), 'color': '#2ECC71', 'title': f"{n}\nOcorrências: {attrs['count']}"})
    for u, v, attrs in G.edges(data=True): attrs.update({'value': attrs['weight'], 'title': f"Co-ocorrências: {attrs['weight']}", 'color': 'rgba(255, 255, 255, 0.2)'})

    net = Network(height='600px', width='100%', bgcolor='#222222', font_color='white', cdn_resources='remote')
    net.from_nx(G)
    path = "grafo_coocorrencia.html"
    net.save_graph(path)
    return path, G.number_of_nodes(), G.number_of_edges()

def preparar_csv_exportacao(dados):
    df = pd.DataFrame(dados)
    for col in ['autores', 'co_orientadores', 'palavras_chave']:
        if col in df.columns: df[col] = df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    return df.to_csv(index=False).encode('utf-8')

# --- INTERFACE ---
st.title("🔭 Exploração Global da Rede")
st.subheader(f"Base de Dados: {st.session_state.get('nome_programa', '')}")

# SEÇÃO 1: GRAFO GERAL
st.header("🕸️ 1. Topologia da Rede")
with st.form("form_grafo"):
    col_g1, col_g2 = st.columns(2)
    n_registros = col_g1.slider("Volume de Documentos:", 1, len(dados_completos), min(40, len(dados_completos)))
    metodo_coloracao = col_g1.selectbox("Cores e Comunidades:", ["Original (Categoria)", "Comunidades (Louvain)", "Comunidades (Greedy Modularity)"])
    metodo_tamanho = col_g2.selectbox("Tamanho dos Nós (SNA):", ["Tamanho Fixo", "Grau Absoluto", "Degree Centrality", "Betweenness"])
    if st.form_submit_button("Renderizar Grafo Global", type="primary"):
        with st.spinner("Construindo rede..."):
            path, nos, arestas, legendas = gerar_html_pyvis(dados_completos[:n_registros], metodo_coloracao, metodo_tamanho)
            with open(path, 'r', encoding='utf-8') as f: components.html(f.read(), height=650)

st.markdown("---")

# SEÇÃO 2: RANKING SNA
st.header("🏆 2. Ranking de Autoridade (SNA)")
with st.form("form_tabela"):
    col_t1, col_t2 = st.columns([3, 1])
    n_registros_tab = col_t1.slider("Volume para cálculo:", 1, len(dados_completos), len(dados_completos))
    top_x = col_t2.number_input("Top X:", 1, 500, 20)
    cat_sel = st.multiselect("Categorias:", ["Documento", "Autor", "Orientador", "Conceito"], default=["Orientador", "Conceito"])
    met_sel = st.multiselect("Métricas:", ["Grau Absoluto", "Degree Centrality", "Betweenness"], default=["Betweenness"])
    if st.form_submit_button("Gerar Ranking", type="primary") and met_sel and cat_sel:
        df_completo = obter_dataframe_metricas(dados_completos[:n_registros_tab])
        df_top = df_completo[df_completo['Categoria'].isin(cat_sel)].sort_values(by=met_sel[0], ascending=False).head(top_x)
        st.dataframe(df_top, use_container_width=True, hide_index=True)

st.markdown("---")

# SEÇÃO 3: EVOLUÇÃO
st.header("📈 3. Evolução Histórica")
df_hist = pd.DataFrame(dados_completos)
df_hist['Ano'] = pd.to_numeric(df_hist.get('ano'), errors='coerce')
df_plot = df_hist.dropna(subset=['Ano']).groupby('Ano').size().reset_index(name='Volume')
st.plotly_chart(px.line(df_plot, x='Ano', y='Volume', markers=True, title="Total de Publicações por Ano", template="plotly_dark"), use_container_width=True)

st.markdown("---")

# SEÇÃO 4: CO-OCORRÊNCIA
st.header("🔗 4. Grafo de Co-ocorrência Temática")
min_peso = st.slider("Conexões que ocorrem juntas pelo menos X vezes:", 1, 20, 2)
if st.button("Mapear Co-ocorrências", type="primary"):
    with st.spinner("Mapeando..."):
        path_co, nos_co, arestas_co = gerar_html_coocorrencia(dados_completos, min_peso)
        with open(path_co, 'r', encoding='utf-8') as f: components.html(f.read(), height=650)

st.markdown("---")

# SEÇÃO 5: EXPORTAÇÃO
st.header("📥 5. Exportação da Base")
c_b1, c_b2 = st.columns(2)
c_b1.download_button("📄 Baixar JSON", data=json.dumps(dados_completos, ensure_ascii=False, indent=4), file_name="dados.json", mime="application/json")
c_b2.download_button("📊 Baixar CSV", data=preparar_csv_exportacao(dados_completos), file_name="dados.csv", mime="text/csv")
