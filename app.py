import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import networkx.algorithms.community as nx_comm
from pyvis.network import Network
import pandas as pd
import json

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Ecologia do Conhecimento UFSC",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização customizada (CSS)
st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; }
    h1, h2, h3 { color: #F39C12; font-family: 'Helvetica Neue', sans-serif; }
    .stMetric { background-color: #2C3E50; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.5); }
    </style>
""", unsafe_allow_html=True)

# --- INICIALIZAÇÃO DE ESTADO ---
if 'grafo_pronto' not in st.session_state:
    st.session_state['grafo_pronto'] = False
if 'tabela_pronta' not in st.session_state:
    st.session_state['tabela_pronta'] = False

# --- FUNÇÕES DE BACK-END INDEPENDENTES ---
@st.cache_data
def carregar_dados_locais():
    try:
        with open('base_ppgegc.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Arquivo base_ppgegc.json não encontrado no repositório.")
        return []

@st.cache_data(show_spinner="A extrair matrizes matemáticas...")
def obter_dataframe_metricas(dados_recorte):
    G = nx.Graph()
    for tese in dados_recorte:
        doc_id = tese['titulo']
        G.add_node(doc_id, tipo='Documento')
        for autor in tese['autores']:
            G.add_node(autor, tipo='Autor')
            G.add_edge(autor, doc_id)
        if tese.get('orientador'):
            G.add_node(tese['orientador'], tipo='Orientador')
            G.add_edge(tese['orientador'], doc_id)
        for pk in tese.get('palavras_chave', []):
            G.add_node(pk, tipo='Conceito')
            G.add_edge(doc_id, pk)

    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G)
    
    lista = []
    for node, attrs in G.nodes(data=True):
        lista.append({
            'Entidade (Nó)': node,
            'Categoria': attrs.get('tipo', 'Desconhecido'),
            'Grau Absoluto': G.degree(node),
            'Degree Centrality': degree_cent[node],
            'Betweenness': betweenness_cent[node]
        })
    return pd.DataFrame(lista)

@st.cache_resource(show_spinner="A calcular layout visual e detetar comunidades...")
def gerar_html_pyvis(dados_recorte, metodo_cor="Original (Categoria)"):
    G = nx.Graph()
    
    # 1. Estruturação Básica
    for tese in dados_recorte:
        doc_id = tese['titulo']
        G.add_node(doc_id, tipo='Documento', ano=tese['ano'])
        for autor in tese['autores']:
            G.add_node(autor, tipo='Autor')
            G.add_edge(autor, doc_id)
        if tese.get('orientador'):
            G.add_node(tese['orientador'], tipo='Orientador')
            G.add_edge(tese['orientador'], doc_id)
        for pk in tese.get('palavras_chave', []):
            G.add_node(pk, tipo='Conceito')
            G.add_edge(doc_id, pk)

    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G)

    # 2. Correção dos Tooltips (Remoção do HTML e uso de \n)
    for node, attrs in G.nodes(data=True):
        tipo = attrs.get('tipo', 'Desconhecido')
        
        # Formatando o texto de forma limpa, sem tags HTML
        janela_sna = f"\n\n--- Métricas SNA ---\nGrau: {G.degree(node)}\nDegree Centrality: {degree_cent[node]:.4f}\nBetweenness: {betweenness_cent[node]:.4f}"
        
        # Adiciona os formatos geométricos
        if tipo == 'Documento':
            attrs.update({'shape': 'square', 'size': 30, 'title': f"Tese:\n{node}\nAno: {attrs.get('ano')}{janela_sna}"})
        elif tipo == 'Autor':
            attrs.update({'shape': 'dot', 'size': 20, 'title': f"Autor:\n{node}{janela_sna}"})
        elif tipo == 'Orientador':
            attrs.update({'shape': 'star', 'size': 25, 'title': f"Orientador:\n{node}{janela_sna}"})
        elif tipo == 'Conceito':
            attrs.update({'shape': 'triangle', 'size': 15, 'title': f"Conceito:\n{node}{janela_sna}"})

    # 3. Lógica de Coloração (Comunidades vs Original)
    if metodo_cor == "Original (Categoria)":
        for node, attrs in G.nodes(data=True):
            tipo = attrs.get('tipo', 'Desconhecido')
            if tipo == 'Documento': attrs['color'] = '#E74C3C'
            elif tipo == 'Autor': attrs['color'] = '#3498DB'
            elif tipo == 'Orientador': attrs['color'] = '#F39C12'
            elif tipo == 'Conceito': attrs['color'] = '#2ECC71'
    else:
        # Algoritmos de Comunidade
        comunidades = []
        if metodo_cor == "Comunidades (Louvain)":
            comunidades = nx_comm.louvain_communities(G)
        elif metodo_cor == "Comunidades (Greedy Modularity)":
            comunidades = nx_comm.greedy_modularity_communities(G)
        elif metodo_cor == "Comunidades (Girvan-Newman)":
            try:
                # Pega a primeira divisão iterativa para evitar travamentos
                gerador_gn = nx_comm.girvan_newman(G)
                comunidades = next(gerador_gn)
            except StopIteration:
                comunidades = [set(G.nodes())]

        # Paleta de cores vibrantes para identificar as diferentes bolhas/comunidades
        paleta = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080']
        
        for i, comm in enumerate(comunidades):
            cor_comunidade = paleta[i % len(paleta)]
            for node in comm:
                G.nodes[node]['color'] = cor_comunidade

    # 4. Configuração Pyvis
    net = Network(height='600px', width='100%', bgcolor='#222222', font_color='white', select_menu=True, filter_menu=True, cdn_resources='remote')
    net.from_nx(G)
    net.set_options('{"physics": {"barnesHut": {"gravitationalConstant": -15000, "springLength": 150}, "stabilization": {"enabled": true, "iterations": 150}}, "interaction": {"hover": true, "navigationButtons": true, "tooltipDelay": 100}}')
    
    path = "grafo_temp.html"
    net.save_graph(path)
    
    # 5. Injeção JavaScript para ocultar nós ao clicar
    script_ocultar = """
    network.on("selectNode", function (params) {
        if (params.nodes.length === 1) {
            var nodeId = params.nodes[0];
            var connectedNodes = network.getConnectedNodes(nodeId);
            var updates = nodes.get().map(n => ({id: n.id, hidden: !(n.id === nodeId || connectedNodes.includes(n.id))}));
            nodes.update(updates);
        }
    });
    network.on("deselectNode", function () {
        nodes.update(nodes.get().map(n => ({id: n.id, hidden: false})));
    });
    """
    with open(path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html_content.replace('return network;', script_ocultar + '\n\treturn network;'))
        
    return path, G.number_of_nodes(), G.number_of_edges()

# --- CONSTRUÇÃO DA INTERFACE FRONT-END ---

st.title("🌌 Ecologia do Conhecimento: PPGEGC UFSC")

st.markdown("""
> Esta plataforma é um instrumento de Ecologia do Conhecimento aplicada à bibliometria. Ela mapeia e quantifica as intrincadas relações entre teses, autores, orientadores e conceitos do Programa de Pós-Graduação através da Ciência de Redes. O seu propósito é revelar a genealogia acadêmica, identificar *hubs* de influência e expor as pontes interdisciplinares que estruturam a produção científica, traduzindo repositórios institucionais em uma topologia visual navegável e em métricas rigorosas de centralidade matemática.
""")
st.markdown("---")

dados_completos = carregar_dados_locais()
total_documentos = len(dados_completos) if len(dados_completos) > 0 else 100

# --- SEÇÃO 1: GRAFO INTERATIVO ---
st.header("🕸️ Topologia e Grafo Interativo")

with st.form("form_grafo"):
    col_g1, col_g2, col_g3 = st.columns([2, 2, 1])
    with col_g1:
        n_registros_grafo = st.slider("Documentos para a Rede Visual:", 5, total_documentos, 40, 5)
    with col_g2:
        metodo_coloracao = st.selectbox(
            "Mapeamento de Cores da Rede:", 
            ["Original (Categoria)", "Comunidades (Louvain)", "Comunidades (Greedy Modularity)", "Comunidades (Girvan-Newman)"]
        )
    with col_g3:
        st.markdown("<br><br>", unsafe_allow_html=True)
        btn_render_grafo = st.form_submit_button("Renderizar Grafo", use_container_width=True)

if btn_render_grafo:
    path, nos, arestas = gerar_html_pyvis(dados_completos[:n_registros_grafo], metodo_cor=metodo_coloracao)
    st.session_state['path_grafo'] = path
    st.session_state['kpis_grafo'] = {'nos': nos, 'arestas': arestas}
    st.session_state['grafo_pronto'] = True

if st.session_state['grafo_pronto']:
    kpis = st.session_state['kpis_grafo']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nós Renderizados", kpis['nos'])
    c2.metric("Arestas Renderizadas", kpis['arestas'])
    c3.metric("Densidade", f"{(kpis['arestas'] / kpis['nos']):.3f}" if kpis['nos'] > 0 else 0)
    c4.info("Dica: Clique em um nó para isolar o seu ecossistema.")
    
    with open(st.session_state['path_grafo'], 'r', encoding='utf-8') as f:
        components.html(f.read(), height=650, scrolling=False)

st.markdown("---")

# --- SEÇÃO 2: ANÁLISE ESTRUTURAL (RANKING) ---
st.header("🏆 Análise Estrutural e Rankings (SNA)")

with st.form("form_tabela"):
    st.write("Configurações do Ranking (Independente do Grafo Visual):")
    
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        n_registros_tabela = st.slider("Documentos analisados matematicamente:", 5, total_documentos, total_documentos, 5)
    with col_t2:
        top_x = st.number_input("Tamanho do Ranking (Top X):", min_value=1, max_value=1000, value=20, step=5)

    col_t3, col_t4, col_t5 = st.columns(3)
    categorias_disponiveis = ["Documento", "Autor", "Orientador", "Conceito"]
    todas_metricas = ["Grau Absoluto", "Degree Centrality", "Betweenness"]
    
    with col_t3:
        cat_sel = st.multiselect("Categorias a incluir:", categorias_disponiveis, default=["Orientador", "Conceito"])
    with col_t4:
        met_sel = st.multiselect("Métricas a exibir:", todas_metricas, default=["Grau Absoluto", "Betweenness"])
    with col_t5:
        met_ord = st.selectbox("Ordenar o Ranking primariamente por:", met_sel if met_sel else todas_metricas)
        
    btn_render_tabela = st.form_submit_button("Processar e Atualizar Tabela", type="primary")

if btn_render_tabela:
    if not met_sel:
        st.warning("Selecione pelo menos uma métrica para a tabela.")
    elif not cat_sel:
        st.warning("Selecione pelo menos uma categoria.")
    else:
        df_completo = obter_dataframe_metricas(dados_completos[:n_registros_tabela])
        
        df_filtrado = df_completo[df_completo['Categoria'].isin(cat_sel)]
        df_top_x = df_filtrado.sort_values(by=met_ord, ascending=False).head(top_x)
        
        st.session_state['df_top_x'] = df_top_x
        st.session_state['colunas_finais'] = ['Entidade (Nó)', 'Categoria'] + met_sel
        st.session_state['met_ord'] = met_ord
        st.session_state['tabela_pronta'] = True

if st.session_state['tabela_pronta']:
    df_exibicao = st.session_state['df_top_x'].copy()
    colunas = st.session_state['colunas_finais']
    
    if 'Degree Centrality' in df_exibicao.columns:
        df_exibicao['Degree Centrality'] = df_exibicao['Degree Centrality'].apply(lambda x: f"{x:.4f}")
    if 'Betweenness' in df_exibicao.columns:
        df_exibicao['Betweenness'] = df_exibicao['Betweenness'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(df_exibicao[colunas], use_container_width=True, hide_index=True)
    
    csv_data = st.session_state['df_top_x'][colunas].to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Exportar Ranking (CSV) - Ordenado por {st.session_state['met_ord']}",
        data=csv_data,
        file_name=f"ranking_sna_{st.session_state['met_ord'].lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )
