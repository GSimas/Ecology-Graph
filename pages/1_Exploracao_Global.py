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
from streamlit_agraph import agraph, Node, Edge, Config
import numpy as np
import scipy as sp
import io


# --- INICIALIZAÇÃO DEFENSIVA DE ESTADO ---
chaves_necessarias = {
    'grafo_pronto': False,
    'tabela_pronta': False,
    'coocorrencia_pronta': False,
    'kpis_grafo': {'nos': 0, 'arestas': 0, 'legendas': []},
    'kpis_co': {'nos': 0, 'arestas': 0},
    'graf_glob_nodes': [],
    'graf_glob_edges': [],
    'co_nodes': [],
    'co_edges': []
}

for chave, valor_padrao in chaves_necessarias.items():
    if chave not in st.session_state:
        st.session_state[chave] = valor_padrao



# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Exploração Global | SNA",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização
st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; }
    h1, h2, h3, h4 { color: #F39C12; font-family: 'Helvetica Neue', sans-serif; }
    button[kind="primary"] { background-color: #2ECC71 !important; color: white !important; font-weight: bold !important; border: none !important; }
    div[data-testid="stMetricValue"] { color: #F39C12 !important; }
    </style>
""", unsafe_allow_html=True)

# --- PROTEÇÃO DE ESTADO ---
if 'dados_completos' not in st.session_state or not st.session_state['dados_completos']:
    st.warning("⚠️ Nenhuma base de dados carregada na memória.")
    st.info("Por favor, vá à página inicial e inicie a extração de um Programa de Pós-Graduação antes de usar a exploração global.")
    st.stop()

dados_completos = st.session_state['dados_completos']

# Inicialização de variáveis de controle da interface
for key in ['grafo_pronto', 'tabela_pronta', 'coocorrencia_pronta']:
    if key not in st.session_state:
        st.session_state[key] = False

# --- PREPARAÇÃO DE VARIÁVEIS GLOBAIS PARA OS FILTROS ---
niveis_disponiveis = sorted(list(set([d.get('nivel_academico', 'Não Classificado') for d in dados_completos])))
orientadores_disponiveis = sorted(list(set([d.get('orientador', 'Não informado') for d in dados_completos if d.get('orientador')])))
lista_todos_conceitos = []
for d in dados_completos: 
    lista_todos_conceitos.extend(d.get('palavras_chave', []))
conceitos_unicos = sorted(list(set(lista_todos_conceitos)))
anos_disponiveis = [int(d.get('ano')) for d in dados_completos if d.get('ano') and str(d.get('ano')).isdigit()]
min_ano_global = min(anos_disponiveis) if anos_disponiveis else 2000
max_ano_global = max(anos_disponiveis) if anos_disponiveis else 2026

# --- FUNÇÕES DE BACK-END ---
@st.cache_data
def preparar_dados_base_df(dados):
    df = pd.DataFrame(dados)
    df['Ano'] = pd.to_numeric(df.get('ano'), errors='coerce')
    df = df.dropna(subset=['Ano'])
    df['Ano'] = df['Ano'].astype(int)
    df['nivel_academico'] = df.get('nivel_academico', 'Outros').fillna('Outros')
    df['titulo'] = df.get('titulo', '').fillna('')
    df['orientador'] = df.get('orientador', 'Não informado').fillna('Não informado')
    return df

@st.cache_data
def obter_dataframe_metricas(dados_recorte):
    G = nx.Graph()
    for tese in dados_recorte:
        doc_id = tese['titulo']
        G.add_node(doc_id, tipo='Documento')
        for autor in tese.get('autores', []):
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
            'Degree Centrality': degree_cent.get(node, 0),
            'Betweenness': betweenness_cent.get(node, 0)
        })
    return pd.DataFrame(lista)

@st.cache_resource
def gerar_nodos_globais_agraph(dados_recorte, metodo_cor="Original (Categoria)", metodo_tamanho="Tamanho Fixo"):
    G = nx.Graph()
    for tese in dados_recorte:
        doc_id = tese['titulo']
        # Adicionamos atributos extras para que o Gephi já reconheça as categorias
        G.add_node(doc_id, label=doc_id[:30], tipo='Documento', nivel=tese.get('nivel_academico', 'N/A'), ano=tese.get('ano', 'N/A'))
        
        if tese.get('orientador'): 
            G.add_node(tese['orientador'], label=tese['orientador'], tipo='Orientador')
            G.add_edge(tese['orientador'], doc_id)
            
        for pk in tese.get('palavras_chave', []): 
            G.add_node(pk, label=pk, tipo='Conceito')
            G.add_edge(doc_id, pk)

    deg_cent, bet_cent, grau_abs = nx.degree_centrality(G), nx.betweenness_centrality(G), dict(G.degree())
    max_deg, max_bet, max_abs = max(deg_cent.values() or [1]), max(bet_cent.values() or [1]), max(grau_abs.values() or [1])

    # Cálculo de Comunidades para exportação
    comunidades = nx_comm.louvain_communities(G)
    legendas_comunidades = []
    paleta = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6']
    
    for i, comm in enumerate(comunidades):
        cor_com = paleta[i % len(paleta)]
        legendas_comunidades.append({"id": i+1, "cor": cor_com, "tamanho": len(comm)})
        for node in comm: 
            G.nodes[node]['color'] = cor_com
            G.nodes[node]['community'] = i + 1 # Atributo para o Gephi

    nodes_agraph, edges_agraph = [], []
    for node, attrs in G.nodes(data=True):
        tipo = attrs.get('tipo', 'Desconhecido')
        grau_atual = grau_abs.get(node, 0)
        
        # Lógica de Tamanho Agraph
        if metodo_tamanho == "Grau Absoluto": tam = 10 + (grau_atual / max_abs) * 40
        elif metodo_tamanho == "Degree Centrality": tam = 10 + (deg_cent.get(node, 0) / max_deg) * 40
        elif metodo_tamanho == "Betweenness": tam = 10 + (bet_cent.get(node, 0) / max_bet) * 40
        else: tam = 20
        
        cor_final = attrs.get('color', ('#E74C3C' if tipo == 'Documento' else '#F39C12' if tipo == 'Orientador' else '#2ECC71'))
        formato = 'star' if tipo == 'Orientador' else 'square' if tipo == 'Documento' else 'dot'
        
        nodes_agraph.append(Node(id=node, label=attrs['label'], size=tam, color=cor_final, shape=formato, title=f"{node}\nTipo: {tipo}"))

    for u, v in G.edges():
        edges_agraph.append(Edge(source=u, target=v, color="#7F8C8D", width=0.5))

    return nodes_agraph, edges_agraph, legendas_comunidades, G # <-- Retornamos o G original aqui

def obter_frequencias_texto(df_hist, fonte_nuvem):
    if fonte_nuvem == "Conceitos (Palavras-chave)":
        lista_c = []
        for lst in df_hist['palavras_chave']: lista_c.extend(lst)
        return dict(Counter(lista_c).most_common(100))
    else:
        textos = df_hist.get('resumo', pd.Series()).dropna().astype(str).tolist() if fonte_nuvem == "Resumos (Abstracts)" else df_hist['titulo'].dropna().astype(str).tolist()
        texto_completo = " ".join(textos).lower()
        texto_completo = re.sub(r'[^\w\s]', '', texto_completo)
        palavras = texto_completo.split()
        stopwords_pt = set(['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'uma', 'para', 'com', 'não', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'das', 'à', 'seu', 'sua', 'ou', 'nos', 'já', 'eu', 'também', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'sem', 'mesmo', 'aos', 'nas', 'me', 'esse', 'essa', 'num', 'nem', 'numa', 'pelos', 'pelas', 'este', 'esta', 'sobre', 'estudo', 'análise', 'proposta', 'uso', 'aplicação', 'desenvolvimento', 'modelo', 'sistema', 'avaliação', 'gestão', 'conhecimento', 'engenharia', 'objetivo', 'pesquisa', 'trabalho', 'resultados', 'método', 'foi', 'foram', 'são', 'ser', 'através', 'forma', 'apresenta'])
        palavras_limpas = [p for p in palavras if p not in stopwords_pt and len(p) > 2]
        return dict(Counter(palavras_limpas).most_common(100))

def renderizar_nuvem_interativa_html(word_freq_dict):
    data_js = json.dumps([{"name": k, "value": v} for k, v in word_freq_dict.items()])
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/echarts-wordcloud@2.1.0/dist/echarts-wordcloud.min.js"></script>
    </head>
    <body style="margin:0; padding:0; background-color:transparent;">
        <div id="main" style="width:100%; height:450px;"></div>
        <script>
            var chart = echarts.init(document.getElementById('main'));
            var option = {{ tooltip: {{ show: true }}, series: [{{ type: 'wordCloud', shape: 'circle', sizeRange: [14, 70], textStyle: {{ fontFamily: 'sans-serif', fontWeight: 'bold', color: function () {{ return 'rgb(' + [Math.round(Math.random() * 150 + 100), Math.round(Math.random() * 150 + 100), Math.round(Math.random() * 150 + 100)].join(',') + ')'; }} }}, data: {data_js} }}] }};
            chart.setOption(option);
            window.onresize = chart.resize;
        </script>
    </body>
    </html>
    """

@st.cache_resource
def gerar_nodos_coocorrencia_agraph(dados_recorte, min_coocorrencia=1):
    G = nx.Graph()
    for d in dados_recorte:
        pks = d.get('palavras_chave', [])
        for pk in pks:
            if G.has_node(pk): G.nodes[pk]['count'] += 1
            else: G.add_node(pk, count=1, tipo='Conceito')
        for pk1, pk2 in itertools.combinations(pks, 2):
            if G.has_edge(pk1, pk2): G[pk1][pk2]['weight'] += 1
            else: G.add_edge(pk1, pk2, weight=1)

    G.remove_edges_from([(u, v) for u, v, attrs in G.edges(data=True) if attrs['weight'] < min_coocorrencia])
    G.remove_nodes_from(list(nx.isolates(G)))

    nodes, edges = [], []
    for node, attrs in G.nodes(data=True):
        tam = min(10 + (attrs['count'] * 1.5), 50)
        nodes.append(Node(id=node, label=node, size=tam, color='#2ECC71', shape='dot', title=f"{node}\nOcorrências: {attrs['count']}"))

    for u, v, attrs in G.edges(data=True):
        edges.append(Edge(source=u, target=v, width=attrs['weight']*0.5, color="rgba(200, 200, 200, 0.5)", title=f"Co-ocorrências: {attrs['weight']}"))

    return nodes, edges

@st.cache_data
def preparar_csv_exportacao(dados):
    df = pd.DataFrame(dados)
    for col in ['autores', 'co_orientadores', 'palavras_chave']:
        if col in df.columns: df[col] = df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    return df.to_csv(index=False).encode('utf-8')


@st.cache_data
def calcular_metricas_complexas(dados):
    """Calcula indicadores de ecologia profunda e topologia avançada."""
    G = nx.Graph()
    for d in dados:
        doc = d.get('titulo')
        if not doc: continue
        G.add_node(doc, tipo='Documento')
        for a in d.get('autores', []): G.add_edge(doc, a)
        ori = d.get('orientador')
        if ori: G.add_edge(doc, ori)
        for pk in d.get('palavras_chave', []): G.add_edge(doc, pk)

    if G.number_of_nodes() == 0: return {}

    # --- MÉTRICAS GLOBAIS ---
    densidade = nx.density(G)
    
    # Graus (Links por nó)
    graus = [d for n, d in G.degree()]
    links_stats = {
        'media': np.mean(graus),
        'min': np.min(graus),
        'max': np.max(graus),
        'std': np.std(graus)
    }

    # Eficiência e Redundância
    eficiencia = nx.global_efficiency(G)
    redundancia = 1 - eficiencia # Simplificação teórica comum

    # Entropia da Rede (Baseada na distribuição de graus)
    pk = np.array(nx.degree_histogram(G))
    pk = pk / pk.sum()
    pk = pk[pk > 0]
    entropia = -np.sum(pk * np.log2(pk))

    # Coeficiente de Agrupamento Médio (Clustering)
    clustering = nx.average_clustering(G)

    # --- MÉTRICAS DE NÓ (MÉDIAS GLOBAIS) ---
    # Para métricas que geram dicts, calculamos a média para o Dashboard
    pagerank_dict = nx.pagerank(G)
    eigen_dict = nx.eigenvector_centrality(G, max_iter=1000, weight=None)
    constraint_dict = nx.constraint(G) # Restrição de Burt

    return {
        'densidade': densidade,
        'links': links_stats,
        'eficiencia': eficiencia,
        'redundancia': redundancia,
        'entropia': entropia,
        'clustering': clustering,
        'pagerank_avg': np.mean(list(pagerank_dict.values())),
        'eigen_avg': np.mean(list(eigen_dict.values())),
        'constraint_avg': np.mean(list(constraint_dict.values())),
        'n_nos': G.number_of_nodes()
    }




def preparar_exportacao_grafo(G, formato):
    """Converte o grafo NetworkX para diferentes formatos de arquivo."""
    output = io.BytesIO()
    # Criamos uma cópia para não sujar o grafo original com strings de exportação
    G_export = G.copy()
    
    if formato == "GEXF (Gephi)":
        # Formato nativo do Gephi
        nx.write_gexf(G_export, output, encoding='utf-8')
        return output.getvalue(), "grafo_ufsc.gexf"
    
    elif formato == "GraphML":
        # Formato universal XML
        nx.write_graphml(G_export, output, encoding='utf-8')
        return output.getvalue(), "grafo_ufsc.graphml"
    
    elif formato == "JSON (Node-Link)":
        # Formato ideal para D3.js e web
        data = nx.node_link_data(G_export)
        return json.dumps(data, ensure_ascii=False).encode('utf-8'), "grafo_ufsc.json"

# =========================================================================
# INÍCIO DA INTERFACE (EXATAMENTE COMO O UTILIZADOR SOLICITOU)
# =========================================================================

st.title("🔭 Exploração Global do Conhecimento")
st.subheader(f"Base de Dados: {st.session_state.get('nome_programa', 'N/A')}")
st.markdown("---")


# --- SEÇÃO: MÉTRICAS DE ECOLOGIA PROFUNDA ---
st.markdown("### 🧬 Métricas de Ecologia Profunda (SNA Avançado)")
m_sna = calcular_metricas_complexas(dados_completos)

if m_sna:
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Densidade da Rede", f"{m_sna['densidade']:.5f}")
    col_s2.metric("Eficiência Global", f"{m_sna['eficiencia']:.4f}")
    col_s3.metric("Entropia (H)", f"{m_sna['entropia']:.2f} bits")
    col_s4.metric("Clustering Médio", f"{m_sna['clustering']:.4f}")

    with st.expander("📊 Estatísticas de Conectividade (Links por Nó)"):
        c_l1, c_l2, c_l3, c_l4 = st.columns(4)
        c_l1.metric("Média de Links", f"{m_sna['links']['media']:.2f}")
        c_l2.metric("Desvio Padrão", f"{m_sna['links']['std']:.2f}")
        c_l3.metric("Mínimo", m_sna['links']['min'])
        c_l4.metric("Máximo", m_sna['links']['max'])

    with st.expander("🧠 Indicadores de Influência e Estrutura (Médias)"):
        c_i1, c_i2, c_i3, c_i4 = st.columns(4)
        c_i1.metric("PageRank Médio", f"{m_sna['pagerank_avg']:.6f}")
        c_i2.metric("Eigenvector Médio", f"{m_sna['eigen_avg']:.6f}")
        c_i3.metric("Restrição (Burt)", f"{m_sna['constraint_avg']:.4f}")
        c_i4.metric("Redundância", f"{m_sna['redundancia']:.4f}")


# --- GLOSSÁRIO TÉCNICO ---
st.markdown("---")
with st.expander("📚 Glossário de Métricas de Ecologia do Conhecimento"):
    st.markdown("""
    ### Topologia e Fluxo
    * **Densidade:** Proporção de conexões reais frente às possíveis. $D = \\frac{2|E|}{|V|(|V|-1)}$. Indica quão "povoada" está a rede.
    * **Eficiência:** Mede quão fácil a informação viaja. Redes eficientes têm caminhos curtos entre quaisquer dois nós.
    * **Entropia da Rede ($H$):** Mede a diversidade e incerteza da distribuição de conexões. Uma entropia alta indica uma ecologia complexa e menos previsível.
    
    ### Centralidade e Poder
    * **PageRank:** Algoritmo do Google que mede a importância de um nó baseando-se na qualidade das suas conexões (não apenas quantidade).
    * **Eigenvector:** Mede a influência de um nó considerando que conexões com nós influentes valem mais.
    * **Restrição (Burt's Constraint):** Mede quanto um indivíduo está "preso" a um grupo. Baixa restrição indica um *Broker* (ponte entre diferentes saberes).
    
    ### Estrutura de Agrupamento
    * **Coeficiente de Agrupamento (Clustering):** Mede a probabilidade de dois vizinhos de um nó também estarem conectados entre si (formação de "bolhas" ou grupos coesos).
    * **Redundância:** Indica o excesso de caminhos para a mesma informação. É o inverso da eficiência na otimização de fluxos.
    """)

# === SEÇÃO 1: GRAFO INTERATIVO GERAL ===
st.header("🕸️ 1. Topologia e Grafo Interativo")
niveis_sel_grafo = st.multiselect("Nível Acadêmico (Grafo):", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_grafo")
dados_grafo = [d for d in dados_completos if d.get('nivel_academico', 'Outros') in niveis_sel_grafo]
total_grafo = len(dados_grafo)


if total_grafo > 0:
    with st.form("form_grafo"):
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            max_docs_g = total_grafo if total_grafo > 1 else 2
            n_registros_grafo = st.slider("Volume de Documentos para a Rede:", 1, max_docs_g, min(40, total_grafo), 1)
            metodo_coloracao = st.selectbox("Mapeamento de Cores e Comunidades:", ["Original (Categoria)", "Comunidades (Louvain)", "Comunidades (Greedy Modularity)", "Comunidades (Girvan-Newman)"])
        with col_g2:
            metodo_tamanho = st.selectbox("Tamanho dos Nós (Métrica SNA):", ["Tamanho Fixo (Original)", "Grau Absoluto", "Degree Centrality", "Betweenness"])
            st.markdown("<br>", unsafe_allow_html=True)
            btn_render_grafo = st.form_submit_button("Renderizar Grafo", use_container_width=True)

    if btn_render_grafo:
        with st.spinner("A construir a rede topológica visual..."):
            # Agora recebemos o objeto G
            nodes, edges, legendas, G_obj = gerar_nodos_globais_agraph(
                dados_grafo[:n_registros_grafo], 
                metodo_cor=metodo_coloracao, 
                metodo_tamanho=metodo_tamanho
            )
            st.session_state['graf_glob_nodes'] = nodes
            st.session_state['graf_glob_edges'] = edges
            st.session_state['kpis_grafo'] = {'nos': len(nodes), 'arestas': len(edges), 'legendas': legendas}
            st.session_state['G_atual'] = G_obj # Guardamos o objeto NetworkX
            st.session_state['grafo_pronto'] = True

    if st.session_state['grafo_pronto']:
        kpis = st.session_state['kpis_grafo']
        # ... (Mantém as métricas de Nós e Arestas que você já tinha)

        # --- NOVA ÁREA DE EXPORTAÇÃO ---
        st.markdown("### 📥 Exportar Estrutura de Rede")
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        
        G_para_exportar = st.session_state.get('G_atual')
        
        if G_para_exportar:
            # Botão 1: Gephi (GEXF)
            data_gexf, nome_gexf = preparar_exportacao_grafo(G_para_exportar, "GEXF (Gephi)")
            col_ex1.download_button("📂 Exportar para Gephi", data=data_gexf, file_name=nome_gexf, use_container_width=True)
            
            # Botão 2: GraphML
            data_ml, nome_ml = preparar_exportacao_grafo(G_para_exportar, "GraphML")
            col_ex2.download_button("💾 Exportar GraphML", data=data_ml, file_name=nome_ml, use_container_width=True)
            
            # Botão 3: JSON
            data_json, nome_json = preparar_exportacao_grafo(G_para_exportar, "JSON (Node-Link)")
            col_ex3.download_button("🌐 Exportar JSON Web", data=data_json, file_name=nome_json, use_container_width=True)

        # Renderização Visual
        config = Config(width="100%", height=650, directed=False, physics=True, nodeHighlightBehavior=True, highlightColor="#F1C40F")
        agraph(nodes=st.session_state['graf_glob_nodes'], edges=st.session_state['graf_glob_edges'], config=config)
else:
    st.warning("Nenhum documento selecionado para o Grafo.")

st.markdown("---")

# === SEÇÃO 2: ANÁLISE ESTRUTURAL (RANKING SNA) ===
st.header("🏆 2. Análise Estrutural e Rankings (SNA)")
with st.form("form_tabela"):
    col_t_filt1, col_t_filt2, col_t_filt3 = st.columns(3)
    with col_t_filt1:
        niveis_sel_tabela = st.multiselect("Nível Acadêmico:", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_tabela")
    with col_t_filt2:
        anos_sel_tabela = st.slider("Filtrar por Período (Ano):", min_ano_global, max_ano_global, (min_ano_global, max_ano_global), 1, key="ano_tab")
    with col_t_filt3:
        conceitos_contexto = st.multiselect("Filtrar por Documentos que contenham os Conceitos:", options=conceitos_unicos, default=[], help="Se vazio, analisa a rede inteira.")

    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        n_registros_tabela = st.slider("Volume de documentos base para o cálculo matemático:", 1, len(dados_completos), len(dados_completos), 1)
    with col_t2:
        top_x = st.number_input("Tamanho do Ranking (Top X):", min_value=1, max_value=5000, value=20, step=5)

    col_t3, col_t4, col_t5 = st.columns(3)
    categorias_disp = ["Documento", "Autor", "Orientador", "Conceito"]
    todas_metricas = ["Grau Absoluto", "Degree Centrality", "Betweenness"]
    
    with col_t3: cat_sel = st.multiselect("Categorias a exibir na tabela:", categorias_disp, default=["Orientador", "Conceito"])
    with col_t4: met_sel = st.multiselect("Métricas a exibir:", todas_metricas, default=["Grau Absoluto", "Betweenness"])
    with col_t5: met_ord = st.selectbox("Ordenar Ranking primariamente por:", met_sel if met_sel else todas_metricas)
        
    btn_render_tabela = st.form_submit_button("Processar e Atualizar Tabela", type="primary")

if btn_render_tabela:
    if met_sel and cat_sel:
        dados_tab_filtrados = []
        for d in dados_completos[:n_registros_tabela]:
            if d.get('nivel_academico', 'Outros') not in niveis_sel_tabela: continue
            ano_d = int(d.get('ano')) if d.get('ano') and str(d.get('ano')).isdigit() else None
            if not ano_d or ano_d < anos_sel_tabela[0] or ano_d > anos_sel_tabela[1]: continue
            if conceitos_contexto:
                pks_doc = set(d.get('palavras_chave', []))
                if not any(c in pks_doc for c in conceitos_contexto): continue
            dados_tab_filtrados.append(d)

        if not dados_tab_filtrados:
            st.warning("Nenhum documento atende aos filtros de Ano/Nível/Conceito definidos.")
        else:
            df_completo = obter_dataframe_metricas(dados_tab_filtrados)
            df_top_x = df_completo[df_completo['Categoria'].isin(cat_sel)].sort_values(by=met_ord, ascending=False).head(top_x)
            df_top_x.insert(0, 'Posição', range(1, len(df_top_x) + 1))
            
            st.session_state['df_top_x'] = df_top_x
            st.session_state['colunas_finais'] = ['Posição', 'Entidade (Nó)', 'Categoria'] + met_sel
            st.session_state['tabela_pronta'] = True

if st.session_state['tabela_pronta']:
    df_exibicao = st.session_state['df_top_x'].copy()
    colunas = st.session_state['colunas_finais']
    if 'Degree Centrality' in df_exibicao.columns: df_exibicao['Degree Centrality'] = df_exibicao['Degree Centrality'].apply(lambda x: f"{x:.4f}")
    if 'Betweenness' in df_exibicao.columns: df_exibicao['Betweenness'] = df_exibicao['Betweenness'].apply(lambda x: f"{x:.4f}")
    st.dataframe(df_exibicao[colunas], use_container_width=True, hide_index=True)

st.markdown("---")

# === SEÇÃO 3: EVOLUÇÃO CRONOLÓGICA ===
st.header("📈 3. Evolução Histórica (Temporal)")
df_geral_base = preparar_dados_base_df(dados_completos)

with st.form("form_historico"):
    col_h_filt1, col_h_filt2, col_h_filt3 = st.columns(3)
    with col_h_filt1:
        niveis_sel_hist = st.multiselect("Nível Acadêmico:", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_hist")
    with col_h_filt2:
        orientador_sel_hist = st.multiselect("Orientador(es):", options=orientadores_disponiveis, default=[], help="Deixe em branco para todos.")
    with col_h_filt3:
        anos_sel_hist = st.slider("Intervalo de Anos:", min_ano_global, max_ano_global, (min_ano_global, max_ano_global), 1, key="ano_hist")

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        agrupar_niveis_hist = st.radio("Visão dos Níveis:", ["Agrupar tudo (Total)", "Separar Teses e Dissertações"], horizontal=True)
    with col_h2:
        modo_grafico = st.radio("Modo de Análise:", ["Visão Geral (Volume)", "Análise por Conceito (Palavras-chave)"], horizontal=True)
        if modo_grafico == "Análise por Conceito (Palavras-chave)":
            top_5_default = pd.Series(lista_todos_conceitos).value_counts().head(5).index.tolist()
            conceitos_sel_hist = st.multiselect("Conceitos a comparar:", conceitos_unicos, default=top_5_default)
        else:
            conceitos_sel_hist = []

    btn_render_hist = st.form_submit_button("Atualizar Gráfico Histórico", type="primary")

if btn_render_hist and not df_geral_base.empty:
    df_hist = df_geral_base[
        (df_geral_base['Ano'] >= anos_sel_hist[0]) & 
        (df_geral_base['Ano'] <= anos_sel_hist[1]) &
        (df_geral_base['nivel_academico'].isin(niveis_sel_hist))
    ].copy()
    
    if orientador_sel_hist:
        df_hist = df_hist[df_hist['orientador'].isin(orientador_sel_hist)]

    if df_hist.empty:
        st.warning("Não há documentos no intervalo e filtros selecionados.")
    else:
        fig = None
        if modo_grafico == "Visão Geral (Volume)":
            if agrupar_niveis_hist == "Agrupar tudo (Total)":
                df_plot = df_hist.groupby('Ano').size().reset_index(name='Volume')
                fig = px.line(df_plot, x='Ano', y='Volume', markers=True, title="Total de Publicações por Ano")
            else:
                df_plot = df_hist.groupby(['Ano', 'nivel_academico']).size().reset_index(name='Volume')
                fig = px.line(df_plot, x='Ano', y='Volume', color='nivel_academico', markers=True, title="Publicações por Ano (Separado por Nível)")
        else:
            if not conceitos_sel_hist:
                st.warning("Selecione pelo menos um conceito.")
            else:
                df_exp = df_hist.explode('palavras_chave')
                df_exp = df_exp[df_exp['palavras_chave'].isin(conceitos_sel_hist)]
                if df_exp.empty:
                    st.info("Os conceitos não aparecem nos filtros selecionados.")
                else:
                    if agrupar_niveis_hist == "Agrupar tudo (Total)":
                        df_plot = df_exp.groupby(['Ano', 'palavras_chave']).size().reset_index(name='Frequência')
                        fig = px.line(df_plot, x='Ano', y='Frequência', color='palavras_chave', markers=True, title="Evolução de Conceitos Específicos")
                    else:
                        df_exp['Linha'] = df_exp['palavras_chave'] + " (" + df_exp['nivel_academico'].str.split(' ').str[0] + ")"
                        df_plot = df_exp.groupby(['Ano', 'Linha']).size().reset_index(name='Frequência')
                        fig = px.line(df_plot, x='Ano', y='Frequência', color='Linha', markers=True, title="Evolução de Conceitos (Separado por Nível)")

        if fig:
            fig.update_layout(xaxis_title="Ano", yaxis_title="Frequência", template="plotly_dark", hovermode="x unified", xaxis=dict(tickmode='linear', dtick=1))
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# === SEÇÃO 4: NUVEM DE PALAVRAS ===
st.header("☁️ 4. Lexicometria e Nuvem de Palavras")
with st.form("form_nuvem"):
    col_n_filt1, col_n_filt2, col_n_filt3 = st.columns(3)
    with col_n_filt1:
        niveis_sel_nuvem = st.multiselect("Nível Acadêmico:", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_nuvem")
    with col_n_filt2:
        orientador_sel_nuvem = st.multiselect("Orientador(es):", options=orientadores_disponiveis, default=[], help="Deixe em branco para considerar todos.")
    with col_n_filt3:
        anos_sel_nuvem = st.slider("Intervalo de Anos:", min_ano_global, max_ano_global, (min_ano_global, max_ano_global), 1, key="ano_nuvem")

    fonte_nuvem = st.radio("Base de texto:", ["Conceitos (Palavras-chave)", "Títulos dos Documentos", "Resumos (Abstracts)"], horizontal=True)
    btn_render_nuvem = st.form_submit_button("Gerar Nuvem de Palavras", type="primary")

if btn_render_nuvem and not df_geral_base.empty:
    df_nuvem = df_geral_base[
        (df_geral_base['Ano'] >= anos_sel_nuvem[0]) & 
        (df_geral_base['Ano'] <= anos_sel_nuvem[1]) &
        (df_geral_base['nivel_academico'].isin(niveis_sel_nuvem))
    ].copy()
    
    if orientador_sel_nuvem:
        df_nuvem = df_nuvem[df_nuvem['orientador'].isin(orientador_sel_nuvem)]

    if df_nuvem.empty:
        st.warning("Não há documentos nos filtros selecionados.")
    else:
        freq_dict = obter_frequencias_texto(df_nuvem, fonte_nuvem)
        if not freq_dict:
            st.info("Não foi possível extrair palavras suficientes.")
        else:
            html_nuvem = renderizar_nuvem_interativa_html(freq_dict)
            components.html(html_nuvem, height=480, scrolling=False)

st.markdown("---")

# === SEÇÃO 5: GRAFO DE CO-OCORRÊNCIA ===
st.header("🔗 5. Grafo de Co-ocorrência de Palavras")
st.write("Analise como os conceitos e palavras-chave se relacionam dentro das teses e dissertações (clusters temáticos).")

with st.form("form_coocorrencia"):
    col_c_filt1, col_c_filt2, col_c_filt3 = st.columns(3)
    with col_c_filt1:
        niveis_sel_co = st.multiselect("Nível Acadêmico:", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_co")
    with col_c_filt2:
        orientador_sel_co = st.multiselect("Orientador(es):", options=orientadores_disponiveis, default=[], help="Deixe em branco para todos.")
    with col_c_filt3:
        anos_sel_co = st.slider("Intervalo de Anos:", min_ano_global, max_ano_global, (min_ano_global, max_ano_global), 1, key="ano_co")
        
    min_peso_co = st.slider("Filtro de Ruído: Mostrar apenas conexões que ocorrem juntas pelo menos X vezes:", min_value=1, max_value=20, value=2)
    btn_render_coocorrencia = st.form_submit_button("Gerar Grafo de Co-ocorrência", type="primary")

if btn_render_coocorrencia:
    dados_co = []
    for d in dados_completos:
        if d.get('nivel_academico', 'Outros') not in niveis_sel_co: continue
        ano_d = int(d.get('ano')) if d.get('ano') and str(d.get('ano')).isdigit() else None
        if not ano_d or ano_d < anos_sel_co[0] or ano_d > anos_sel_co[1]: continue
        if orientador_sel_co and d.get('orientador', 'Não informado') not in orientador_sel_co: continue
        dados_co.append(d)

    if not dados_co:
        st.warning("Não há documentos nos filtros selecionados para a Co-ocorrência.")
    else:
        with st.spinner("A mapear co-ocorrências..."):
            nodes, edges = gerar_nodos_coocorrencia_agraph(dados_co, min_coocorrencia=min_peso_co)
            st.session_state['co_nodes'] = nodes
            st.session_state['co_edges'] = edges
            st.session_state['coocorrencia_pronta'] = True

if st.session_state.get('coocorrencia_pronta'):
    # Usamos .get() com um fallback vazio para evitar o KeyError de vez
    kpis_co = st.session_state.get('kpis_co', {'nos': 0, 'arestas': 0})
    nodes_co = st.session_state.get('co_nodes', [])
    edges_co = st.session_state.get('co_edges', [])

    c1, c2, c3 = st.columns(3)
    c1.metric("Conceitos Interligados", kpis_co['nos'])
    c2.metric("Conexões Formadas", kpis_co['arestas'])
    c3.info("Dica: Use o mouse para zoom e arraste os nós para organizar a visão.")
    
    if not nodes_co:
        st.info("Gere o grafo para visualizar os dados.")
    else:
        config_co = Config(
            width="100%", 
            height=650, 
            directed=False, 
            physics=True, 
            nodeHighlightBehavior=True, 
            highlightColor="#2ECC71",
            collapsible=False
        )
        
        agraph(nodes=nodes_co, edges=edges_co, config=config_co)

st.markdown("---")

# === SEÇÃO 6: EXPORTAÇÃO DA BASE ===
st.header("📥 6. Exportação da Base de Dados Bruta")
col_b1, col_b2 = st.columns(2)
with col_b1:
    json_string = json.dumps(dados_completos, ensure_ascii=False, indent=4)
    st.download_button("📄 Baixar Base Completa (JSON)", file_name="base_ppg.json", mime="application/json", data=json_string)
with col_b2:
    csv_bytes = preparar_csv_exportacao(dados_completos)
    st.download_button("📊 Baixar Base Completa (CSV)", file_name="base_ppg.csv", mime="text/csv", data=csv_bytes)
