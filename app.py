import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import networkx.algorithms.community as nx_comm
from pyvis.network import Network
import pandas as pd
import plotly.express as px
import json
import re
from collections import Counter

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Ecologia do Conhecimento UFSC",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed" # Esconde a barra lateral, pois os filtros agora são independentes
)

# Estilização customizada (CSS)
st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; }
    h1, h2, h3, h4 { color: #F39C12; font-family: 'Helvetica Neue', sans-serif; }
    .stMetric { background-color: #2C3E50; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.5); }
    
    button[kind="primary"] {
        background-color: #2ECC71 !important;
        color: white !important;
        border-color: #27AE60 !important;
        font-weight: bold !important;
    }
    button[kind="primary"]:hover {
        background-color: #27AE60 !important;
        border-color: #2ECC71 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- INICIALIZAÇÃO DE ESTADO ---
if 'grafo_pronto' not in st.session_state: st.session_state['grafo_pronto'] = False
if 'tabela_pronta' not in st.session_state: st.session_state['tabela_pronta'] = False

# --- FUNÇÕES DE BACK-END INDEPENDENTES ---
@st.cache_data
def carregar_dados_locais():
    try:
        with open('base_ppgegc.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Ficheiro base_ppgegc.json não encontrado no repositório.")
        return []

@st.cache_data
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

@st.cache_data
def preparar_dados_base_df(dados):
    df = pd.DataFrame(dados)
    df['Ano'] = pd.to_numeric(df.get('ano'), errors='coerce')
    df = df.dropna(subset=['Ano'])
    df['Ano'] = df['Ano'].astype(int)
    df['nivel_academico'] = df.get('nivel_academico', 'Outros / Não Especificado').fillna('Outros / Não Especificado')
    df['titulo'] = df.get('titulo', '').fillna('')
    return df

@st.cache_data
def preparar_csv_exportacao(dados):
    """Achata as listas do JSON para permitir a exportação em CSV puro."""
    df = pd.DataFrame(dados)
    for col in ['autores', 'co_orientadores', 'palavras_chave']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    return df.to_csv(index=False).encode('utf-8')

@st.cache_resource
def gerar_html_pyvis(dados_recorte, metodo_cor="Original (Categoria)"):
    G = nx.Graph()
    for tese in dados_recorte:
        doc_id = tese['titulo']
        nivel = tese.get('nivel_academico', 'Não classificado')
        G.add_node(doc_id, tipo='Documento', ano=tese.get('ano', 'N/A'), nivel=nivel, autores=", ".join(tese.get('autores', [])), orientador=tese.get('orientador', 'Não informado'))
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

    # Dicionário e Lista para as Comunidades
    legendas_comunidades = []
    mapeamento_comunidade = {}

    if metodo_cor != "Original (Categoria)":
        comunidades = []
        if metodo_cor == "Comunidades (Louvain)": comunidades = nx_comm.louvain_communities(G)
        elif metodo_cor == "Comunidades (Greedy Modularity)": comunidades = nx_comm.greedy_modularity_communities(G)
        elif metodo_cor == "Comunidades (Girvan-Newman)":
            try: comunidades = next(nx_comm.girvan_newman(G))
            except StopIteration: comunidades = [set(G.nodes())]
            
        paleta = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000']
        
        for i, comm in enumerate(comunidades):
            cor_com = paleta[i % len(paleta)]
            id_com = i + 1
            legendas_comunidades.append({"id": id_com, "cor": cor_com, "tamanho": len(comm)})
            for node in comm:
                G.nodes[node]['color'] = cor_com
                mapeamento_comunidade[node] = id_com

    for node, attrs in G.nodes(data=True):
        tipo = attrs.get('tipo', 'Desconhecido')
        janela_sna = f"\n\n--- MÉTRICAS SNA ---\nGrau: {G.degree(node)}\nCentralidade de Grau: {degree_cent[node]:.4f}\nIntermediação: {betweenness_cent[node]:.4f}"
        
        if node in mapeamento_comunidade:
            janela_sna += f"\n👉 Comunidade: {mapeamento_comunidade[node]}"

        if tipo == 'Documento':
            n_acad = attrs.get('nivel', 'N/A')
            attrs.update({'shape': 'square', 'size': 30, 'title': f"DOCUMENTO ({n_acad}):\n{node}\nAno: {attrs.get('ano')}\nAutor(es): {attrs.get('autores')}\nOrientador: {attrs.get('orientador')}{janela_sna}"})
        elif tipo == 'Autor':
            attrs.update({'shape': 'dot', 'size': 20, 'title': f"AUTOR:\n{node}{janela_sna}"})
        elif tipo == 'Orientador':
            attrs.update({'shape': 'star', 'size': 25, 'title': f"ORIENTADOR:\n{node}{janela_sna}"})
        elif tipo == 'Conceito':
            attrs.update({'shape': 'triangle', 'size': 15, 'title': f"CONCEITO:\n{node}{janela_sna}"})

    if metodo_cor == "Original (Categoria)":
        for node, attrs in G.nodes(data=True):
            tipo = attrs.get('tipo', 'Desconhecido')
            if tipo == 'Documento': attrs['color'] = '#E74C3C'
            elif tipo == 'Autor': attrs['color'] = '#3498DB'
            elif tipo == 'Orientador': attrs['color'] = '#F39C12'
            elif tipo == 'Conceito': attrs['color'] = '#2ECC71'

    net = Network(height='600px', width='100%', bgcolor='#222222', font_color='white', select_menu=True, filter_menu=True, cdn_resources='remote')
    net.from_nx(G)
    net.set_options('{"physics": {"barnesHut": {"gravitationalConstant": -15000, "springLength": 150}, "stabilization": {"enabled": true, "iterations": 150}}, "interaction": {"hover": true, "navigationButtons": true, "tooltipDelay": 100}}')
    path = "grafo_temp.html"
    net.save_graph(path)
    
    script_ocultar = """
    network.on("selectNode", function (params) {
        if (params.nodes.length === 1) {
            var nodeId = params.nodes[0];
            var connectedNodes = network.getConnectedNodes(nodeId);
            nodes.update(nodes.get().map(n => ({id: n.id, hidden: !(n.id === nodeId || connectedNodes.includes(n.id))})));
        }
    });
    network.on("deselectNode", function () {
        nodes.update(nodes.get().map(n => ({id: n.id, hidden: false})));
    });
    """
    with open(path, 'r', encoding='utf-8') as f: html_content = f.read()
    with open(path, 'w', encoding='utf-8') as f: f.write(html_content.replace('return network;', script_ocultar + '\n\treturn network;'))
    return path, G.number_of_nodes(), G.number_of_edges(), legendas_comunidades

def obter_frequencias_texto(df_hist, fonte_nuvem):
    if fonte_nuvem == "Títulos dos Documentos":
        texto = " ".join(df_hist['titulo'].dropna().astype(str).tolist()).lower()
        texto = re.sub(r'[^\w\s]', '', texto)
        palavras = texto.split()
        stopwords_pt = set(['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'uma', 'para', 'com', 'não', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'das', 'à', 'seu', 'sua', 'ou', 'nos', 'já', 'eu', 'também', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'sem', 'mesmo', 'aos', 'nas', 'me', 'esse', 'essa', 'num', 'nem', 'numa', 'pelos', 'pelas', 'este', 'esta', 'sobre', 'estudo', 'análise', 'proposta', 'uso', 'aplicação', 'desenvolvimento', 'modelo', 'sistema', 'avaliação', 'gestão', 'conhecimento', 'engenharia'])
        palavras_limpas = [p for p in palavras if p not in stopwords_pt and len(p) > 2]
        return dict(Counter(palavras_limpas).most_common(100))
    else:
        lista_c = []
        for lst in df_hist['palavras_chave']: lista_c.extend(lst)
        return dict(Counter(lista_c).most_common(100))

def renderizar_nuvem_interativa_html(word_freq_dict):
    data_js = json.dumps([{"name": k, "value": v} for k, v in word_freq_dict.items()])
    html_content = f"""
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
            var option = {{
                tooltip: {{ show: true, formatter: '<b>{{b}}</b><br/>Ocorrências: {{c}}' }},
                series: [{{
                    type: 'wordCloud',
                    shape: 'circle',
                    left: 'center', top: 'center', width: '95%', height: '95%',
                    sizeRange: [14, 70],
                    rotationRange: [-45, 90], rotationStep: 45,
                    gridSize: 8,
                    drawOutOfBound: false, layoutAnimation: true,
                    textStyle: {{
                        fontFamily: 'sans-serif', fontWeight: 'bold',
                        color: function () {{
                            return 'rgb(' + [
                                Math.round(Math.random() * 150 + 100),
                                Math.round(Math.random() * 150 + 100),
                                Math.round(Math.random() * 150 + 100)
                            ].join(',') + ')';
                        }}
                    }},
                    emphasis: {{ focus: 'self', textStyle: {{ textShadowBlur: 10, textShadowColor: '#333' }} }},
                    data: {data_js}
                }}]
            }};
            chart.setOption(option);
            window.onresize = chart.resize;
        </script>
    </body>
    </html>
    """
    return html_content

# --- CONSTRUÇÃO DA INTERFACE FRONT-END ---

st.title("🌌 Ecologia do Conhecimento: PPGEGC UFSC")
st.markdown("> Plataforma de inteligência bibliométrica para mapeamento de redes académicas, evolução histórica e análise topológica estrutural do conhecimento.")
st.markdown("---")

dados_completos = carregar_dados_locais()
if not dados_completos:
    st.stop()

# Descobre os níveis disponíveis na base
niveis_disponiveis = list(set([d.get('nivel_academico', 'Não Classificado') for d in dados_completos]))
niveis_disponiveis.sort()

# --- SEÇÃO 1: GRAFO INTERATIVO ---
st.header("🕸️ Topologia e Grafo Interativo")
niveis_sel_grafo = st.multiselect("Filtrar Nível Académico (Exclusivo do Grafo):", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_grafo")
dados_grafo = [d for d in dados_completos if d.get('nivel_academico', 'Outros') in niveis_sel_grafo]
total_grafo = len(dados_grafo)

if total_grafo == 0:
    st.warning("Nenhum documento selecionado para o Grafo.")
else:
    with st.form("form_grafo"):
        col_g1, col_g2, col_g3 = st.columns([2, 2, 1])
        with col_g1:
            max_docs_g = total_grafo if total_grafo > 1 else 2
            n_registros_grafo = st.slider("Volume de Documentos para a Rede:", 1, max_docs_g, min(40, total_grafo), 1)
        with col_g2:
            metodo_coloracao = st.selectbox("Mapeamento de Cores e Comunidades:", ["Original (Categoria)", "Comunidades (Louvain)", "Comunidades (Greedy Modularity)", "Comunidades (Girvan-Newman)"])
        with col_g3:
            st.markdown("<br><br>", unsafe_allow_html=True)
            btn_render_grafo = st.form_submit_button("Renderizar Grafo", use_container_width=True)

    if btn_render_grafo:
        with st.spinner("A construir a rede topológica visual..."):
            path, nos, arestas, legendas = gerar_html_pyvis(dados_grafo[:n_registros_grafo], metodo_cor=metodo_coloracao)
            st.session_state['path_grafo'] = path
            st.session_state['kpis_grafo'] = {'nos': nos, 'arestas': arestas, 'legendas': legendas}
            st.session_state['grafo_pronto'] = True

    if st.session_state['grafo_pronto']:
        kpis = st.session_state['kpis_grafo']
        
        # Exibe as métricas e instruções
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nós", kpis['nos'])
        c2.metric("Arestas", kpis['arestas'])
        c3.metric("Densidade", f"{(kpis['arestas'] / kpis['nos']):.3f}" if kpis['nos'] > 0 else 0)
        c4.info("Dica: Clique num nó para isolá-lo.")
        
     # Desenha a Lenda Dinâmica se houver Comunidades
        if kpis.get('legendas'): # Usando .get() para evitar KeyError
            st.markdown("#### 🎨 Comunidades Identificadas")
            html_legend = "<div style='background-color:#2C3E50; padding:10px; border-radius:5px; margin-bottom:15px; display:flex; flex-wrap:wrap;'>"
            # Ordena por tamanho para mostrar as maiores comunidades primeiro
            lendas_ordenadas = sorted(kpis.get('legendas', []), key=lambda x: x['tamanho'], reverse=True)
            for leg in lendas_ordenadas:
                html_legend += f"<div style='margin-right:20px; margin-bottom:5px; align-items:center;'><span style='display:inline-block; width:15px; height:15px; background-color:{leg['cor']}; border-radius:50%; vertical-align:middle; margin-right:5px;'></span><b>Comunidade {leg['id']}</b> ({leg['tamanho']} nós)</div>"
            html_legend += "</div>"
            st.markdown(html_legend, unsafe_allow_html=True)

        with open(st.session_state['path_grafo'], 'r', encoding='utf-8') as f:
            components.html(f.read(), height=650, scrolling=False)

st.markdown("---")

# --- SEÇÃO 2: ANÁLISE ESTRUTURAL (RANKING) ---
st.header("🏆 Análise Estrutural e Rankings (SNA)")
niveis_sel_tabela = st.multiselect("Filtrar Nível Académico (Exclusivo da Tabela):", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_tabela")
dados_tabela = [d for d in dados_completos if d.get('nivel_academico', 'Outros') in niveis_sel_tabela]
total_tabela = len(dados_tabela)

if total_tabela == 0:
    st.warning("Nenhum documento selecionado para a Tabela.")
else:
    with st.form("form_tabela"):
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            max_docs_t = total_tabela if total_tabela > 1 else 2
            n_registros_tabela = st.slider("Documentos analisados matematicamente:", 1, max_docs_t, total_tabela, 1)
        with col_t2:
            top_x = st.number_input("Tamanho do Ranking (Top X):", min_value=1, max_value=5000, value=20, step=5)

        col_t3, col_t4, col_t5 = st.columns(3)
        categorias_disp = ["Documento", "Autor", "Orientador", "Conceito"]
        todas_metricas = ["Grau Absoluto", "Degree Centrality", "Betweenness"]
        
        with col_t3: cat_sel = st.multiselect("Categorias a incluir:", categorias_disp, default=["Orientador", "Conceito"])
        with col_t4: met_sel = st.multiselect("Métricas a exibir:", todas_metricas, default=["Grau Absoluto", "Betweenness"])
        with col_t5: met_ord = st.selectbox("Ordenar o Ranking por:", met_sel if met_sel else todas_metricas)
            
        btn_render_tabela = st.form_submit_button("Processar e Atualizar Tabela", type="primary")

    if btn_render_tabela:
        if met_sel and cat_sel:
            df_completo = obter_dataframe_metricas(dados_tabela[:n_registros_tabela])
            df_top_x = df_completo[df_completo['Categoria'].isin(cat_sel)].sort_values(by=met_ord, ascending=False).head(top_x)
            st.session_state['df_top_x'] = df_top_x
            st.session_state['colunas_finais'] = ['Entidade (Nó)', 'Categoria'] + met_sel
            st.session_state['tabela_pronta'] = True

    if st.session_state['tabela_pronta']:
        df_exibicao = st.session_state['df_top_x'].copy()
        colunas = st.session_state['colunas_finais']
        if 'Degree Centrality' in df_exibicao.columns: df_exibicao['Degree Centrality'] = df_exibicao['Degree Centrality'].apply(lambda x: f"{x:.4f}")
        if 'Betweenness' in df_exibicao.columns: df_exibicao['Betweenness'] = df_exibicao['Betweenness'].apply(lambda x: f"{x:.4f}")
        st.dataframe(df_exibicao[colunas], use_container_width=True, hide_index=True)

st.markdown("---")

# --- SEÇÃO 3: EVOLUÇÃO CRONOLÓGICA ---
st.header("📈 Evolução Histórica (Temporal)")
niveis_sel_hist = st.multiselect("Filtrar Nível Académico (Exclusivo do Gráfico Temporal):", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_hist")
dados_hist_raw = [d for d in dados_completos if d.get('nivel_academico', 'Outros') in niveis_sel_hist]

df_hist_geral = preparar_dados_base_df(dados_hist_raw)

if df_hist_geral.empty:
    st.warning("Não há dados válidos para gerar gráficos temporais.")
else:
    min_ano = int(df_hist_geral['Ano'].min())
    max_ano = int(df_hist_geral['Ano'].max())
    lista_todos_conceitos = []
    for c_list in df_hist_geral['palavras_chave']: lista_todos_conceitos.extend(c_list)
    conceitos_unicos = sorted(list(set(lista_todos_conceitos)))
    top_5_conceitos = pd.Series(lista_todos_conceitos).value_counts().head(5).index.tolist()

    with st.form("form_historico"):
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            anos_sel_hist = st.slider("Intervalo de Anos:", min_ano, max_ano, (min_ano, max_ano), 1)
            agrupar_niveis_hist = st.radio("Visão dos Níveis:", ["Agrupar tudo (Total)", "Separar Teses e Dissertações"], horizontal=True)
        with col_h2:
            modo_grafico = st.radio("Modo de Análise:", ["Visão Geral (Volume)", "Análise por Conceito (Palavras-chave)"], horizontal=True)
            if modo_grafico == "Análise por Conceito (Palavras-chave)":
                conceitos_sel = st.multiselect("Conceitos:", conceitos_unicos, default=top_5_conceitos)
            else:
                conceitos_sel = []

        btn_render_hist = st.form_submit_button("Atualizar Gráfico Histórico", type="primary")

    if btn_render_hist:
        df_hist = df_hist_geral[(df_hist_geral['Ano'] >= anos_sel_hist[0]) & (df_hist_geral['Ano'] <= anos_sel_hist[1])].copy()
        if df_hist.empty:
            st.warning("Não há documentos no intervalo selecionado.")
        else:
            if modo_grafico == "Visão Geral (Volume)":
                if agrupar_niveis_hist == "Agrupar tudo (Total)":
                    df_plot = df_hist.groupby('Ano').size().reset_index(name='Volume')
                    fig = px.line(df_plot, x='Ano', y='Volume', markers=True, title="Total de Publicações por Ano")
                else:
                    df_plot = df_hist.groupby(['Ano', 'nivel_academico']).size().reset_index(name='Volume')
                    fig = px.line(df_plot, x='Ano', y='Volume', color='nivel_academico', markers=True, title="Publicações por Ano (Separado por Nível)")
            else:
                if not conceitos_sel:
                    st.warning("Selecione pelo menos um conceito.")
                    fig = None
                else:
                    df_exp = df_hist.explode('palavras_chave')
                    df_exp = df_exp[df_exp['palavras_chave'].isin(conceitos_sel)]
                    if df_exp.empty:
                        st.info("Os conceitos não aparecem no intervalo selecionado.")
                        fig = None
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

# --- SEÇÃO 4: NUVEM DE PALAVRAS (INTERATIVA) ---
st.header("☁️ Lexicometria e Nuvem de Palavras")
niveis_sel_nuvem = st.multiselect("Filtrar Nível Académico (Exclusivo da Nuvem):", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_nuvem")
dados_nuvem_raw = [d for d in dados_completos if d.get('nivel_academico', 'Outros') in niveis_sel_nuvem]
df_nuvem_geral = preparar_dados_base_df(dados_nuvem_raw)

if df_nuvem_geral.empty:
    st.warning("Não há dados válidos para gerar a nuvem.")
else:
    min_ano_n = int(df_nuvem_geral['Ano'].min())
    max_ano_n = int(df_nuvem_geral['Ano'].max())

    with st.form("form_nuvem"):
        col_n1, col_n2 = st.columns(2)
        with col_n1:
            anos_sel_nuvem = st.slider("Intervalo de Anos (Nuvem):", min_ano_n, max_ano_n, (min_ano_n, max_ano_n), 1)
        with col_n2:
            fonte_nuvem = st.radio("Base de texto:", ["Conceitos (Palavras-chave)", "Títulos dos Documentos"], horizontal=True)

        btn_render_nuvem = st.form_submit_button("Gerar Nuvem de Palavras", type="primary")

    if btn_render_nuvem:
        df_nuvem = df_nuvem_geral[(df_nuvem_geral['Ano'] >= anos_sel_nuvem[0]) & (df_nuvem_geral['Ano'] <= anos_sel_nuvem[1])].copy()
        if df_nuvem.empty:
            st.warning("Não há documentos no intervalo selecionado.")
        else:
            freq_dict = obter_frequencias_texto(df_nuvem, fonte_nuvem)
            if not freq_dict:
                st.info("Não foi possível extrair palavras suficientes para a nuvem.")
            else:
                html_nuvem = renderizar_nuvem_interativa_html(freq_dict)
                components.html(html_nuvem, height=480, scrolling=False)

st.markdown("---")

# --- SEÇÃO 5: ACESSO À BASE DE DADOS BRUTA ---
st.header("📥 Exportação da Base de Dados Bruta")
st.write("Transfira os metadados integrais do PPGEGC para análises externas ou importação noutros softwares.")

col_b1, col_b2 = st.columns(2)

with col_b1:
    # Descarregar JSON puro
    json_string = json.dumps(dados_completos, ensure_ascii=False, indent=4)
    st.download_button(
        label="📄 Baixar Base Completa (JSON Original)",
        file_name="base_ppgegc.json",
        mime="application/json",
        data=json_string,
    )

with col_b2:
    # Descarregar CSV limpo
    csv_bytes = preparar_csv_exportacao(dados_completos)
    st.download_button(
        label="📊 Baixar Base Completa (Formato CSV)",
        file_name="base_ppgegc.csv",
        mime="text/csv",
        data=csv_bytes,
    )
