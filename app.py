import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
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

# --- INICIALIZAÇÃO DE ESTADO (SESSION STATE) ---
# Necessário para manter o grafo visível ao clicar em outros botões
if 'grafo_renderizado' not in st.session_state:
    st.session_state['grafo_renderizado'] = False
if 'tabela_renderizada' not in st.session_state:
    st.session_state['tabela_renderizada'] = False

# --- FUNÇÕES DE BACK-END ---
@st.cache_data
def carregar_dados_locais():
    try:
        with open('base_ppgegc.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Ficheiro base_ppgegc.json não encontrado no repositório.")
        return []

@st.cache_resource(show_spinner="A processar a topologia e a calcular métricas SNA...")
def gerar_grafo_e_metricas(dados, n_registros):
    # Recorta os dados logo no início para poupar processamento
    dados_filtrados = dados[:n_registros]
    G = nx.Graph()
    
    for tese in dados_filtrados:
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
    
    lista_metricas = []

    for node, attrs in G.nodes(data=True):
        tipo = attrs.get('tipo', 'Desconhecido')
        conexoes = G.degree(node)
        deg_c = degree_cent[node]
        bet_c = betweenness_cent[node]
        
        lista_metricas.append({
            'Entidade (Nó)': node,
            'Categoria': tipo,
            'Grau Absoluto': conexoes,
            'Degree Centrality': deg_c,
            'Betweenness': bet_c
        })
        
        janela_sna = f"""
        <hr><b>📊 Métricas SNA:</b><br>
        Grau Absoluto: {conexoes}<br>
        Degree Centrality: {deg_c:.4f}<br>
        Betweenness: {bet_c:.4f}
        """
        
        if tipo == 'Documento':
            attrs['color'] = '#E74C3C'
            attrs['shape'] = 'square'
            attrs['size'] = 30
            attrs['title'] = f"<b>Tese:</b><br>{node}<br><b>Ano:</b> {attrs.get('ano')}" + janela_sna
        elif tipo == 'Autor':
            attrs['color'] = '#3498DB'
            attrs['shape'] = 'dot'
            attrs['size'] = 20
            attrs['title'] = f"<b>Autor:</b><br>{node}" + janela_sna
        elif tipo == 'Orientador':
            attrs['color'] = '#F39C12'
            attrs['shape'] = 'star'
            attrs['size'] = 25
            attrs['title'] = f"<b>Orientador:</b><br>{node}" + janela_sna
        elif tipo == 'Conceito':
            attrs['color'] = '#2ECC71'
            attrs['shape'] = 'triangle'
            attrs['size'] = 15
            attrs['title'] = f"<b>Conceito:</b><br>{node}" + janela_sna

    df_metricas = pd.DataFrame(lista_metricas)

    net = Network(height='600px', width='100%', bgcolor='#222222', font_color='white', select_menu=True, filter_menu=True, cdn_resources='remote')
    net.from_nx(G)
    
    net.set_options("""
    var options = {
      "physics": {
          "barnesHut": {"gravitationalConstant": -15000, "springLength": 150},
          "stabilization": {"enabled": true, "iterations": 150}
      },
      "interaction": { "hover": true, "navigationButtons": true, "tooltipDelay": 100 }
    }
    """)
    
    path = "grafo_temp.html"
    net.save_graph(path)
    
    # Injeção de JavaScript para ocultar nós ao clicar
    script_ocultar_nos = """
    network.on("selectNode", function (params) {
        if (params.nodes.length === 1) {
            var nodeId = params.nodes[0];
            var connectedNodes = network.getConnectedNodes(nodeId);
            var allNodes = nodes.get();
            var updates = [];
            for (var i = 0; i < allNodes.length; i++) {
                var nId = allNodes[i].id;
                updates.push({id: nId, hidden: !(nId === nodeId || connectedNodes.includes(nId))});
            }
            nodes.update(updates);
        }
    });
    network.on("deselectNode", function (params) {
        var allNodes = nodes.get();
        var updates = [];
        for (var i = 0; i < allNodes.length; i++) {
            updates.push({id: allNodes[i].id, hidden: false});
        }
        nodes.update(updates);
    });
    """
    with open(path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    html_content = html_content.replace('return network;', script_ocultar_nos + '\n\treturn network;')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    return path, G.number_of_nodes(), G.number_of_edges(), df_metricas

# --- CONSTRUÇÃO DA INTERFACE FRONT-END ---

st.title("🌌 Ecologia do Conhecimento: PPGEGC UFSC")

# TEXTO EXPLICATIVO DENSO E OBJETIVO
st.markdown("""
> Esta plataforma é um instrumento de Ecologia do Conhecimento aplicada à bibliometria. Ela mapeia e quantifica as intrincadas relações entre teses, autores, orientadores e conceitos do Programa de Pós-Graduação em Engenharia e Gestão do Conhecimento (PPGEGC/UFSC) através da Ciência de Redes. O seu propósito é revelar a genealogia académica, identificar *hubs* de influência e expor as pontes interdisciplinares que estruturam a produção científica do programa, traduzindo repositórios institucionais numa topologia visual navegável e em métricas rigorosas de centralidade matemática.
""")
st.markdown("---")

dados_completos = carregar_dados_locais()
total_documentos = len(dados_completos) if len(dados_completos) > 0 else 100

# 1. CONTROLO DO GRAFO (LATERAL)
with st.sidebar:
    st.header("1. Motor de Renderização")
    # O uso do formulário impede a atualização automática ao mexer no slider
    with st.form("form_grafo"):
        n_registros = st.slider("Documentos-base da rede:", min_value=5, max_value=total_documentos, value=40, step=5)
        btn_renderizar_grafo = st.form_submit_button("Renderizar Visualização Pyvis")
        
    if btn_renderizar_grafo:
        # Quando clicado, guarda os resultados em cache e marca como renderizado
        caminho_html, num_nos, num_arestas, df_completo = gerar_grafo_e_metricas(dados_completos, n_registros)
        st.session_state['caminho_html'] = caminho_html
        st.session_state['num_nos'] = num_nos
        st.session_state['num_arestas'] = num_arestas
        st.session_state['df_completo'] = df_completo
        st.session_state['grafo_renderizado'] = True

    st.markdown("---")
    st.info("**Interação na Rede:**\n"
            "1. **Hover:** Veja métricas no nó.\n"
            "2. **Clique:** Isole o ecossistema.\n"
            "3. **Clique fora:** Restaure a rede.")

# 2. ÁREA PRINCIPAL (REDE)
if st.session_state['grafo_renderizado']:
    # KPIs
    st.markdown("### 🕸️ Grafo Interativo e Topologia")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documentos", n_registros)
    col2.metric("Entidades (Nós)", st.session_state['num_nos'])
    col3.metric("Conexões (Arestas)", st.session_state['num_arestas'])
    
    densidade = (st.session_state['num_arestas'] / st.session_state['num_nos']) if st.session_state['num_nos'] > 0 else 0
    col4.metric("Densidade", f"{densidade:.3f}")

    # Renderização visual
    with open(st.session_state['caminho_html'], 'r', encoding='utf-8') as f:
        components.html(f.read(), height=650, scrolling=False)

    st.markdown("---")

    # 3. ÁREA INFERIOR (TABELA E RANKING)
    st.markdown("### 🏆 Ranking Estrutural e Exportação")
    
    df_base = st.session_state['df_completo']
    categorias_disponiveis = df_base['Categoria'].unique().tolist()
    todas_metricas = ["Grau Absoluto", "Degree Centrality", "Betweenness"]

    # Formulário para a tabela (também impede atualização automática ao selecionar múltiplos itens)
    with st.form("form_tabela"):
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            cat_sel = st.multiselect("Categorias de Nós:", options=categorias_disponiveis, default=categorias_disponiveis)
        with col_f2:
            met_sel = st.multiselect("Métricas na Tabela:", options=todas_metricas, default=["Grau Absoluto", "Betweenness"])
        with col_f3:
            # Dropdown para forçar o utilizador a escolher apenas uma regra matemática para ordenar
            met_ord = st.selectbox("Regra Matemática de Ordenação (Top 10):", options=met_sel if met_sel else todas_metricas)
            
        btn_atualizar_tabela = st.form_submit_button("Atualizar Tabela")

    if btn_atualizar_tabela:
        # Guarda as opções no estado para sobrevivência da tabela
        st.session_state['tabela_opcoes'] = {'cat': cat_sel, 'met': met_sel, 'ord': met_ord}
        st.session_state['tabela_renderizada'] = True

    # Renderização da Tabela
    if st.session_state['tabela_renderizada']:
        opcoes = st.session_state['tabela_opcoes']
        
        if not opcoes['met']:
            st.warning("Selecione pelo menos uma métrica para exibir.")
        else:
            df_filtrado = df_base[df_base['Categoria'].isin(opcoes['cat'])]
            
            if not df_filtrado.empty:
                df_top10 = df_filtrado.sort_values(by=opcoes['ord'], ascending=False).head(10)
                
                # Formatação visual (cópia para não corromper o CSV com textos)
                df_visual = df_top10.copy()
                if 'Degree Centrality' in df_visual.columns:
                    df_visual['Degree Centrality'] = df_visual['Degree Centrality'].apply(lambda x: f"{x:.4f}")
                if 'Betweenness' in df_visual.columns:
                    df_visual['Betweenness'] = df_visual['Betweenness'].apply(lambda x: f"{x:.4f}")
                
                colunas_finais = ['Entidade (Nó)', 'Categoria'] + opcoes['met']
                
                st.dataframe(df_visual[colunas_finais], use_container_width=True, hide_index=True)
                
                # Exportação CSV
                csv_data = df_top10[colunas_finais].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descarregar Ranking (CSV)",
                    data=csv_data,
                    file_name=f"top10_{opcoes['ord'].lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("Nenhuma entidade encontrada para as categorias selecionadas.")
