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
def gerar_grafo_e_metricas(dados):
    G = nx.Graph()
    
    for tese in dados:
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
        <hr>
        <b>📊 Métricas SNA:</b><br>
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
      "interaction": {
          "hover": true, 
          "navigationButtons": true,
          "tooltipDelay": 100
      }
    }
    """)
    
    path = "grafo_temp.html"
    net.save_graph(path)
    
    script_ocultar_nos = """
    network.on("selectNode", function (params) {
        if (params.nodes.length === 1) {
            var nodeId = params.nodes[0];
            var connectedNodes = network.getConnectedNodes(nodeId);
            var allNodes = nodes.get();
            var updates = [];
            
            for (var i = 0; i < allNodes.length; i++) {
                var nId = allNodes[i].id;
                if (nId === nodeId || connectedNodes.includes(nId)) {
                    updates.push({id: nId, hidden: false});
                } else {
                    updates.push({id: nId, hidden: true});
                }
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
st.markdown("*Navegue pelos saberes: Passe o rato para ver métricas. Clique num nó para isolar o seu ecossistema.*")

dados_completos = carregar_dados_locais()
total_documentos = len(dados_completos) if len(dados_completos) > 0 else 100

with st.sidebar:
    st.header("Análise Estrutural")
    n_registros = st.slider("Documentos para desenhar na rede:", min_value=5, max_value=total_documentos, value=40, step=5)
    st.markdown("---")
    st.info("**Instruções de Interação:**\n"
            "1. **Hover:** Sobre qualquer nó para ver as métricas SNA.\n"
            "2. **Clique:** Num nó para isolar a rede.\n"
            "3. **Clique fora:** No espaço negro para restaurar.")

# Filtro Principal de Dados para a Rede Visual
dados_filtrados = dados_completos[:n_registros]
caminho_html, num_nos, num_arestas, df_completo = gerar_grafo_e_metricas(dados_filtrados)

# KPIs
st.markdown("### 📊 Topologia da Rede Atual")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Documentos Renderizados", len(dados_filtrados))
col2.metric("Entidades (Nós)", num_nos)
col3.metric("Conexões (Arestas)", num_arestas)
col4.metric("Densidade", f"{(num_arestas / num_nos):.3f}" if num_nos > 0 else "0")

st.markdown("---")

# Layout em duas colunas: Uma para o Grafo, outra para o Ranking e Filtros
col_grafo, col_tabela = st.columns([2, 1])

with col_grafo:
    st.markdown("### 🕸️ Grafo Interativo")
    with open(caminho_html, 'r', encoding='utf-8') as f:
        components.html(f.read(), height=650, scrolling=False)

with col_tabela:
    st.markdown("### 🏆 Top 10 Hubs Analíticos")
    
    # --- FILTROS MULTIPLOS DA TABELA ---
    categorias_disponiveis = df_completo['Categoria'].unique().tolist()
    categorias_selecionadas = st.multiselect(
        "Filtre as Categorias:",
        options=categorias_disponiveis,
        default=categorias_disponiveis  # Por padrão, mostra todas
    )
    
    todas_metricas = ["Grau Absoluto", "Degree Centrality", "Betweenness"]
    metricas_selecionadas = st.multiselect(
        "Métricas para exibir na tabela:",
        options=todas_metricas,
        default=["Grau Absoluto", "Betweenness"]
    )
    
    # Proteção caso o utilizador desmarque todas as métricas
    if not metricas_selecionadas:
        st.warning("Selecione pelo menos uma métrica para exibir.")
        metricas_selecionadas = ["Grau Absoluto"]
        
    metrica_ordenacao = st.selectbox(
        "Ordenar o Ranking (Top 10) por:",
        options=metricas_selecionadas
    )
    
    st.markdown("<br>", unsafe_allow_html=True) # Espaçamento
    
    # --- PROCESSAMENTO DO PANDAS (FILTROS) ---
    # 1. Filtra pelas categorias escolhidas
    df_filtrado = df_completo[df_completo['Categoria'].isin(categorias_selecionadas)]
    
    # 2. Ordena pela métrica alvo e extrai o Top 10
    if not df_filtrado.empty:
        df_top10 = df_filtrado.sort_values(by=metrica_ordenacao, ascending=False).head(10)
        
        # 3. Formatação visual (apenas para a exibição na tela)
        df_visual = df_top10.copy()
        if 'Degree Centrality' in df_visual.columns:
            df_visual['Degree Centrality'] = df_visual['Degree Centrality'].apply(lambda x: f"{x:.4f}")
        if 'Betweenness' in df_visual.columns:
            df_visual['Betweenness'] = df_visual['Betweenness'].apply(lambda x: f"{x:.4f}")
        
        # Seleciona as colunas dinâmicas escolhidas pelo utilizador
        colunas_finais = ['Entidade (Nó)', 'Categoria'] + metricas_selecionadas
        
        st.dataframe(
            df_visual[colunas_finais], 
            use_container_width=True,
            hide_index=True
        )
        
        # --- EXPORTAÇÃO CSV ---
        # Exportamos os dados reais (df_top10) sem a conversão de texto para não atrapalhar cálculos no Excel
        csv_data = df_top10[colunas_finais].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Exportar Ranking (CSV)",
            data=csv_data,
            file_name=f"top10_{metrica_ordenacao.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Nenhuma entidade encontrada para as categorias selecionadas.")
