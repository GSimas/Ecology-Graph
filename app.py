import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network
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

@st.cache_resource(show_spinner="A calcular métricas SNA e a desenhar a rede...")
def gerar_grafo_html(dados):
    G = nx.Graph()
    
    # 1. Montar a topologia do Grafo (Nós e Arestas)
    for tese in dados:
        doc_id = tese['titulo']
        G.add_node(doc_id, tipo='documento', ano=tese['ano'])
        
        for autor in tese['autores']:
            G.add_node(autor, tipo='autor')
            G.add_edge(autor, doc_id)
            
        if tese.get('orientador'):
            G.add_node(tese['orientador'], tipo='orientador')
            G.add_edge(tese['orientador'], doc_id)
            
        for pk in tese.get('palavras_chave', []):
            G.add_node(pk, tipo='conceito')
            G.add_edge(doc_id, pk)

    # 2. Calcular Métricas de Social Network Analysis (SNA)
    # Nota: Betweenness pode demorar alguns segundos em grafos muito grandes, por isso está com cache!
    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G)
    
    # 3. Adicionar Atributos Visuais e Janelas de Métricas (Tooltips)
    for node, attrs in G.nodes(data=True):
        tipo = attrs.get('tipo', 'desconhecido')
        
        # Formatar a "janelinha" de texto HTML com as métricas daquele nó específico
        conexoes = G.degree(node)
        deg_c = degree_cent[node]
        bet_c = betweenness_cent[node]
        
        janela_sna = f"""
        <hr>
        <b>📊 Métricas SNA:</b><br>
        Grau Absoluto (Conexões): {conexoes}<br>
        Degree Centrality: {deg_c:.4f}<br>
        Betweenness: {bet_c:.4f}
        """
        
        if tipo == 'documento':
            attrs['color'] = '#E74C3C'
            attrs['shape'] = 'square'
            attrs['size'] = 30
            attrs['title'] = f"<b>Tese:</b><br>{node}<br><b>Ano:</b> {attrs.get('ano')}" + janela_sna
            
        elif tipo == 'autor':
            attrs['color'] = '#3498DB'
            attrs['shape'] = 'dot'
            attrs['size'] = 20
            attrs['title'] = f"<b>Autor:</b><br>{node}" + janela_sna
            
        elif tipo == 'orientador':
            attrs['color'] = '#F39C12'
            attrs['shape'] = 'star'
            attrs['size'] = 25
            attrs['title'] = f"<b>Orientador:</b><br>{node}" + janela_sna
            
        elif tipo == 'conceito':
            attrs['color'] = '#2ECC71'
            attrs['shape'] = 'triangle'
            attrs['size'] = 15
            attrs['title'] = f"<b>Conceito:</b><br>{node}" + janela_sna

    # 4. Configuração do Pyvis
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
    
    # --- 5. O SEGREDO: Injeção de JavaScript para ocultar nós ao clicar ---
    script_ocultar_nos = """
    // Captura o evento de clique num nó
    network.on("selectNode", function (params) {
        if (params.nodes.length === 1) {
            var nodeId = params.nodes[0];
            var connectedNodes = network.getConnectedNodes(nodeId);
            var allNodes = nodes.get();
            var updates = [];
            
            for (var i = 0; i < allNodes.length; i++) {
                var nId = allNodes[i].id;
                // Se o nó for o selecionado OU for vizinho dele, mantém visível. Caso contrário, oculta.
                if (nId === nodeId || connectedNodes.includes(nId)) {
                    updates.push({id: nId, hidden: false});
                } else {
                    updates.push({id: nId, hidden: true});
                }
            }
            nodes.update(updates);
        }
    });

    // Captura o evento de clique no fundo preto (desmarcar) para mostrar tudo novamente
    network.on("deselectNode", function (params) {
        var allNodes = nodes.get();
        var updates = [];
        for (var i = 0; i < allNodes.length; i++) {
            updates.push({id: allNodes[i].id, hidden: false});
        }
        nodes.update(updates);
    });
    """
    
    # Abrimos o HTML gerado, colamos o nosso script antes de finalizar a variável 'network', e salvamos.
    with open(path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    html_content = html_content.replace('return network;', script_ocultar_nos + '\n\treturn network;')
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    return path, G.number_of_nodes(), G.number_of_edges()

# --- CONSTRUÇÃO DA INTERFACE FRONT-END ---

st.title("🌌 Ecologia do Conhecimento: PPGEGC UFSC")
st.markdown("*Navegue pelos saberes: Passe o rato para ver métricas. Clique num nó para isolar o seu ecossistema.*")

dados_completos = carregar_dados_locais()
total_documentos = len(dados_completos) if len(dados_completos) > 0 else 100

with st.sidebar:
    st.header("Análise Estrutural")
    n_registros = st.slider("Documentos para analisar:", min_value=5, max_value=total_documentos, value=40, step=5)
    st.markdown("---")
    st.info("**Instruções de Interação:**\n"
            "1. **Passe o rato (Hover):** Sobre qualquer nó para ver as métricas SNA de centralidade.\n"
            "2. **Clique:** Num nó para fazer com que todo o resto da rede desapareça.\n"
            "3. **Clique fora:** No espaço negro para restaurar a rede completa.")

# Filtro
dados_filtrados = dados_completos[:n_registros]

# Geração
caminho_html, num_nos, num_arestas = gerar_grafo_html(dados_filtrados)

# KPIs
st.markdown("### 📊 Topologia da Rede Atual")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Documentos", len(dados_filtrados))
col2.metric("Entidades (Nós)", num_nos)
col3.metric("Conexões (Arestas)", num_arestas)
col4.metric("Densidade", f"{(num_arestas / num_nos):.3f}" if num_nos > 0 else "0")

st.markdown("---")

# Renderização
with open(caminho_html, 'r', encoding='utf-8') as f:
    components.html(f.read(), height=650, scrolling=False)
