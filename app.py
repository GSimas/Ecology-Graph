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
    """Lê o ficheiro JSON estático do repositório."""
    try:
        with open('base_ppgegc.json', 'r', encoding='utf-8') as f:
            dados = json.load(f)
        return dados
    except FileNotFoundError:
        st.error("Ficheiro base_ppgegc.json não encontrado. Verifique se ele está no repositório.")
        return []

@st.cache_resource
def gerar_grafo_html(dados):
    """Monta o grafo NetworkX e renderiza o HTML do Pyvis."""
    G = nx.Graph()
    for tese in dados:
        doc_id = tese['titulo']
        G.add_node(doc_id, tipo='documento', ano=tese['ano'], title=f"<b>Tese:</b><br>{doc_id}<br><b>Ano:</b> {tese['ano']}", color='#E74C3C', shape='square', size=30)
        
        for autor in tese['autores']:
            G.add_node(autor, tipo='autor', title=f"<b>Autor:</b><br>{autor}", color='#3498DB', shape='dot', size=20)
            G.add_edge(autor, doc_id)
            
        if tese['orientador']:
            G.add_node(tese['orientador'], tipo='orientador', title=f"<b>Orientador:</b><br>{tese['orientador']}", color='#F39C12', shape='star', size=25)
            G.add_edge(tese['orientador'], doc_id)
            
        for pk in tese['palavras_chave']:
            G.add_node(pk, tipo='conceito', title=f"<b>Conceito:</b><br>{pk}", color='#2ECC71', shape='triangle', size=15)
            G.add_edge(doc_id, pk)

    # Configuração do Pyvis
    net = Network(height='600px', width='100%', bgcolor='#222222', font_color='white', select_menu=True, filter_menu=True, cdn_resources='remote')
    net.from_nx(G)
    
    # OTIMIZAÇÃO: Adicionado 'stabilization' para congelar a física após carregar e evitar lentidão extrema
    net.set_options("""
    var options = {
      "physics": {
          "barnesHut": {"gravitationalConstant": -15000, "springLength": 150},
          "stabilization": {"enabled": true, "iterations": 150}
      },
      "interaction": {"hover": true, "navigationButtons": true}
    }
    """)
    
    path = "grafo_temp.html"
    net.save_graph(path)
    return path, G.number_of_nodes(), G.number_of_edges()

# --- CONSTRUÇÃO DA INTERFACE FRONT-END ---

st.title("🌌 Ecologia do Conhecimento: Rede Acadêmica UFSC")
st.markdown("*Uma exploração interativa das tessituras do saber científico, suas raízes e ramificações.*")

# 1. Carrega todos os dados primeiro para saber o tamanho total
dados_completos = carregar_dados_locais()
total_documentos = len(dados_completos) if len(dados_completos) > 0 else 100

# Barra lateral de controles
with st.sidebar:
    st.header("Configurações da Pesquisa")
    st.write("Ajuste os parâmetros para visualizar a rede.")
    
    # 2. O slider agora usa o total de documentos como limite máximo e inicia em 30 para ser rápido
    n_registros = st.slider("Número de documentos para visualizar inicialmente:", 
                            min_value=5, 
                            max_value=total_documentos, 
                            value=30, 
                            step=5)
    
    st.markdown("---")
    st.info("Legenda da Rede:\n"
            "- 🟥 Quadrados: Teses/Dissertações\n"
            "- 🔵 Círculos: Autores\n"
            "- ⭐ Estrelas: Orientadores\n"
            "- 🔺 Triângulos: Palavras-chave")

# 3. FILTRO ATIVO: Corta a lista de dados consoante o valor escolhido no slider pelo utilizador
dados_filtrados = dados_completos[:n_registros]

# Executa as funções APENAS com os dados filtrados
caminho_html, num_nos, num_arestas = gerar_grafo_html(dados_filtrados)

# Painel de Métricas Rápidas (KPIs)
st.markdown("### 📊 Panorama da Rede Renderizada")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Documentos no ecrã", len(dados_filtrados))
col2.metric("Total de Nós (Entidades)", num_nos)
col3.metric("Total de Conexões", num_arestas)
col4.metric("Densidade", f"{(num_arestas / num_nos):.2f} arestas/nó" if num_nos > 0 else "0")

st.markdown("---")

# Renderização do Grafo Interativo dentro do Streamlit
st.markdown("### 🕸️ Grafo Interativo")

with open(caminho_html, 'r', encoding='utf-8') as f:
    html_data = f.read()

components.html(html_data, height=650, scrolling=False)
