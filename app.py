import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network
from sickle import Sickle
import unicodedata
import json

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Ecologia do Conhecimento UFSC",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização customizada (CSS) para deixar a interface mais moderna
st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; }
    h1, h2, h3 { color: #F39C12; font-family: 'Helvetica Neue', sans-serif; }
    .stMetric { background-color: #2C3E50; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.5); }
    </style>
""", unsafe_allow_html=True)

# --- FUNÇÕES DE BACK-END (COM CACHE PARA PERFORMANCE) ---
@st.cache_data
def carregar_dados_locais():
    """Lê o arquivo JSON estático do repositório."""
    try:
        with open('base_ppgegc.json', 'r', encoding='utf-8') as f:
            dados = json.load(f)
        return dados
    except FileNotFoundError:
        st.error("Arquivo base_ppgegc.json não encontrado. Verifique se ele está no repositório.")
        return []

# E na hora de chamar os dados, você simplesmente usa:
# dados = carregar_dados_locais()

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
    net.set_options("""
    var options = {
      "physics": {"barnesHut": {"gravitationalConstant": -15000, "springLength": 150}},
      "interaction": {"hover": true, "navigationButtons": true}
    }
    """)
    # Salva o arquivo temporário
    path = "grafo_temp.html"
    net.save_graph(path)
    return path, G.number_of_nodes(), G.number_of_edges()

# --- CONSTRUÇÃO DA INTERFACE FRONT-END ---

st.title("🌌 Ecologia do Conhecimento: Rede Acadêmica UFSC")
st.markdown("*Uma exploração interativa das tessituras do saber científico, suas raízes e ramificações.*")

# Barra lateral de controles
with st.sidebar:
    st.header("Configurações da Pesquisa")
    st.write("Ajuste os parâmetros para buscar teses e dissertações no repositório.")
    n_registros = st.slider("Número de documentos para extrair:", min_value=10, max_value=100, value=30, step=10)
    st.markdown("---")
    st.info("Legenda da Rede:\n"
            "- 🟥 Quadrados: Teses/Dissertações\n"
            "- 🔵 Círculos: Autores\n"
            "- ⭐ Estrelas: Orientadores\n"
            "- 🔺 Triângulos: Palavras-chave")

# Executa as funções (mostra um spinner carregando na tela)
dados = extrair_dados_ufsc(limite=n_registros)
caminho_html, num_nos, num_arestas = gerar_grafo_html(dados)

# Painel de Métricas Rápidas (KPIs)
st.markdown("### 📊 Panorama da Rede Extraída")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Documentos", len(dados))
col2.metric("Total de Nós (Entidades)", num_nos)
col3.metric("Total de Conexões", num_arestas)
col4.metric("Densidade", f"{(num_arestas / num_nos):.2f} arestas/nó")

st.markdown("---")

# Renderização do Grafo Interativo dentro do Streamlit
st.markdown("### 🕸️ Grafo Interativo")
st.write("Utilize os menus dentro do quadro abaixo para buscar entidades específicas ou filtrar por categoria.")

# Lê o HTML gerado pelo Pyvis e injeta no Streamlit
with open(caminho_html, 'r', encoding='utf-8') as f:
    html_data = f.read()

components.html(html_data, height=650, scrolling=False)
