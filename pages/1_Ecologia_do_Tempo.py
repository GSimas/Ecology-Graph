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
    page_title="Ecologia do Tempo & Genealogia | PPGEGC",
    page_icon="⏳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização customizada (CSS)
st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; }
    h1, h2, h3, h4, h5 { color: #F39C12; font-family: 'Helvetica Neue', sans-serif; }
    .stMetric { background-color: #2C3E50; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.5); }
    
    button[kind="primary"] {
        background-color: #2ECC71 !important;
        color: white !important;
        border-color: #27AE60 !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- FUNÇÕES DE BACK-END ---
@st.cache_data
def carregar_dados():
    try:
        with open('base_ppgegc.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Ficheiro base_ppgegc.json não encontrado.")
        return []

@st.cache_data
def preparar_dados_temporais(dados):
    df = pd.DataFrame(dados)
    df['Ano'] = pd.to_numeric(df.get('ano'), errors='coerce')
    df = df.dropna(subset=['Ano'])
    df['Ano'] = df['Ano'].astype(int)
    df['nivel_academico'] = df.get('nivel_academico', 'Outros').fillna('Outros')
    return df

@st.cache_data
def detetar_explosoes(df_exp, min_ocorrencias_totais, sensibilidade_z):
    # (Mantendo a lógica de Burst Detection anterior para consistência do ficheiro)
    df_freq = df_exp.explode('palavras_chave').groupby(['Ano', 'palavras_chave']).size().reset_index(name='Frequencia')
    contagem_total = df_freq.groupby('palavras_chave')['Frequencia'].sum()
    conceitos_validos = contagem_total[contagem_total >= min_ocorrencias_totais].index
    df_freq = df_freq[df_freq['palavras_chave'].isin(conceitos_validos)]
    
    if df_freq.empty: return pd.DataFrame()

    anos_unicos = range(df_freq['Ano'].min(), df_freq['Ano'].max() + 1)
    grelha = pd.MultiIndex.from_product([anos_unicos, conceitos_validos], names=['Ano', 'palavras_chave']).to_frame(index=False)
    df_completo = pd.merge(grelha, df_freq, on=['Ano', 'palavras_chave'], how='left').fillna(0).sort_values(['palavras_chave', 'Ano'])
    
    df_completo['Media_Historica'] = df_completo.groupby('palavras_chave')['Frequencia'].transform(lambda x: x.expanding().mean().shift(1).fillna(0))
    df_completo['Desvio_Historico'] = df_completo.groupby('palavras_chave')['Frequencia'].transform(lambda x: x.expanding().std().shift(1).fillna(0))
    df_completo['Limiar_Explosao'] = df_completo['Media_Historica'] + (sensibilidade_z * df_completo['Desvio_Historico'])
    df_completo['Em_Explosao'] = (df_completo['Frequencia'] > df_completo['Limiar_Explosao']) & (df_completo['Frequencia'] >= 2)
    return df_completo

@st.cache_resource
def gerar_grafo_genealogico(dados, orientadores_foco):
    """
    Cria um Grafo Direcionado (DAG) que rastreia a linhagem: 
    Orientador -> Aluno (Mestrado) -> Aluno (Doutorado) -> Novo Orientador.
    """
    G = nx.DiGraph()
    
    # 1. Mapear trajectórias individuais
    pessoa_docs = {} # {nome: [{'ano': 2010, 'tipo': 'Dissertação'}, ...]}
    
    for d in dados:
        titulo = d['titulo']
        ano = int(d['ano'])
        nivel = d['nivel_academico']
        orientador = d.get('orientador')
        autores = d.get('autores', [])
        
        for autor in autores:
            # Aresta de Mentoria: Orientador -> Autor
            if orientador:
                G.add_edge(orientador, autor, label=f"{nivel} ({ano})", title=f"Trabalho: {titulo}")
            
            # Guardar para trajectória pessoal
            if autor not in pessoa_docs: pessoa_docs[autor] = []
            pessoa_docs[autor].append({'ano': ano, 'nivel': nivel})

    # 2. Se um autor tem Mestrado e depois Doutorado, reforçamos a progressão
    # (Embora a aresta Orientador -> Aluno já crie a árvore, isto ajuda na lógica de DAG)
    
    # 3. Filtragem por Dinastias (Subgrafo a partir de orientadores chave)
    if orientadores_foco:
        nós_descendentes = set()
        for o in orientadores_foco:
            if G.has_node(o):
                # Pega todos os pupilos e pupilos dos pupilos (descendentes)
                nós_descendentes.update(nx.descendants(G, o))
                nós_descendentes.add(o)
        G = G.subgraph(nós_descendentes).copy()

    # 4. Configuração Visual Hierárquica
    for node, attrs in G.nodes(data=True):
        # Verificar se o nó é um "Mestre" (tem pupilos) ou "Pupilo"
        tem_pupilos = G.out_degree(node) > 0
        foi_pupilo = G.in_degree(node) > 0
        
        if tem_pupilos and foi_pupilo:
            attrs.update({'color': '#F39C12', 'size': 30, 'label': f"🎓 {node}", 'title': "Ex-aluno que se tornou Orientador"})
        elif tem_pupilos:
            attrs.update({'color': '#E74C3C', 'size': 35, 'label': f"🏛️ {node}", 'title': "Orientador Raiz (Patriarca/Matriarca)"})
        else:
            attrs.update({'color': '#3498DB', 'size': 20, 'label': node, 'title': "Pesquisador / Egresso"})

    net = Network(height='700px', width='100%', bgcolor='#222222', font_color='white', directed=True, layout=False)
    
    # Ativar layout hierárquico (Árvore de cima para baixo)
    net.from_nx(G)
    net.set_options("""
    var options = {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "levelSeparation": 150,
          "nodeSpacing": 200,
          "treeSpacing": 200,
          "direction": "UD",
          "sortMethod": "directed"
        }
      },
      "physics": {"enabled": false},
      "edges": {
        "arrows": {"to": {"enabled": true}},
        "color": {"inherit": "from"},
        "smooth": {"type": "cubicBezier", "forceDirection": "vertical"}
      }
    }
    """)
    
    path = "genealogia_temp.html"
    net.save_graph(path)
    return path, G.number_of_nodes(), G.number_of_edges()

# --- INTERFACE PRINCIPAL ---
st.title("⏳ Ecologia das Dinastias e do Tempo")

aba1, aba2 = st.tabs(["🔥 Burst Detection (Ideias)", "🌳 Genealogia Académica (Pessoas)"])

dados_brutos = carregar_dados()
df_base = preparar_dados_temporais(dados_brutos)

# --- ABA 1: BURST DETECTION ---
with aba1:
    st.subheader("Deteção de Emergências e Explosões Epistémicas")
    with st.form("form_burst"):
        col1, col2, col3 = st.columns(3)
        with col1: niveis_sel = st.multiselect("Nível:", sorted(df_base['nivel_academico'].unique()), default=sorted(df_base['nivel_academico'].unique()))
        with col2: min_freq = st.number_input("Frequência Mínima:", 2, 50, 5)
        with col3: z_score = st.slider("Sensibilidade (Z-Score):", 1.0, 3.0, 1.5, 0.1)
        btn_burst = st.form_submit_button("Analisar Explosões", type="primary")

    if btn_burst:
        df_burst = detetar_explosoes(df_base[df_base['nivel_academico'].isin(niveis_sel)], min_freq, z_score)
        if not df_burst.empty and df_burst['Em_Explosao'].any():
            eventos = df_burst[df_burst['Em_Explosao']]
            
            # Visualização Timeline
            fig_timeline = px.scatter(eventos, x="Ano", y="palavras_chave", size="Frequencia", color="palavras_chave",
                                     title="Momentos de Rutura Paradigmática", template="plotly_dark")
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            st.dataframe(eventos[['Ano', 'palavras_chave', 'Frequencia']].sort_values(by='Ano', ascending=False), use_container_width=True)
        else:
            st.info("Nenhuma explosão detetada com estes parâmetros.")

# --- ABA 2: GENEALOGIA ACADÉMICA ---
with aba2:
    st.subheader("DNA Académico: Linhagens e Dinastias do PPGEGC")
    st.markdown("""
    Este grafo direcionado revela como o conhecimento flui através das gerações. 
    - **Nós Vermelhos (🏛️):** Orientadores que deram origem a linhagens.
    - **Nós Cor-de-Laranja (🎓):** Pesquisadores que foram alunos e hoje são Orientadores (os elos da Dinastia).
    - **Nós Azuis:** Egressos e pesquisadores da rede.
    """)
    
    todos_orientadores = sorted(list(set([d.get('orientador') for d in dados_brutos if d.get('orientador')])))
    
    with st.form("form_genealogia"):
        orientadores_foco = st.multiselect(
            "Selecione o(s) Patriarca(s)/Matriarca(s) para isolar uma Dinastia:",
            options=todos_orientadores,
            help="Se deixar vazio, mostrará a rede completa (pode ser muito densa)."
        )
        btn_genealogia = st.form_submit_button("Mapear Descendência", type="primary")

    if btn_genealogia:
        path_gen, nos_gen, arestas_gen = gerar_grafo_genealogico(dados_brutos, orientadores_foco)
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Membros na Linhagem", nos_gen)
        col_m2.metric("Vínculos de Orientação", arestas_gen)
        
        with open(path_gen, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=800, scrolling=False)
            
    st.info("💡 **Dica Visual:** O grafo é hierárquico. Os orientadores raiz aparecem no topo, e os seus 'descendentes' académicos fluem para baixo.")

# Rodapé de Navegação
st.markdown("---")
st.caption("Fronteiras Avançadas de Ecologia do Conhecimento | PPGEGC UFSC")
