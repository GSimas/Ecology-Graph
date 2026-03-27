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
    page_title="Análise Avançada",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização customizada (CSS)
st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; }
    h1, h2, h3, h4, h5 { color: #F39C12; font-family: 'Helvetica Neue', sans-serif; }
    .stMetric { background-color: #2C3E50; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.5); }
    button[kind="primary"] { background-color: #2ECC71 !important; color: white !important; font-weight: bold !important; }
    </style>
""", unsafe_allow_html=True)

# --- FUNÇÕES DE APOIO ---
@st.cache_data
def carregar_dados():
    try:
        with open('base_ppgegc.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        st.error("Erro ao carregar base_ppgegc.json.")
        return []

@st.cache_data
def preparar_dataframe(dados):
    df = pd.DataFrame(dados)
    df['Ano'] = pd.to_numeric(df.get('ano'), errors='coerce')
    df = df.dropna(subset=['Ano'])
    df['Ano'] = df['Ano'].astype(int)
    return df

# --- 1. LÓGICA BURST DETECTION ---
@st.cache_data
def detetar_explosoes(df_exp, min_freq, z_score):
    df_f = df_exp.explode('palavras_chave').groupby(['Ano', 'palavras_chave']).size().reset_index(name='Frequencia')
    validos = df_f.groupby('palavras_chave')['Frequencia'].sum()
    validos = validos[validos >= min_freq].index
    df_f = df_f[df_f['palavras_chave'].isin(validos)]
    if df_f.empty: return pd.DataFrame()
    
    anos = range(df_f['Ano'].min(), df_f['Ano'].max() + 1)
    grelha = pd.MultiIndex.from_product([anos, validos], names=['Ano', 'palavras_chave']).to_frame(index=False)
    df_c = pd.merge(grelha, df_f, on=['Ano', 'palavras_chave'], how='left').fillna(0).sort_values(['palavras_chave', 'Ano'])
    
    df_c['Media'] = df_c.groupby('palavras_chave')['Frequencia'].transform(lambda x: x.expanding().mean().shift(1).fillna(0))
    df_c['Std'] = df_c.groupby('palavras_chave')['Frequencia'].transform(lambda x: x.expanding().std().shift(1).fillna(0))
    df_c['Em_Explosao'] = (df_c['Frequencia'] > (df_c['Media'] + (z_score * df_c['Std']))) & (df_c['Frequencia'] >= 2)
    return df_c

# --- 2. LÓGICA BURT'S CONSTRAINT ---
@st.cache_data
def calcular_metricas_burt(dados):
    G = nx.Graph()
    for d in dados:
        orientador = d.get('orientador')
        if orientador:
            G.add_node(orientador, tipo='Orientador')
            for pk in d.get('palavras_chave', []):
                G.add_node(pk, tipo='Conceito')
                G.add_edge(orientador, pk)
    
    # Cálculo da Restrição de Burt (Constraint)
    # Baixa restrição = Maior autonomia e acesso a furos estruturais (Brokers)
    constraint = nx.constraint(G)
    betweenness = nx.betweenness_centrality(G)
    degree = dict(G.degree())
    
    resumo = []
    for node in G.nodes():
        if G.nodes[node].get('tipo') == 'Orientador':
            resumo.append({
                'Orientador': node,
                'Restrição (Constraint)': constraint[node],
                'Intermediação (Betweenness)': betweenness[node],
                'Diversidade Epistémica (Grau)': degree[node]
            })
    return pd.DataFrame(resumo)

# --- 3. LÓGICA SANKEY ---
@st.cache_data
def preparar_dados_sankey(dados, top_n_orientadores=10):
    df = pd.DataFrame(dados)
    # Filtra top orientadores para não poluir
    top_orient = df['orientador'].value_counts().head(top_n_orientadores).index.tolist()
    df_filtrado = df[df['orientador'].isin(top_orient)]
    
    # Criar fluxos: Orientador -> Nível e Nível -> Conceito (Top 3 de cada trabalho)
    fluxos = []
    for _, row in df_filtrado.iterrows():
        ori = row['orientador']
        niv = row['nivel_academico']
        pks = row['palavras_chave'][:3]
        
        # Fluxo 1: Orientador -> Nível
        fluxos.append({'source': ori, 'target': niv, 'value': 1})
        # Fluxo 2: Nível -> Conceitos
        for pk in pks:
            fluxos.append({'source': niv, 'target': pk, 'value': 1})
            
    df_fluxos = pd.DataFrame(fluxos).groupby(['source', 'target']).sum().reset_index()
    
    # Mapeamento de nomes para IDs
    all_nodes = list(set(df_fluxos['source']).union(set(df_fluxos['target'])))
    mapping = {name: i for i, name in enumerate(all_nodes)}
    
    df_fluxos['source_id'] = df_fluxos['source'].map(mapping)
    df_fluxos['target_id'] = df_fluxos['target'].map(mapping)
    
    return all_nodes, df_fluxos

# --- INTERFACE ---
st.title("🔬 Ecologia do Conhecimento: Análise Profunda")
st.markdown("---")

dados = carregar_dados()
df_base = preparar_dataframe(dados)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔥 Burst Detection", 
    "🌳 Genealogia", 
    "🕳️ Furos Estruturais (Burt)", 
    "🌊 Fluxos de Energia (Sankey)"
])

# --- ABA 3: FUROS ESTRUTURAIS ---
with tab3:
    st.subheader("Furos Estruturais e Autonomia de Pesquisa")
    st.markdown("""
    **Restrição de Burt (Constraint):** Mede o quanto um orientador está preso a uma "bolha".
    - **Baixa Restrição (Eixo Y baixo):** Indica um *Broker* que conecta áreas diferentes.
    - **Alta Diversidade (Eixo X alto):** Indica um grande volume de diferentes conceitos orientados.
    """)
    
    df_burt = calcular_metricas_burt(dados)
    
    fig_burt = px.scatter(
        df_burt, 
        x="Diversidade Epistémica (Grau)", 
        y="Restrição (Constraint)",
        size="Intermediação (Betweenness)",
        hover_name="Orientador",
        color="Restrição (Constraint)",
        color_continuous_scale="RdYlGn_r", # Verde para baixa restrição (melhor)
        title="O Mapa dos Brokers: Diversidade vs. Restrição"
    )
    fig_burt.update_layout(template="plotly_dark")
    st.plotly_chart(fig_burt, use_container_width=True)
    
    st.info("💡 **Análise Poética:** Orientadores no canto inferior direito são os 'Polinizadores' do programa: têm muitos alunos e transitam livremente entre diferentes furos estruturais do conhecimento.")

# --- ABA 4: SANKEY ---
with tab4:
    st.subheader("O Fluxo da Produção Intelectual")
    st.write("Visualize como a 'energia' dos orientadores flui através dos níveis académicos até desaguar nos grandes temas.")
    
    n_ori = st.slider("Número de Orientadores no fluxo:", 5, 30, 10)
    nodes, df_s = preparar_dados_sankey(dados, n_ori)
    
    fig_sankey = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15, thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = nodes,
          color = "#F39C12"
        ),
        link = dict(
          source = df_s['source_id'],
          target = df_s['target_id'],
          value = df_s['value'],
          color = "rgba(243, 156, 18, 0.4)"
        ))])

    fig_sankey.update_layout(title_text="Cadeia de Valor do Conhecimento: Orientador → Nível → Conceito", 
                             font_size=12, template="plotly_dark", height=800)
    st.plotly_chart(fig_sankey, use_container_width=True)

# --- ABA 1: BURST DETECTION ---
with tab1:
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
with tab2:
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
