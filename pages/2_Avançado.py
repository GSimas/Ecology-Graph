import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
import json
import streamlit.components.v1 as components
import numpy as np

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Análises Avançadas | Universal",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização Dark Mode
st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; }
    h1, h2, h3, h4 { color: #F39C12; font-family: 'Helvetica Neue', sans-serif; }
    .stMetric { background-color: #2C3E50; padding: 15px; border-radius: 10px; }
    button[kind="primary"] { background-color: #2ECC71 !important; color: white !important; font-weight: bold !important; border: none !important; }
    </style>
""", unsafe_allow_html=True)

# --- PROTEÇÃO E LEITURA DA MEMÓRIA VIVA (SESSION STATE) ---
if 'dados_completos' not in st.session_state or not st.session_state['dados_completos']:
    st.warning("⚠️ Nenhuma base de dados carregada na memória.")
    st.info("Por favor, vá à página inicial e inicie a extração de um Programa de Pós-Graduação antes de usar as análises avançadas.")
    st.stop()

dados_gerais = st.session_state['dados_completos']
nome_programa = st.session_state.get('nome_programa', 'Programa Desconhecido')

# --- FUNÇÕES DE BACK-END ---

@st.cache_data
def preparar_dataframe(dados_lista):
    df = pd.DataFrame(dados_lista)
    df['Ano'] = pd.to_numeric(df.get('ano'), errors='coerce')
    df = df.dropna(subset=['Ano'])
    df['Ano'] = df['Ano'].astype(int)
    df['nivel_academico'] = df.get('nivel_academico', 'Outros').fillna('Outros')
    return df

# --- LÓGICA 1: BURST DETECTION ---
@st.cache_data
def detetar_explosoes(df_base, min_freq, z_score):
    if df_base.empty or 'palavras_chave' not in df_base.columns: return pd.DataFrame()
    df_f = df_base.explode('palavras_chave').groupby(['Ano', 'palavras_chave']).size().reset_index(name='Frequencia')
    if df_f.empty: return pd.DataFrame()
    
    contagem = df_f.groupby('palavras_chave')['Frequencia'].sum()
    validos = contagem[contagem >= min_freq].index
    df_f = df_f[df_f['palavras_chave'].isin(validos)]
    
    if df_f.empty: return pd.DataFrame()
    
    anos = range(df_f['Ano'].min(), df_f['Ano'].max() + 1)
    grelha = pd.MultiIndex.from_product([anos, validos], names=['Ano', 'palavras_chave']).to_frame(index=False)
    df_c = pd.merge(grelha, df_f, on=['Ano', 'palavras_chave'], how='left').fillna(0).sort_values(['palavras_chave', 'Ano'])
    
    df_c['Media'] = df_c.groupby('palavras_chave')['Frequencia'].transform(lambda x: x.expanding().mean().shift(1).fillna(0))
    df_c['Std'] = df_c.groupby('palavras_chave')['Frequencia'].transform(lambda x: x.expanding().std().shift(1).fillna(0))
    df_c['Em_Explosao'] = (df_c['Frequencia'] > (df_c['Media'] + (z_score * df_c['Std']))) & (df_c['Frequencia'] >= 2)
    return df_c

# --- LÓGICA 2: GENEALOGIA ---
@st.cache_resource
def gerar_grafo_genealogico(dados_lista, orientadores_foco):
    G = nx.DiGraph()
    for d in dados_lista:
        ori = d.get('orientador')
        autores = d.get('autores', [])
        if ori:
            for autor in autores:
                G.add_edge(ori, autor, label=f"{d.get('nivel_academico', '')} ({d.get('ano', '')})")

    if orientadores_foco:
        nós_desc = set()
        for o in orientadores_foco:
            if G.has_node(o):
                nós_desc.update(nx.descendants(G, o))
                nós_desc.add(o)
        G = G.subgraph(nós_desc).copy()

    for node in G.nodes():
        tem_pupilos = G.out_degree(node) > 0
        foi_pupilo = G.in_degree(node) > 0
        if tem_pupilos and foi_pupilo: color, label = '#F39C12', f"🎓 {node}"
        elif tem_pupilos: color, label = '#E74C3C', f"🏛️ {node}"
        else: color, label = '#3498DB', node
        G.nodes[node].update({'color': color, 'label': label, 'size': 25 if tem_pupilos else 15})

    net = Network(height='700px', width='100%', bgcolor='#222222', font_color='white', directed=True, cdn_resources='remote')
    net.from_nx(G)
    net.set_options('{"layout": {"hierarchical": {"enabled": true, "direction": "UD", "sortMethod": "directed"}}, "physics": {"enabled": false}}')
    path = "temp_gen.html"
    net.save_graph(path)
    return path, G.number_of_nodes(), G.number_of_edges()

# --- LÓGICA 3: BURT'S CONSTRAINT ---
@st.cache_data
def calcular_burt(dados_lista):
    G = nx.Graph()
    for d in dados_lista:
        ori = d.get('orientador')
        if ori:
            G.add_node(ori, tipo='Orientador')
            for pk in d.get('palavras_chave', []):
                G.add_node(pk, tipo='Conceito')
                G.add_edge(ori, pk)
    constraint = nx.constraint(G)
    betweenness = nx.betweenness_centrality(G)
    degree = dict(G.degree())
    resumo = []
    for node in G.nodes():
        if G.nodes[node].get('tipo') == 'Orientador':
            resumo.append({
                'Orientador': node,
                'Restrição (Constraint)': constraint.get(node, 0),
                'Intermediação (Betweenness)': betweenness.get(node, 0),
                'Diversidade': degree.get(node, 0)
            })
    return pd.DataFrame(resumo)

# --- LÓGICA 4: SANKEY ---
@st.cache_data
def preparar_sankey(dados_lista, top_n=10):
    df = pd.DataFrame(dados_lista)
    if 'orientador' not in df.columns: return [], [], [], []
    top_orient = df['orientador'].value_counts().head(top_n).index.tolist()
    df_f = df[df['orientador'].isin(top_orient)]
    fluxos = []
    for _, row in df_f.iterrows():
        ori, niv = row.get('orientador'), row.get('nivel_academico', 'Outros')
        if not ori or not pd.notna(ori): continue
        fluxos.append({'src': ori, 'tgt': niv, 'val': 1})
        for pk in row.get('palavras_chave', [])[:2]:
            if pk: fluxos.append({'src': niv, 'tgt': pk, 'val': 1})
            
    if not fluxos: return [], [], [], []
    df_fluxos = pd.DataFrame(fluxos).groupby(['src', 'tgt']).sum().reset_index()
    nodes = list(set(df_fluxos['src']).union(set(df_fluxos['tgt'])))
    mapping = {name: i for i, name in enumerate(nodes)}
    return nodes, df_fluxos['src'].map(mapping), df_fluxos['tgt'].map(mapping), df_fluxos['val']


# --- LÓGICA 5: MÉTRICAS MEMÉTICAS (Atualizado com NLP e Padronização) ---
@st.cache_data
def calcular_metricas_memeticas(df_base):
    import re
    import unicodedata # Importação nativa do Python para tratamento de caracteres
    
    if df_base.empty: 
        return pd.DataFrame(), pd.DataFrame(), 0, 0, pd.DataFrame(), pd.DataFrame()

    # Dicionário de palavras vazias para limpar os títulos
    stopwords = {'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'uma', 'para', 'com', 'não', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'das', 'à', 'seu', 'sua', 'ou', 'nos', 'já', 'eu', 'também', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'sem', 'mesmo', 'aos', 'nas', 'me', 'esse', 'essa', 'num', 'nem', 'numa', 'pelos', 'pelas', 'este', 'esta', 'sobre', 'estudo', 'análise', 'proposta', 'uso', 'aplicação', 'desenvolvimento', 'modelo', 'sistema', 'avaliação', 'gestão', 'conhecimento', 'engenharia', 'objetivo', 'pesquisa', 'trabalho', 'resultados', 'método', 'foi', 'foram', 'são', 'ser', 'através', 'forma', 'apresenta', 'the', 'of', 'and', 'in', 'to', 'is', 'for', 'by', 'on', 'with', 'an', 'as', 'this', 'that', 'which', 'from', 'it', 'or', 'be', 'are', 'at', 'has', 'have', 'was', 'were', 'not', 'but', 'baseado', 'partir', 'sob', 'perspectiva', 'frente'}

    def remover_acentos(texto):
        """Remove todos os acentos e marcadores diacríticos do texto."""
        if not isinstance(texto, str): return ""
        return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

    # Normalizamos as stopwords também para que o "match" seja perfeito com os textos limpos
    stopwords_norm = {remover_acentos(w) for w in stopwords}

    def extrair_memes_completos(row):
        # 1. Pega as palavras-chave oficiais, passa para minúscula, remove acentos e espaços extras
        pks = row.get('palavras_chave', [])
        if not isinstance(pks, list):
            pks = [pks] if pd.notna(pks) else []
        memes = set([remover_acentos(str(p).lower().strip()) for p in pks if str(p).strip()])
        
        # 2. Pega as palavras do Título (limpo de acentos, mínimo de 3 letras e sem stopwords)
        titulo_norm = remover_acentos(str(row.get('titulo', '')).lower())
        
        # Como removemos os acentos no passo anterior, a regex [a-z] agora é suficiente e à prova de falhas
        palavras_titulo = re.findall(r'\b[a-z]{3,}\b', titulo_norm)
        memes.update([p for p in palavras_titulo if p not in stopwords_norm])
        
        return list(memes)

    # Aplica o extrator de DNA
    df_copy = df_base.copy()
    df_copy['memes_todos'] = df_copy.apply(extrair_memes_completos, axis=1)
    
    # Explode para ter uma linha por meme
    df_explodido = df_copy.explode('memes_todos')
    df_explodido = df_explodido[df_explodido['memes_todos'].notna()]
    df_explodido = df_explodido[df_explodido['memes_todos'].str.strip() != '']
    df_explodido['meme'] = df_explodido['memes_todos']

    # 1. FECUNDIDADE: Em quantos documentos diferentes o meme conseguiu entrar?
    fecundidade = df_explodido.groupby('meme')['titulo'].nunique().reset_index(name='fecundidade')
    
    # Separação das Populações (Mortos vs Vivos)
    df_mortos = fecundidade[fecundidade['fecundidade'] == 1][['meme']].rename(columns={'meme': 'Memes Mortos (1 Aparição)'})
    df_vivos = fecundidade[fecundidade['fecundidade'] > 1].sort_values('fecundidade', ascending=False).rename(columns={'meme': 'Memes Sobreviventes', 'fecundidade': 'Nº de Aparições'})

    mortalidade_count = len(df_mortos)
    sobreviventes_count = len(df_vivos)

    # 3. LONGEVIDADE (Meia-Vida baseada em Títulos e Keywords)
    longevidade = df_explodido.groupby('meme').agg(
        ano_nascimento=('Ano', 'min'),
        ano_extincao=('Ano', 'max'),
        total_aparicoes=('titulo', 'nunique')
    ).reset_index()
    
    longevidade['tempo_vida_anos'] = longevidade['ano_extincao'] - longevidade['ano_nascimento']
    longevidade_valida = longevidade[longevidade['total_aparicoes'] > 1].copy()

    return fecundidade, longevidade_valida, mortalidade_count, sobreviventes_count, df_mortos, df_vivos
    
# --- INTERFACE ---

st.title(f"🧪 Ecologia do Conhecimento: Análises Profundas")
st.subheader(f"Explorando o DNA de: {nome_programa}")
df_geral = preparar_dataframe(dados_gerais)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔥 Burst", "🌳 Genealogia", "🕳️ Furos (Burt)", "🌊 Fluxos (Sankey)", "🧬 Memética"])

with tab1:
    st.markdown("### Deteção de Explosões (Emergências)")
    with st.form("f_burst"):
        c1, c2, c3 = st.columns(3)
        min_f = c1.number_input("Freq. Mínima (Filtro Ruído):", 2, 50, 5)
        z_s = c2.slider("Sensibilidade (Z-Score):", 1.0, 3.0, 1.5)
        btn_b = st.form_submit_button("Analisar")
    if btn_b:
        df_b = detetar_explosoes(df_geral, min_f, z_s)
        if not df_b.empty and df_b['Em_Explosao'].any():
            st.plotly_chart(px.scatter(df_b[df_b['Em_Explosao']], x="Ano", y="palavras_chave", size="Frequencia", color="palavras_chave", template="plotly_dark", title="Rupturas Epistemológicas"))
        else: st.info("Sem explosões detetadas com os parâmetros atuais.")

with tab2:
    st.markdown("### DNA Académico (Linhagens)")
    todos_ori = sorted(list(set([d.get('orientador') for d in dados_gerais if d.get('orientador')])))
    ori_foco = st.multiselect("Filtrar Dinastia (Patriarcas/Matriarcas):", todos_ori)
    if st.button("Mapear Árvore de Descendência", type="primary"):
        p, n, e = gerar_grafo_genealogico(dados_gerais, ori_foco)
        st.write(f"**Tamanho da Dinastia:** {n} membros conectados através de {e} vínculos de orientação.")
        with open(p, 'r', encoding='utf-8') as f: components.html(f.read(), height=700)

with tab3:
    st.markdown("### Métrica de Burt (Brokers vs Bolhas)")
    df_bt = calcular_burt(dados_gerais)
    if not df_bt.empty:
        st.plotly_chart(px.scatter(df_bt, x="Diversidade", y="Restrição (Constraint)", size="Intermediação (Betweenness)", hover_name="Orientador", color="Restrição (Constraint)", color_continuous_scale="RdYlGn_r", template="plotly_dark", title="Mapa de Furos Estruturais"))
    else: st.warning("Dados insuficientes para calcular a métrica de restrição.")

with tab4:
    st.markdown("### Fluxos de Energia Epistémica (Sankey)")
    n_s = st.slider("Qtd Orientadores Principais:", 5, 30, 10)
    nodes, src, tgt, val = preparar_sankey(dados_gerais, n_s)
    if nodes:
        fig = go.Figure(go.Sankey(node=dict(pad=15, thickness=20, label=nodes, color="#F39C12"), link=dict(source=src, target=tgt, value=val, color="rgba(243, 156, 18, 0.4)")))
        fig.update_layout(template="plotly_dark", height=700)
        st.plotly_chart(fig, use_container_width=True)
    else: st.warning("Não há vínculos suficientes entre orientadores e conceitos.")

with tab5:
    st.markdown("### A Genética das Ideias (Teoria Memética)")
    st.write("Análise do ciclo de vida dos conceitos como entidades replicantes (memes) extraídos das palavras-chave e dos títulos das teses.")
    
    df_fecundidade, df_longevidade, mortos, vivos, df_mortos, df_vivos = calcular_metricas_memeticas(df_geral)
    
    if df_fecundidade.empty:
        st.warning("Dados insuficientes para calcular métricas meméticas.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Fecundidade vs. Mortalidade Infantil")
            st.caption("Memes que falharam em se replicar vs. Memes que sobreviveram.")
            
            fig_mortalidade = go.Figure(data=[go.Pie(
                labels=['Memes Mortos (1 aparição)', 'Memes Sobreviventes (>1 aparição)'],
                values=[mortos, vivos],
                hole=.6,
                marker_colors=['#E74C3C', '#2ECC71']
            )])
            fig_mortalidade.update_layout(
                template="plotly_dark", 
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_mortalidade, use_container_width=True)
            
            # NOVO: Tabelas de inspeção de memes
            with st.expander("👁️ Ver Catálogo de Memes (Vivos vs Mortos)", expanded=False):
                st.write("**🟢 Memes Sobreviventes (Top Fecundidade)**")
                st.dataframe(df_vivos, use_container_width=True, hide_index=True, height=250)
                
                st.write("**🔴 Memes Mortos (Cemitério de Ideias)**")
                # Mostramos uma amostra aleatória se houver muitos mortos para não travar a UI
                amostra_mortos = df_mortos.sample(min(100, len(df_mortos))) if len(df_mortos) > 0 else df_mortos
                st.dataframe(amostra_mortos, use_container_width=True, hide_index=True, height=250)
            
        with col2:
            st.markdown("#### Os Super-Memes (Maior Fecundidade)")
            st.caption("Conceitos com maior capacidade de replicação (espalhamento) na história do programa.")
            
            # Pega os 15 mais fecundos para o gráfico
            top_fecundos = df_vivos.head(15).rename(columns={'Memes Sobreviventes': 'meme', 'Nº de Aparições': 'fecundidade'})
            
            fig_fecundidade = px.bar(
                top_fecundos, 
                x='fecundidade', 
                y='meme', 
                orientation='h',
                color='fecundidade',
                color_continuous_scale='Viridis'
            )
            fig_fecundidade.update_layout(
                template="plotly_dark", 
                yaxis={'categoryorder':'total ascending'},
                xaxis_title="Nº de Teses/Dissertações Diferentes",
                yaxis_title="",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_fecundidade, use_container_width=True)

        st.markdown("---")
        
        st.markdown("#### Tempo de Meia-Vida do Conhecimento (Longevidade)")
        st.caption("Analisa a 'idade' dos conceitos considerando sua presença em Títulos e Palavras-chave. Memes com vida longa indicam pilares estruturais; memes recentes podem ser a vanguarda tecnológica.")
        
        min_aparicoes_long = st.slider("Filtrar por nº mínimo de replicações (para ver apenas memes consistentes):", min_value=2, max_value=50, value=5)
        
        df_long_filtrado = df_longevidade[df_longevidade['total_aparicoes'] >= min_aparicoes_long].copy()
        
        if not df_long_filtrado.empty:
            fig_longevidade = px.scatter(
                df_long_filtrado, 
                x="ano_nascimento", 
                y="tempo_vida_anos", 
                size="total_aparicoes", 
                color="ano_extincao",
                hover_name="meme",
                color_continuous_scale='Plasma',
                labels={
                    "ano_nascimento": "Ano de Nascimento (1ª Aparição)",
                    "tempo_vida_anos": "Longevidade (Anos de Sobrevivência)",
                    "total_aparicoes": "Total de Réplicas (Fecundidade)",
                    "ano_extincao": "Ano da Última Aparição",
                    "meme": "Meme Acadêmico"
                }
            )
            fig_longevidade.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig_longevidade, use_container_width=True)
        else:
            st.info("Nenhum meme encontrado com essa taxa de replicação mínima. Reduza o filtro acima.")