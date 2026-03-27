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
import itertools
import unicodedata
from sickle import Sickle

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Plataforma Universal de Ecologia",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILIZAÇÃO CUSTOMIZADA (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; }
    
    /* Estilo dos Cards Estatísticos (KPIs) */
    [data-testid="stMetricValue"] { font-size: 2rem !important; color: #F39C12 !important; }
    [data-testid="stMetricLabel"] { font-size: 1rem !important; color: #BDC3C7 !important; font-weight: bold; }
    
    div[data-testid="metric-container"] {
        background-color: #2C3E50;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.4);
        border-left: 5px solid #F39C12;
        transition: transform 0.3s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-left: 5px solid #2ECC71;
    }
    
    h1, h2, h3, h4, h5 { color: #F39C12; font-family: 'Helvetica Neue', sans-serif; }
    
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
if 'coocorrencia_pronta' not in st.session_state: st.session_state['coocorrencia_pronta'] = False







# --- FUNÇÕES DE EXTRAÇÃO AO VIVO (OAI-PMH) ---

@st.cache_data(ttl=86400) # Cache de 1 dia para a lista de programas
def obter_programas_ufsc():
    try:
        sickle = Sickle('https://repositorio.ufsc.br/oai/request')
        sets = sickle.ListSets()
        colecoes = {s.setName: s.setSpec for s in sets if s.setSpec.startswith('col_')}
        return dict(sorted(colecoes.items()))
    except Exception as e:
        st.error(f"Erro ao conectar com a UFSC: {e}")
        return {}

def extrair_melhor_ano(lista_datas):
    if not lista_datas: return None
    anos = [int(m) for d in lista_datas for m in re.findall(r'\b(19\d{2}|20\d{2})\b', str(d))]
    return str(min(anos)) if anos else None

def normalizar_nome(nome):
    if not nome: return ""
    return ''.join(c for c in unicodedata.normalize('NFD', nome) if unicodedata.category(c) != 'Mn').strip().title()

def normalizar_palavra_chave(pk):
    if not pk: return ""
    return ''.join(c for c in unicodedata.normalize('NFD', pk.lower().strip()) if unicodedata.category(c) != 'Mn')

def identificar_nivel(tipos, titulo=""):
    tipos_str = " ".join(tipos).lower()
    titulo_lower = titulo.lower()
    if 'doctoral' in tipos_str or 'tese' in tipos_str or 'tese' in titulo_lower: return 'Tese (Doutorado)'
    if 'master' in tipos_str or 'disserta' in tipos_str or 'disserta' in titulo_lower: return 'Dissertação (Mestrado)'
    return 'Outros'

def realizar_extracao(set_spec, status_placeholder):
    sickle = Sickle('https://repositorio.ufsc.br/oai/request')
    try: records = sickle.ListRecords(metadataPrefix='oai_dc', set=set_spec)
    except: return []

    dados_extraidos = []
    titulos_vistos = set()

    for i, record in enumerate(records):
        if i % 50 == 0:
            status_placeholder.info(f"⏳ A extrair dados do servidor da UFSC... Documentos processados: **{i}**")

        if record.header.deleted or not record.metadata: continue
            
        try:
            meta = record.metadata
            titulo = meta.get('title', [''])[0].strip()
            if not titulo or titulo in titulos_vistos: continue
            titulos_vistos.add(titulo)
            
            autores = [normalizar_nome(a) for a in meta.get('creator', []) if a.strip()]
            ano_real = extrair_melhor_ano(meta.get('date', []))
            nivel = identificar_nivel(meta.get('type', []), titulo)
            
            contrib = [normalizar_nome(c) for c in meta.get('contributor', []) if "universidade" not in c.lower() and "ufsc" not in c.lower()]
            orientador = contrib[0] if len(contrib) > 0 else None
            co_orientadores = contrib[1:] if len(contrib) > 1 else []
            
            pks = list(set([normalizar_palavra_chave(pk) for pk in meta.get('subject', []) if pk]))
            
            dados_extraidos.append({
                'titulo': titulo, 'nivel_academico': nivel, 'autores': autores,
                'orientador': orientador, 'co_orientadores': co_orientadores,
                'possui_coorientador': len(co_orientadores) > 0, 'palavras_chave': pks, 'ano': ano_real
            })
        except: continue

    status_placeholder.success(f"✅ Extração concluída! {len(dados_extraidos)} documentos capturados.")
    return dados_extraidos

# (MANTENHA AS FUNÇÕES `obter_dataframe_metricas`, `preparar_dados_base_df`, `preparar_csv_exportacao`, `gerar_html_pyvis`, `gerar_html_coocorrencia` etc. EXATAMENTE COMO ESTÃO)


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
    df['orientador'] = df.get('orientador', 'Não informado').fillna('Não informado')
    return df

@st.cache_data
def preparar_csv_exportacao(dados):
    df = pd.DataFrame(dados)
    for col in ['autores', 'co_orientadores', 'palavras_chave']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    return df.to_csv(index=False).encode('utf-8')

@st.cache_resource
def gerar_html_pyvis(dados_recorte, metodo_cor="Original (Categoria)", metodo_tamanho="Tamanho Fixo (Original)"):
    G = nx.Graph()
    for tese in dados_recorte:
        doc_id = tese['titulo']
        nivel = tese.get('nivel_academico', 'Não classificado')
        G.add_node(doc_id, tipo='Documento', ano=tese.get('ano', 'N/A'), nivel=nivel, autores=", ".join(tese.get('autores', [])), orientador=tese.get('orientador', 'Não informado'))
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
    graus_absolutos = dict(G.degree())

    max_deg = max(degree_cent.values()) if degree_cent else 1
    max_bet = max(betweenness_cent.values()) if betweenness_cent else 1
    max_grau = max(graus_absolutos.values()) if graus_absolutos else 1

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

    tamanhos_padrao = {'Documento': 30, 'Autor': 20, 'Orientador': 25, 'Conceito': 15}

    for node, attrs in G.nodes(data=True):
        tipo = attrs.get('tipo', 'Desconhecido')
        grau_atual = graus_absolutos[node]
        deg_c_atual = degree_cent[node]
        bet_c_atual = betweenness_cent[node]
        
        janela_sna = f"\n\n--- MÉTRICAS SNA ---\nGrau Absoluto: {grau_atual}\nCentralidade de Grau: {deg_c_atual:.4f}\nIntermediação: {bet_c_atual:.4f}"
        if node in mapeamento_comunidade:
            janela_sna += f"\n👉 Comunidade: {mapeamento_comunidade[node]}"

        if metodo_tamanho == "Grau Absoluto":
            tamanho = 10 + (grau_atual / max_grau) * 50
        elif metodo_tamanho == "Degree Centrality":
            tamanho = 10 + (deg_c_atual / max_deg) * 50 if max_deg > 0 else tamanhos_padrao.get(tipo, 20)
        elif metodo_tamanho == "Betweenness":
            tamanho = 10 + (bet_c_atual / max_bet) * 50 if max_bet > 0 else tamanhos_padrao.get(tipo, 20)
        else:
            tamanho = tamanhos_padrao.get(tipo, 20)

        if tipo == 'Documento':
            n_acad = attrs.get('nivel', 'N/A')
            attrs.update({'shape': 'square', 'size': tamanho, 'title': f"DOCUMENTO ({n_acad}):\n{node}\nAno: {attrs.get('ano')}\nAutor(es): {attrs.get('autores')}\nOrientador: {attrs.get('orientador')}{janela_sna}"})
        elif tipo == 'Autor':
            attrs.update({'shape': 'dot', 'size': tamanho, 'title': f"AUTOR:\n{node}{janela_sna}"})
        elif tipo == 'Orientador':
            attrs.update({'shape': 'star', 'size': tamanho, 'title': f"ORIENTADOR:\n{node}{janela_sna}"})
        elif tipo == 'Conceito':
            attrs.update({'shape': 'triangle', 'size': tamanho, 'title': f"CONCEITO:\n{node}{janela_sna}"})

        if metodo_cor == "Original (Categoria)":
            if tipo == 'Documento': attrs['color'] = '#E74C3C'
            elif tipo == 'Autor': attrs['color'] = '#3498DB'
            elif tipo == 'Orientador': attrs['color'] = '#F39C12'
            elif tipo == 'Conceito': attrs['color'] = '#2ECC71'

    net = Network(height='600px', width='100%', bgcolor='#222222', font_color='white', select_menu=True, filter_menu=True, cdn_resources='remote')
    net.from_nx(G)
    net.set_options('{"physics": {"barnesHut": {"gravitationalConstant": -15000, "springLength": 150}, "stabilization": {"enabled": true, "iterations": 150}}, "interaction": {"hover": true, "navigationButtons": true, "selectConnectedEdges": true}}')
    path = "grafo_temp.html"
    net.save_graph(path)
    return path, G.number_of_nodes(), G.number_of_edges(), legendas_comunidades

@st.cache_resource
def gerar_html_coocorrencia(dados_recorte, min_coocorrencia=1):
    G = nx.Graph()
    for d in dados_recorte:
        pks = d.get('palavras_chave', [])
        for pk in pks:
            if G.has_node(pk): G.nodes[pk]['count'] += 1
            else: G.add_node(pk, count=1, tipo='Conceito')
        
        for pk1, pk2 in itertools.combinations(pks, 2):
            if G.has_edge(pk1, pk2): G[pk1][pk2]['weight'] += 1
            else: G.add_edge(pk1, pk2, weight=1)

    arestas_remover = [(u, v) for u, v, attrs in G.edges(data=True) if attrs['weight'] < min_coocorrencia]
    G.remove_edges_from(arestas_remover)
    G.remove_nodes_from(list(nx.isolates(G)))

    for node, attrs in G.nodes(data=True):
        tamanho = 10 + (attrs['count'] * 1.5)
        attrs.update({'shape': 'dot', 'size': min(tamanho, 60), 'color': '#2ECC71', 'title': f"<b>Conceito:</b> {node}\nOcorrências Totais: {attrs['count']}"})

    for u, v, attrs in G.edges(data=True):
        peso = attrs['weight']
        attrs.update({'value': peso, 'title': f"Co-ocorrências: {peso}", 'color': 'rgba(255, 255, 255, 0.2)'})

    net = Network(height='600px', width='100%', bgcolor='#222222', font_color='white', select_menu=True, filter_menu=True, cdn_resources='remote')
    net.from_nx(G)
    net.set_options('{"physics": {"barnesHut": {"gravitationalConstant": -15000, "springLength": 150}, "stabilization": {"enabled": true, "iterations": 150}}, "interaction": {"hover": true, "navigationButtons": true, "selectConnectedEdges": true}}')
    path = "grafo_coocorrencia.html"
    net.save_graph(path)
    return path, G.number_of_nodes(), G.number_of_edges()

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
                            return 'rgb(' + [Math.round(Math.random() * 150 + 100), Math.round(Math.random() * 150 + 100), Math.round(Math.random() * 150 + 100)].join(',') + ')';
                        }}
                    }},
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

# --- INÍCIO DA INTERFACE FRONT-END ---

# ==========================================
# INÍCIO DA INTERFACE (GESTOR DE EXTRAÇÃO VS DASHBOARD)
# ==========================================

if 'dados_completos' not in st.session_state:
    st.title("🔌 Conexão Direta: Repositório Institucional UFSC")
    st.markdown("O sistema está conectado ao servidor OAI-PMH da universidade para listar os programas disponíveis.")
    
    with st.spinner("A mapear coleções na UFSC..."):
        colecoes_disponiveis = obter_programas_ufsc()
    
    if colecoes_disponiveis:
        with st.form("form_extracao"):
            st.markdown("### Selecione o Programa para Analisar")
            programa_selecionado = st.selectbox("Lista de Programas de Pós-Graduação (PPGs):", list(colecoes_disponiveis.keys()))
            st.info("⚠️ A extração ao vivo pode demorar alguns minutos. Não feche a janela.")
            btn_extrair = st.form_submit_button("Iniciar Extração ao Vivo", type="primary")
            
        status_box = st.empty()
        
        if btn_extrair:
            set_spec_alvo = colecoes_disponiveis[programa_selecionado]
            dados = realizar_extracao(set_spec_alvo, status_box)
            
            if dados:
                st.session_state['dados_completos'] = dados
                st.session_state['nome_programa'] = programa_selecionado
                st.rerun() # Recarrega para mostrar o Dashboard
    else:
        st.error("Não foi possível aceder à lista da UFSC neste momento.")
    st.stop() # Para a execução aqui se ainda não houver dados extraídos


# --- O DASHBOARD (SÓ APARECE APÓS A EXTRAÇÃO) ---
dados_completos = st.session_state['dados_completos']
nome_programa = st.session_state['nome_programa']

st.title(f"🌌 Ecologia do Conhecimento")
st.subheader(f"Base de Dados: {nome_programa}")

# Botão na barra lateral para voltar e escolher outro programa
if st.sidebar.button("🔄 Trocar de Programa / Nova Extração", type="primary"):
    del st.session_state['dados_completos']
    st.rerun()

# ---------------------------------------------------------
# DAQUI PARA BAIXO, MANTENHA O SEU CÓDIGO INTACTO!
# ---------------------------------------------------------
# (Onde começam as listas: niveis_disponiveis = sorted(list(set...)))
# (Os Cards estatísticos, Seção 1 do Grafo, Seção 2 Ranking, Seção 3 Evolução, etc.)



# --- PREPARAÇÃO DAS LISTAS GLOBAIS ---
niveis_disponiveis = sorted(list(set([d.get('nivel_academico', 'Não Classificado') for d in dados_completos])))
orientadores_disponiveis = sorted(list(set([d.get('orientador', 'Não informado') for d in dados_completos if d.get('orientador')])))
lista_todos_conceitos = []
for d in dados_completos: lista_todos_conceitos.extend(d.get('palavras_chave', []))
conceitos_unicos = sorted(list(set(lista_todos_conceitos)))
anos_disponiveis = [int(d.get('ano')) for d in dados_completos if d.get('ano') and str(d.get('ano')).isdigit()]
min_ano_global = min(anos_disponiveis) if anos_disponiveis else 2000
max_ano_global = max(anos_disponiveis) if anos_disponiveis else 2025

# --- CÁLCULO DOS KPIs PARA OS CARDS ---
total_docs = len(dados_completos)
teses = len([d for d in dados_completos if "Tese" in d.get('nivel_academico', '')])
dissertacoes = len([d for d in dados_completos if "Disserta" in d.get('nivel_academico', '')])

autores_set = set()
orientadores_set = set()
coorientadores_set = set()
keywords_set = set()

for d in dados_completos:
    for a in d.get('autores', []): autores_set.add(a)
    if d.get('orientador'): orientadores_set.add(d.get('orientador'))
    for co in d.get('co_orientadores', []): coorientadores_set.add(co)
    for kw in d.get('palavras_chave', []): keywords_set.add(kw)

# --- RENDERIZAÇÃO DO CABEÇALHO ---
st.title("🌌 Ecologia do Conhecimento: PPGEGC UFSC")
st.markdown("Plataforma de inteligência bibliométrica para mapeamento de redes académicas, evolução histórica e análise topológica estrutural do conhecimento.")

# Linha 1 de Cards
c1, c2, c3 = st.columns(3)
c1.metric("📄 Documentos Totais", total_docs)
c2.metric("🎓 Teses (Doutorado)", teses)
c3.metric("📜 Dissertações", dissertacoes)

st.markdown("<br>", unsafe_allow_html=True)

# Linha 2 de Cards
c4, c5, c6, c7 = st.columns(4)
c4.metric("✍️ Autores Únicos", len(autores_set))
c5.metric("🏫 Orientadores", len(orientadores_set))
c6.metric("🤝 Co-orientadores", len(coorientadores_set))
c7.metric("💡 Conceitos (Keywords)", len(keywords_set))

st.markdown("---")

# === SEÇÃO 1: GRAFO INTERATIVO GERAL ===
st.header("🕸️ 1. Topologia e Grafo Interativo")
niveis_sel_grafo = st.multiselect("Nível Académico (Grafo):", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_grafo")
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
            path, nos, arestas, legendas = gerar_html_pyvis(dados_grafo[:n_registros_grafo], metodo_cor=metodo_coloracao, metodo_tamanho=metodo_tamanho)
            st.session_state['path_grafo'] = path
            st.session_state['kpis_grafo'] = {'nos': nos, 'arestas': arestas, 'legendas': legendas}
            st.session_state['grafo_pronto'] = True

    if st.session_state['grafo_pronto']:
        kpis = st.session_state['kpis_grafo']
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        col_k1.metric("Nós no Grafo", kpis['nos'])
        col_k2.metric("Arestas no Grafo", kpis['arestas'])
        col_k3.metric("Densidade", f"{(kpis['arestas'] / kpis['nos']):.3f}" if kpis['nos'] > 0 else 0)
        col_k4.info("Dica: Clique num nó para ver ligações diretas (Highlight).")
        
        if kpis.get('legendas'):
            st.markdown("#### 🎨 Comunidades Identificadas")
            html_legend = "<div style='background-color:#2C3E50; padding:10px; border-radius:5px; margin-bottom:15px; display:flex; flex-wrap:wrap;'>"
            lendas_ordenadas = sorted(kpis.get('legendas', []), key=lambda x: x['tamanho'], reverse=True)
            for leg in lendas_ordenadas:
                html_legend += f"<div style='margin-right:20px; margin-bottom:5px; align-items:center;'><span style='display:inline-block; width:15px; height:15px; background-color:{leg['cor']}; border-radius:50%; vertical-align:middle; margin-right:5px;'></span><b>Comunidade {leg['id']}</b> ({leg['tamanho']} nós)</div>"
            html_legend += "</div>"
            st.markdown(html_legend, unsafe_allow_html=True)

        with open(st.session_state['path_grafo'], 'r', encoding='utf-8') as f:
            components.html(f.read(), height=650, scrolling=False)
else:
    st.warning("Nenhum documento selecionado para o Grafo.")

st.markdown("---")

# === SEÇÃO 2: ANÁLISE ESTRUTURAL (RANKING SNA) ===
st.header("🏆 2. Análise Estrutural e Rankings (SNA)")
with st.form("form_tabela"):
    col_t_filt1, col_t_filt2, col_t_filt3 = st.columns(3)
    with col_t_filt1:
        niveis_sel_tabela = st.multiselect("Nível Académico:", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_tabela")
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
        niveis_sel_hist = st.multiselect("Nível Académico:", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_hist")
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
        niveis_sel_nuvem = st.multiselect("Nível Académico:", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_nuvem")
    with col_n_filt2:
        orientador_sel_nuvem = st.multiselect("Orientador(es):", options=orientadores_disponiveis, default=[], help="Deixe em branco para considerar todos.")
    with col_n_filt3:
        anos_sel_nuvem = st.slider("Intervalo de Anos:", min_ano_global, max_ano_global, (min_ano_global, max_ano_global), 1, key="ano_nuvem")

    fonte_nuvem = st.radio("Base de texto:", ["Conceitos (Palavras-chave)", "Títulos dos Documentos"], horizontal=True)
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
        niveis_sel_co = st.multiselect("Nível Académico:", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_co")
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
            path_co, nos_co, arestas_co = gerar_html_coocorrencia(dados_co, min_coocorrencia=min_peso_co)
            st.session_state['path_co'] = path_co
            st.session_state['kpis_co'] = {'nos': nos_co, 'arestas': arestas_co}
            st.session_state['coocorrencia_pronta'] = True

if st.session_state['coocorrencia_pronta']:
    kpis_co = st.session_state['kpis_co']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conceitos Interligados", kpis_co['nos'])
    c2.metric("Conexões Formadas", kpis_co['arestas'])
    
    if kpis_co['nos'] == 0:
        st.info("O filtro de ruído está muito alto. Tente diminuir o número mínimo de co-ocorrências.")
    else:
        with open(st.session_state['path_co'], 'r', encoding='utf-8') as f:
            components.html(f.read(), height=650, scrolling=False)

st.markdown("---")

# === SEÇÃO 6: EXPORTAÇÃO DA BASE ===
st.header("📥 6. Exportação da Base de Dados Bruta")
col_b1, col_b2 = st.columns(2)
with col_b1:
    json_string = json.dumps(dados_completos, ensure_ascii=False, indent=4)
    st.download_button("📄 Baixar Base Completa (JSON)", file_name="base_ppgegc.json", mime="application/json", data=json_string)
with col_b2:
    csv_bytes = preparar_csv_exportacao(dados_completos)
    st.download_button("📊 Baixar Base Completa (CSV)", file_name="base_ppgegc.csv", mime="text/csv", data=csv_bytes)
