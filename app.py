import streamlit as st
import streamlit.components.v1 as components
from streamlit_agraph import agraph, Node, Edge, Config
import networkx as nx
import networkx.algorithms.community as nx_comm
import pandas as pd
import json
import plotly.express as px
import re
import unicodedata
from collections import Counter
import gzip
import google.generativeai as genai
import requests
import urllib.parse

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Ecologia do Conhecimento UFSC", page_icon="🌌", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; }
    [data-testid="stMetricValue"] { font-size: 2rem !important; color: #F39C12 !important; }
    [data-testid="stMetricLabel"] { font-size: 1rem !important; color: #BDC3C7 !important; font-weight: bold; }
    div[data-testid="metric-container"] { background-color: #2C3E50; padding: 15px; border-radius: 12px; border-left: 5px solid #F39C12; }
    h1, h2, h3, h4, h5 { color: #F39C12; font-family: 'Helvetica Neue', sans-serif; }
    button[kind="primary"] { background-color: #2ECC71 !important; color: white !important; font-weight: bold !important; border: none !important; }
    </style>
""", unsafe_allow_html=True)

# --- INICIALIZAÇÃO DE ESTADO ---
if 'busca_tipo' not in st.session_state: 
    st.session_state.update({'busca_tipo': "Documento", 'busca_termo': None})
if 'macrotemas_computados' not in st.session_state:
    st.session_state['macrotemas_computados'] = False

def navegar_para(novo_tipo, novo_termo): 
    st.session_state.update({'busca_tipo': novo_tipo, 'busca_termo': novo_termo})


# --- FUNÇÕES DE BACKEND (EXTRAÇÃO E BUSCA) ---
def gerar_tabela_entidades_por_macrotema(docs_macrotema, dados_totais):
    """Gera tabela de Orientadores, Co-orientadores e PKs de um Macrotema, incluindo o QL."""
    if not docs_macrotema: return
            
    tabela = []
    for d in docs_macrotema:
        nivel = d.get('nivel_academico', 'Outros')
        
        if d.get('orientador'):
            tabela.append({'Entidade': d['orientador'], 'Tipo': 'Orientador', 'Nível': nivel})
        for co in d.get('co_orientadores', []):
            tabela.append({'Entidade': co, 'Tipo': 'Co-orientador', 'Nível': nivel})
        for pk in d.get('palavras_chave', []):
            tabela.append({'Entidade': pk, 'Tipo': 'Palavra-chave', 'Nível': nivel})
            
    if not tabela: return
        
    df = pd.DataFrame(tabela)
    resumo = pd.crosstab(index=[df['Entidade'], df['Tipo']], columns=df['Nível']).reset_index()
    
    for col in ['Tese (Doutorado)', 'Dissertação (Mestrado)', 'Outros']:
        if col not in resumo.columns: resumo[col] = 0
    
    resumo = resumo.rename(columns={'Tese (Doutorado)': 'Teses', 'Dissertação (Mestrado)': 'Dissertações'})
    resumo['Total'] = resumo['Teses'] + resumo['Dissertações'] + resumo['Outros']
    
    # --- CÁLCULO DO QUOCIENTE LOCACIONAL (QL) INVERSO ---
    O_total = len(dados_totais) # Total geral (O)
    O_k = len(docs_macrotema) # Total de docs no macrotema alvo (Ok)
    
    # Contagens globais otimizadas para não travar o loop
    contagem_ori = Counter([d.get('orientador') for d in dados_totais if d.get('orientador')])
    contagem_coori = Counter([co for d in dados_totais for co in d.get('co_orientadores', [])])
    contagem_pk = Counter([pk for d in dados_totais for pk in d.get('palavras_chave', [])])
    
    ql_valores = []
    for idx, row in resumo.iterrows():
        i = row['Entidade']
        tipo = row['Tipo']
        O_ik = row['Total'] # Quantidade que este Orientador fez NESTE macrotema
        
        # Puxa o total global deste Orientador/Co/PK específico (Oi)
        if tipo == 'Orientador': O_i = contagem_ori.get(i, 0)
        elif tipo == 'Co-orientador': O_i = contagem_coori.get(i, 0)
        else: O_i = contagem_pk.get(i, 0)
        
        # Fórmula: QL = (Oik / Oi) / (Ok / O)
        if O_i == 0 or O_k == 0:
            ql = 0.0
        else:
            ql = (O_ik / O_i) / (O_k / O_total)
            
        ql_valores.append(round(ql, 2))
        
    resumo['QL (Especialização)'] = ql_valores
    
    # Ordena primeiro pelo QL para ver as maiores autoridades no topo
    resumo = resumo[['Entidade', 'Tipo', 'Teses', 'Dissertações', 'Total', 'QL (Especialização)']].sort_values(by=['Tipo', 'QL (Especialização)'], ascending=[True, False])
    
    st.write("**📊 Entidades conectadas a este Macrotema e Grau de Especialização (QL):**")
    
    resumo['Tipo'] = resumo['Tipo'].astype('category')
    max_tot = int(resumo['Total'].max()) if not resumo.empty else 100
    
    def color_ql(val):
        try:
            v = float(val)
            if v > 1: return 'color: #00FF00; font-weight: bold;'
            elif v < 1: return 'color: #FF4B4B;'
            return 'color: #F8E71C;'
        except: return ''
        
    styler = resumo.style.map(color_ql, subset=['QL (Especialização)'])
    
    st.dataframe(
        styler, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Total": st.column_config.ProgressColumn("Total", min_value=0, max_value=max_tot, format="%d"),
            "Teses": st.column_config.ProgressColumn("Teses", min_value=0, max_value=max_tot, format="%d"),
            "Dissertações": st.column_config.ProgressColumn("Dissertações", min_value=0, max_value=max_tot, format="%d"),
            "QL (Especialização)": st.column_config.NumberColumn("QL (Especialização)", format="%.2f")
        }
    )
    st.markdown("<br>", unsafe_allow_html=True)

@st.cache_data
def calcular_sna_global(dados):
    G = nx.Graph()
    for d in dados:
        doc = d.get('titulo')
        if not doc: continue
        G.add_node(doc, tipo='Documento')
        for a in d.get('autores', []): G.add_node(a, tipo='Autor'); G.add_edge(doc, a)
        ori = d.get('orientador')
        if ori: G.add_node(ori, tipo='Orientador'); G.add_edge(doc, ori)
        for pk in d.get('palavras_chave', []): G.add_node(pk, tipo='Palavra-chave'); G.add_edge(doc, pk)
        mt = d.get('macrotema')
        if mt: G.add_node(mt, tipo='Macrotema'); G.add_edge(doc, mt)

    deg_cent = nx.degree_centrality(G)
    bet_cent = nx.betweenness_centrality(G)
    close_cent = nx.closeness_centrality(G) # Novo: Closeness
    grau_abs = dict(G.degree())
    
    try: mapa_comunidades = {node: i+1 for i, comm in enumerate(nx_comm.louvain_communities(G)) for node in comm}
    except: mapa_comunidades = {}
    rank_bet = {node: rank+1 for rank, (node, _) in enumerate(sorted(bet_cent.items(), key=lambda x: x[1], reverse=True))}

    return {node: {
        'Grau Absoluto': grau_abs.get(node, 0), 
        'Degree Centrality': deg_cent.get(node, 0), 
        'Betweenness': bet_cent.get(node, 0), 
        'Closeness': close_cent.get(node, 0), # Novo
        'Comunidade': mapa_comunidades.get(node, 'N/A'), 
        'Ranking Global': rank_bet.get(node, 'N/A')
    } for node in G.nodes()}
    
@st.cache_resource
def gerar_nodos_agraph(dados_recorte, termo_foco, grau_separacao=1, metodo_tamanho="Tamanho Fixo", _sna_global=None):
    G = nx.Graph()
    for tese in dados_recorte:
        doc_id = tese['titulo']
        G.add_node(doc_id, tipo='Documento', ano=tese.get('ano', 'N/A'), nivel=tese.get('nivel_academico', 'N/A'))
        for autor in tese.get('autores', []): G.add_node(autor, tipo='Autor'); G.add_edge(autor, doc_id)
        if tese.get('orientador'): G.add_node(tese['orientador'], tipo='Orientador'); G.add_edge(tese['orientador'], doc_id)
        for co in tese.get('co_orientadores', []): G.add_node(co, tipo='Co-orientador'); G.add_edge(co, doc_id)
        for pk in tese.get('palavras_chave', []): G.add_node(pk, tipo='Palavra-chave'); G.add_edge(pk, doc_id)
        if tese.get('macrotema'): G.add_node(tese['macrotema'], tipo='Macrotema'); G.add_edge(tese['macrotema'], doc_id)

    if termo_foco not in G.nodes(): return [], []
    
    ego_G = nx.ego_graph(G, termo_foco, radius=grau_separacao)
    graus_locais = dict(ego_G.degree())
    
    if _sna_global and metodo_tamanho != "Tamanho Fixo":
        max_metrica = max([_sna_global.get(n, {}).get(metodo_tamanho, 1) for n in ego_G.nodes()])
        if max_metrica == 0: max_metrica = 1

    nodes, edges = [], []

    for node, attrs in ego_G.nodes(data=True):
        tipo = attrs.get('tipo', 'Desconhecido')
        
        if node == termo_foco:
            tam = 45 
        else:
            if metodo_tamanho == "Tamanho Fixo":
                tam = 15 + (graus_locais[node] * 1.5)
            elif _sna_global:
                valor_metrica = _sna_global.get(node, {}).get(metodo_tamanho, 0)
                tam = 10 + (valor_metrica / max_metrica) * 30
            else:
                tam = 15
                
        cor = '#FFFFFF' if node == termo_foco else ('#E74C3C' if tipo == 'Documento' else '#3498DB' if tipo == 'Autor' else '#F39C12' if tipo == 'Orientador' else '#2ECC71' if tipo == 'Palavra-chave' else '#9B59B6' if tipo == 'Macrotema' else '#95A5A6')
        formato = 'diamond' if node == termo_foco else ('star' if tipo == 'Orientador' else 'square' if tipo == 'Documento' else 'triangle' if tipo == 'Palavra-chave' else 'hexagon' if tipo == 'Macrotema' else 'dot')
        
        hover = f"Tipo: {tipo}\nGrau Local: {graus_locais[node]}"
        if _sna_global and node in _sna_global:
            hover += f"\nBetweenness Global: {_sna_global[node].get('Betweenness', 0):.4f}"
        
        rotulo = node[:25] + "..." if len(node) > 25 and tipo == 'Documento' else node
        nodes.append(Node(id=node, label=rotulo, size=tam, color=cor, title=f"{node}\n{hover}", shape=formato))

    for u, v in ego_G.edges():
        edges.append(Edge(source=u, target=v, color="#7F8C8D", width=1.0))

    return nodes, edges

# --- FUNÇÃO AUXILIAR PARA DOSSIÊ ---
def gerar_tabela_macrotemas_perfil(docs, dados_totais):
    """Gera uma tabela consolidando os macrotemas de um perfil e calcula o Quociente Locacional (QL)."""
    if not st.session_state.get('macrotemas_computados'):
        return
        
    tabela = []
    for d in docs:
        mt = d.get('macrotema', 'Multidisciplinar / Transversal')
        nivel = d.get('nivel_academico', 'Outros')
        tabela.append({'Macrotema': mt, 'Nível': nivel})
        
    if not tabela: return
        
    df = pd.DataFrame(tabela)
    resumo = pd.crosstab(df['Macrotema'], df['Nível']).reset_index()
    
    # Garante as colunas básicas
    for col in ['Tese (Doutorado)', 'Dissertação (Mestrado)', 'Outros']:
        if col not in resumo.columns: resumo[col] = 0
        
    resumo = resumo.rename(columns={'Tese (Doutorado)': 'Teses', 'Dissertação (Mestrado)': 'Dissertações'})
    resumo['Total'] = resumo['Teses'] + resumo['Dissertações'] + resumo['Outros']
    
    # --- CÁLCULO DO QUOCIENTE LOCACIONAL (QL) ---
    O_total = len(dados_totais) # Total de documentos no repositório (O)
    O_i = len(docs) # Total de documentos do perfil atual (Oi)
    contagem_global_mt = Counter([d.get('macrotema', 'Multidisciplinar / Transversal') for d in dados_totais])
    
    ql_valores = []
    for idx, row in resumo.iterrows():
        k = row['Macrotema']
        O_ik = row['Total'] # Docs do perfil (i) neste macrotema (k)
        O_k = contagem_global_mt[k] # Total de docs neste macrotema (k) globalmente
        
        # Fórmula: QL = (Oik / Oi) / (Ok / O)
        if O_i == 0 or O_k == 0:
            ql = 0.0
        else:
            ql = (O_ik / O_i) / (O_k / O_total)
            
        ql_valores.append(round(ql, 2))
        
    resumo['QL (Especialização)'] = ql_valores
    
    # Reordenando para focar nos temas onde o perfil é MAIS especializado
    resumo = resumo[['Macrotema', 'Teses', 'Dissertações', 'Outros', 'Total', 'QL (Especialização)']].sort_values(by='QL (Especialização)', ascending=False)
    
    st.write("**📊 Frequência Temática e Especialização (QL):**")
    
    resumo['Macrotema'] = resumo['Macrotema'].astype('category')
    max_tot = int(resumo['Total'].max()) if not resumo.empty else 100
    
    def color_ql2(val):
        try:
            v = float(val)
            if v > 1: return 'color: #00FF00; font-weight: bold;'
            elif v < 1: return 'color: #FF4B4B;'
            return 'color: #F8E71C;'
        except: return ''
        
    styler2 = resumo.style.map(color_ql2, subset=['QL (Especialização)'])
    
    st.dataframe(
        styler2, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Total": st.column_config.ProgressColumn("Total", min_value=0, max_value=max_tot, format="%d"),
            "Teses": st.column_config.ProgressColumn("Teses", min_value=0, max_value=max_tot, format="%d"),
            "Dissertações": st.column_config.ProgressColumn("Dissertações", min_value=0, max_value=max_tot, format="%d"),
            "QL (Especialização)": st.column_config.NumberColumn("QL (Especialização)", format="%.2f")
        }
    )
    st.caption("*Nota: QL > 1 indica que a entidade é estatisticamente especializada neste macrotema em relação à média geral da universidade.*")
    st.markdown("<br>", unsafe_allow_html=True)

def renderizar_nuvem_interativa_html(word_freq_dict):
    data_js = json.dumps([{"name": k, "value": v} for k, v in word_freq_dict.items()])
    return f"""
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
            var option = {{ tooltip: {{ show: true }}, series: [{{ type: 'wordCloud', shape: 'circle', sizeRange: [14, 70], textStyle: {{ fontFamily: 'sans-serif', fontWeight: 'bold', color: function () {{ return 'rgb(' + [Math.round(Math.random() * 150 + 100), Math.round(Math.random() * 150 + 100), Math.round(Math.random() * 150 + 100)].join(',') + ')'; }} }}, data: {data_js} }}] }};
            chart.setOption(option);
            window.onresize = chart.resize;
        </script>
    </body>
    </html>
    """


# --- MOTOR DE SÍNTESE SOB DEMANDA (APP) ---
@st.cache_data(show_spinner=False)
def gerar_descritivo_sessao(nomes_programas, amostra_textos, api_key):
    """Gera o descritivo na hora e salva em cache para economizar requisições."""
    genai.configure(api_key=api_key)
    try: model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception: model = genai.GenerativeModel('gemini-2.0-flash')

    nomes_str = ", ".join(nomes_programas)
    prompt = f"""Você é um analista sênior de avaliação acadêmica.
Sua missão é criar um parágrafo descritivo e direto (máximo de 60 palavras) apresentando o perfil de pesquisa e o ecossistema do(s) seguinte(s) programa(s): {nomes_str}.

Utilize a amostra de teses/conceitos abaixo como base:
---
{amostra_textos}
---

Diretrizes rigorosas:
- Comece direto com a descrição.
- Sintetize os grandes domínios de conhecimento baseando-se nos documentos e no seu conhecimento prévio.
- NÃO repita o nome do(s) programa(s) no texto.
- Retorne APENAS o parágrafo limpo.
"""
    try:
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.3))
        return response.text.strip().replace('**', '').replace('"', '')
    except Exception as e:
        return f"Não foi possível gerar a síntese dinâmica no momento. (Aviso: {e})"




import requests

# --- MOTOR DE CONSULTA À CAPES SUCUPIRA ---
@st.cache_data(show_spinner=False, ttl=86400) # Mantém em cache por 24 horas
def carregar_catalogo_capes_ufsc():
    """Baixa todos os programas da UFSC de uma vez e cria um dicionário para busca instantânea."""
    url = "https://apigw-proxy.capes.gov.br/observatorio/data/observatorio/ppg"
    headers = {"Accept": "application/json"}
    programas_capes = {}
    page = 0
    
    while True:
        parametros = {"query": "id-ies:(4362)", "page": page, "size": 100}
        try:
            resposta = requests.get(url, params=parametros, headers=headers, timeout=10)
            if resposta.status_code != 200: break
            
            dados = resposta.json()
            resultados = dados if isinstance(dados, list) else dados.get('content', dados.get('data', []))
            if not resultados: break
            
            for ppg in resultados:
                nome_capes = ppg.get("nome", "").strip().upper()
                # Remove acentos para facilitar o match cruzado
                nome_norm = ''.join(c for c in unicodedata.normalize('NFD', nome_capes) if unicodedata.category(c) != 'Mn')
                
                programas_capes[nome_norm] = {
                    "Nome": ppg.get("nome", "Não informado"),
                    "Código": ppg.get("codigo", "Não informado"),
                    "Nota": ppg.get("conceito", "Não informado"),
                    "Grande Área": ppg.get("nomeGrandeAreaConhecimento", "Não informado"),
                    "Área de Avaliação": ppg.get("nomeAreaAvaliacao", "Não informado"),
                    "Área de Conhecimento": ppg.get("nomeAreaConhecimento", "Não informado"),
                    "Modalidade": ppg.get("modalidade", "Não informado"),
                    "Situação": ppg.get("situacao", "Não informado"),
                    "Modalidade de Ensino": ppg.get("nomeModalidadeEnsino", "Não informado"),
                    "Grau Acadêmico": ppg.get("grau", "Não informado")
                }
            
            if len(resultados) < 100: break
            page += 1
        except Exception:
            break
            
    return programas_capes


# --- CARREGAMENTO E SELEÇÃO DA BASE CONSOLIDADA ---
@st.cache_data
def carregar_base_consolidada():
    try:
        # Lê o arquivo GZIP diretamente da memória ('rt' = Read Text)
        with gzip.open('base_consolidada_ufsc.json.gz', 'rt', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# --- FUNÇÃO AUXILIAR PARA O CATÁLOGO LEVE ---
@st.cache_data
def carregar_catalogo_programas():
    try:
        # Lê apenas o arquivo leve (alguns KBs) para montar o menu rapidamente
        with open('programas_ufsc.json', 'r', encoding='utf-8') as f: 
            return json.load(f)
    except FileNotFoundError:
        st.error("⚠️ Ficheiro 'programas_ufsc.json' não encontrado na raiz.")
        return {}

# --- TELA DE SELEÇÃO INICIAL (CARREGAMENTO PREGUIÇOSO / LAZY LOADING) ---
if 'dados_completos' not in st.session_state or st.session_state.get('recarregar'):
    st.title("🔌 Seleção de Programas (PPGs)")
    st.markdown("A base consolidada com os Macrotemas está pronta. Para otimizar a memória e a velocidade da rede, selecione os programas que deseja analisar antes de carregar os dados.")
    
    catalogo_leve = carregar_catalogo_programas()
    programas_disponiveis = sorted(list(catalogo_leve.keys()))
    
    programas_selecionados = st.multiselect("Selecione um ou mais Programas de Pós-Graduação:", programas_disponiveis)
    
    if st.button("Carregar Dados e Iniciar Análise", type="primary"):
        if programas_selecionados:
            with st.spinner("Lendo a base consolidada e filtrando os PPGs selecionados. Isso levará apenas alguns segundos..."):
                try:
                    # O arquivo pesado de +100MB só é lido AQUI, após o clique
                    with gzip.open('base_consolidada_ufsc.json.gz', 'rt', encoding='utf-8') as f:
                        base_total = json.load(f)
                        
                    # Filtra mantendo apenas os programas escolhidos
                    dados_filtrados = [d for d in base_total if d.get('programa_origem') in programas_selecionados]
                    
                    if not dados_filtrados:
                        st.warning("Nenhum documento encontrado para os PPGs selecionados. Talvez o pipeline ainda não os tenha extraído.")
                        st.stop()
                        
                    # Salva na sessão e destrava o aplicativo
                    st.session_state['dados_completos'] = dados_filtrados
                    st.session_state['programas_selecionados_lista'] = programas_selecionados
                    st.session_state['nome_programa'] = f"{len(programas_selecionados)} PPG(s) Selecionado(s): {', '.join(programas_selecionados)}"
                    st.session_state['macrotemas_computados'] = True
                    st.session_state['recarregar'] = False
                    st.rerun()
                    
                except FileNotFoundError:
                    st.error("O arquivo 'base_consolidada_ufsc.json.gz' não foi encontrado. Por favor, rode o pipeline de extração primeiro.")
                    st.stop()
        else:
            st.warning("Por favor, selecione pelo menos um programa para continuar.")
            
    # Trava a execução do resto do app (SNA, gráficos, etc.) até o usuário passar desta tela
    st.stop()

# --- DASHBOARD PRINCIPAL ---
dados_completos = st.session_state['dados_completos']
st.title("🌌 Ecologia do Conhecimento")
st.subheader(f"Base: {st.session_state['nome_programa']}")

# Botão na barra lateral para voltar e escolher outros PPGs
if st.sidebar.button("🔄 Escolher outros PPGs", type="primary"):
    st.session_state['recarregar'] = True
    st.rerun()


# KPIs Básicos
autores_set = set([a for d in dados_completos for a in d.get('autores', [])])
orientadores_set = set([d.get('orientador') for d in dados_completos if d.get('orientador')])
coorientadores_set = set([co for d in dados_completos for co in d.get('co_orientadores', [])])
keywords_set = set([kw for d in dados_completos for kw in d.get('palavras_chave', [])])

c1, c2, c3 = st.columns(3)
c1.metric("📄 Documentos Totais", len(dados_completos))
c2.metric("🎓 Teses (Doutorado)", len([d for d in dados_completos if "Tese" in d.get('nivel_academico', '')]))
c3.metric("📜 Dissertações", len([d for d in dados_completos if "Disserta" in d.get('nivel_academico', '')]))

c4, c5, c6, c7 = st.columns(4)
c4.metric("✍️ Autores Únicos", len(autores_set))
c5.metric("🏫 Orientadores", len(orientadores_set))
c6.metric("🤝 Co-orientadores", len(coorientadores_set))
c7.metric("💡 Conceitos (Keywords)", len(keywords_set))


# --- APRESENTAÇÃO DO PERFIL DINÂMICO ---

try:
    api_key_app = st.secrets["GEMINI_API_KEY"]
    
    # Extrai até 25 documentos espaçados uniformemente para criar uma amostra representativa
    amostra_docs = []
    salto = max(1, len(dados_completos) // 25)
    for i in range(0, len(dados_completos), salto):
        d = dados_completos[i]
        amostra_docs.append(f"- {d.get('titulo', '')} | {', '.join(d.get('palavras_chave', []))}")
        if len(amostra_docs) >= 25: break
            
    nomes_ppgs = list(set([d.get('programa_origem', 'Programa Desconhecido') for d in dados_completos if d.get('programa_origem')]))
    
        
except KeyError:
    st.warning("🔑 Chave da API do Gemini não configurada nos secrets locais. O perfil dinâmico está desativado.")
st.markdown("<br>", unsafe_allow_html=True)



if len(st.session_state.get('programas_selecionados_lista', [])) > 1:
    st.markdown("---")
    st.subheader("📊 Comparativo entre PPGs")
    
    comparativo_data = []
    
    from collections import defaultdict
    dados_por_ppg = defaultdict(list)
    for d in dados_completos:
        ppg = d.get('programa_origem', 'Desconhecido')
        dados_por_ppg[ppg].append(d)
        
    for ppg, docs_ppg in dados_por_ppg.items():
        comparativo_data.append({
            "PPG": ppg,
            "📄 Documentos Totais": len(docs_ppg),
            "🎓 Teses (Doutorado)": len([d for d in docs_ppg if "Tese" in d.get('nivel_academico', '')]),
            "📜 Dissertações": len([d for d in docs_ppg if "Disserta" in d.get('nivel_academico', '')]),
            "✍️ Autores Únicos": len(set([a for d in docs_ppg for a in d.get('autores', [])])),
            "🏫 Orientadores": len(set([d.get('orientador') for d in docs_ppg if d.get('orientador')])),
            "🤝 Co-orientadores": len(set([co for d in docs_ppg for co in d.get('co_orientadores', [])])),
            "💡 Conceitos (Keywords)": len(set([kw for d in docs_ppg for kw in d.get('palavras_chave', [])]))
        })
        
    df_comp = pd.DataFrame(comparativo_data)
    df_melted = df_comp.melt(id_vars="PPG", var_name="Métrica", value_name="Quantidade")
    
    fig_comp = px.bar(
        df_melted, 
        x="PPG", 
        y="Quantidade", 
        color="PPG", 
        facet_col="Métrica", 
        facet_col_wrap=4, 
        template="plotly_dark", 
        text="Quantidade",
        height=650
    )
    fig_comp.update_yaxes(matches=None, showticklabels=False, title="") 
    fig_comp.update_xaxes(showticklabels=False, title="")
    fig_comp.update_traces(textposition='outside')
    fig_comp.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    
    st.plotly_chart(fig_comp, use_container_width=True)


# --- APRESENTAÇÃO DO PERFIL E DADOS OFICIAIS ---
st.markdown("#### 🏛️ Ficha Técnica e Perfil Institucional")

nomes_ppgs = list(set([d.get('programa_origem', 'Programa Desconhecido') for d in dados_completos if d.get('programa_origem')]))

# Puxa o dicionário completo da CAPES da memória (Instantâneo)
catalogo_capes = carregar_catalogo_capes_ufsc()

for nome_ppg in nomes_ppgs:
    # 1. Limpeza agressiva do nome do repositório
    nome_limpo = nome_ppg.replace("Programa de Pós-Graduação em ", "").replace("Programa de Pós-Graduação ", "").replace("PPG em ", "").strip().upper()
    nome_norm_busca = ''.join(c for c in unicodedata.normalize('NFD', nome_limpo) if unicodedata.category(c) != 'Mn')
    
    # 2. Tenta encontrar correspondência exata primeiro (o mais rápido)
    dados_capes = catalogo_capes.get(nome_norm_busca)
    
    # 3. Inteligência de Strings (Fuzzy Matching) para lidar com variações da UFSC/CAPES
    if not dados_capes:
        import difflib # Biblioteca nativa do Python para cálculo de similaridade
        
        chaves_disponiveis = list(catalogo_capes.keys())
        # Procura a chave mais parecida com pelo menos 65% de similaridade estrutural
        melhores_matches = difflib.get_close_matches(nome_norm_busca, chaves_disponiveis, n=1, cutoff=0.65)
        
        if melhores_matches:
            dados_capes = catalogo_capes[melhores_matches[0]]
        else:
            # 4. Fallback Final: Interseção de Palavras-Chave (Ignorando preposições curtas)
            palavras_busca = set([p for p in nome_norm_busca.split() if len(p) > 2])
            for key_capes, dados in catalogo_capes.items():
                palavras_capes = set([p for p in key_capes.split() if len(p) > 2])
                
                # Se houver um cruzamento muito forte das palavras principais (ex: ENGENHARIA, GESTAO, CONHECIMENTO)
                if len(palavras_busca.intersection(palavras_capes)) >= max(1, len(palavras_busca) - 1):
                    dados_capes = dados
                    break

    # 5. Desenha o Cartão Oficial da CAPES
    if dados_capes:
        st.markdown(f"**{dados_capes['Nome']} ({dados_capes['Código']}) | Nota CAPES: {dados_capes['Nota']}**")

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Grande Área:** {dados_capes['Grande Área']}")
            st.write(f"**Área de Avaliação:** {dados_capes['Área de Avaliação']}")
            st.write(f"**Área de Conhecimento:** {dados_capes['Área de Conhecimento']}")
        with col2:
            st.write(f"**Modalidade:** {dados_capes['Modalidade']} ({dados_capes['Grau Acadêmico']})")
            st.write(f"**Situação:** {dados_capes['Situação']}")
            st.write(f"**Ensino:** {dados_capes['Modalidade de Ensino']}")
    else:
        st.markdown(f"**{nome_ppg}**")
        st.caption("⚠️ Dados oficiais não localizados na base da CAPES (Possível variação de nomenclatura institucional).")
        
st.markdown("<br>", unsafe_allow_html=True)

# 4. Descritivo Dinâmico da IA (A Alma do Programa)
try:
    api_key_app = st.secrets["GEMINI_API_KEY"]
    
    amostra_docs = []
    salto = max(1, len(dados_completos) // 25)
    for i in range(0, len(dados_completos), salto):
        d = dados_completos[i]
        amostra_docs.append(f"- {d.get('titulo', '')} | {', '.join(d.get('palavras_chave', []))}")
        if len(amostra_docs) >= 25: break
            
    with st.spinner("A IA está analisando a amostra de documentos para sintetizar o perfil epistemológico..."):
        descritivo_dinamico = gerar_descritivo_sessao(tuple(nomes_ppgs), "\n".join(amostra_docs), api_key_app)
        st.info(f"**Síntese de Pesquisa do PPG:** {descritivo_dinamico}")
        
except KeyError:
    st.warning("🔑 Chave da API do Gemini não configurada nos secrets locais.")
st.markdown("---")


# IMPORTANTE: Movemos o cálculo SNA para cá para alimentar a nova Super-Tabela
sna_global = calcular_sna_global(dados_completos)

# (Módulo de visualização de Macrotemas movido para a secção inferior)


# --- MOTOR DE BUSCA (EGO-GRAPH) ---
st.header("🔍 Motor de Busca e Dossiê")

opcoes_busca = ["Documento", "Autor", "Orientador", "Co-orientador", "Palavra-chave"]
if st.session_state['macrotemas_computados']:
    opcoes_busca.append("Macrotema")

tipo_busca = st.radio("Procurar por Entidade:", opcoes_busca, horizontal=True, key="busca_tipo")

if tipo_busca == "Documento": opcoes = [d['titulo'] for d in dados_completos]
elif tipo_busca == "Autor": opcoes = list(autores_set)
elif tipo_busca == "Orientador": opcoes = list(orientadores_set)
elif tipo_busca == "Co-orientador": opcoes = list(coorientadores_set)
elif tipo_busca == "Palavra-chave": opcoes = list(keywords_set)
elif tipo_busca == "Macrotema": opcoes = list(set([d.get('macrotema') for d in dados_completos if d.get('macrotema')]))

if st.session_state['busca_termo'] not in opcoes: st.session_state['busca_termo'] = None
termo_selecionado = st.selectbox("Selecione:", sorted(opcoes), index=sorted(opcoes).index(st.session_state['busca_termo']) if st.session_state['busca_termo'] in opcoes else None, placeholder="Pesquise aqui...")

if termo_selecionado != st.session_state['busca_termo']:
    st.session_state['busca_termo'] = termo_selecionado
    st.rerun()

termo_ativo = st.session_state['busca_termo']

if termo_ativo:
    col_info, col_sna = st.columns([2, 1])
    with col_info:
        st.info(f"**{termo_ativo}**")
        
        if tipo_busca == "Documento":
            doc = next((d for d in dados_completos if d['titulo'] == termo_ativo), {})
            
            st.write(f"**Ano:** {doc.get('ano', 'N/A')} | **Nível:** {doc.get('nivel_academico', 'N/A')} | **Programa:** {doc.get('programa_origem', 'N/A')}")
            
            if doc.get('url'):
                st.markdown(f"🔗 **Link Oficial na UFSC:** [{doc['url']}]({doc['url']})")
                
            if st.session_state['macrotemas_computados']:
                tema = doc.get('macrotema', 'Multidisciplinar / Transversal')
                st.write("**Macrotema Classificado:**")
                st.button(f"🏷️ {tema}", on_click=navegar_para, args=("Macrotema", tema))
                
            st.write("**Rede de Autoria e Orientação:**")
            for a in doc.get('autores', []): 
                st.button(f"👤 {a}", on_click=navegar_para, args=("Autor", a))
            if doc.get('orientador'): 
                st.button(f"🏫 {doc['orientador']}", on_click=navegar_para, args=("Orientador", doc['orientador']))
            for co in doc.get('co_orientadores', []):
                st.button(f"🤝 {co}", on_click=navegar_para, args=("Co-orientador", co))
                
            st.write("**Palavras-chave:**")
            for pk in doc.get('palavras_chave', []): 
                st.button(f"💡 {pk}", on_click=navegar_para, args=("Palavra-chave", pk))
                
            with st.expander("Ler Resumo (Abstract)"): 
                st.write(doc.get('resumo', 'Resumo não disponível.'))
                
        elif tipo_busca == "Autor":
            docs = [d for d in dados_completos if termo_ativo in d.get('autores', [])]
            
            programas = sorted(list(set([d.get('programa_origem') for d in docs if d.get('programa_origem')])))
            if programas:
                st.write(f"**🏛️ Programas (PPG):** {', '.join(programas)}")
            
            orientadores = set()
            co_orientadores = set()
            for d in docs:
                if d.get('orientador'):
                    orientadores.add(d['orientador'])
                for co in d.get('co_orientadores', []):
                    co_orientadores.add(co)
            
            if orientadores or co_orientadores:
                st.write("**👨‍🏫 Orientadores e Co-orientadores:**")
                for ori in sorted(list(orientadores)):
                    st.button(f"🏫 Orientador: {ori}", key=f"btn_ori_aut_{abs(hash(ori))}", on_click=navegar_para, args=("Orientador", ori))
                for co in sorted(list(co_orientadores)):
                    st.button(f"🤝 Co-orientador: {co}", key=f"btn_co_aut_{abs(hash(co))}", on_click=navegar_para, args=("Co-orientador", co))
                
                st.markdown("<br>", unsafe_allow_html=True)

            st.write(f"**Documentos Escritos ({len(docs)}):**")
            for i, d in enumerate(docs): st.button(f"📄 {d['titulo']}", key=f"btn_aut_{i}", on_click=navegar_para, args=("Documento", d['titulo']))
            
        elif tipo_busca in ["Orientador", "Co-orientador"]:
            if tipo_busca == "Orientador":
                docs = [d for d in dados_completos if d.get('orientador') == termo_ativo]
            else:
                docs = [d for d in dados_completos if termo_ativo in d.get('co_orientadores', [])]
                
            programas = sorted(list(set([d.get('programa_origem') for d in docs if d.get('programa_origem')])))
            if programas:
                st.write(f"**🏛️ Programas (PPG):** {', '.join(programas)}")
                
            gerar_tabela_macrotemas_perfil(docs, dados_completos)
            
            alunos_orientados = set()
            for d in docs:
                for autor in d.get('autores', []):
                    alunos_orientados.add(autor)
            alunos_orientados = sorted(list(alunos_orientados))
            
            st.write(f"**🎓 Alunos {'Orientados' if tipo_busca == 'Orientador' else 'Co-orientados'} ({len(alunos_orientados)}):**")
            with st.expander(f"Ver lista de {len(alunos_orientados)} alunos"):
                for i, aluno in enumerate(alunos_orientados):
                    st.button(f"👤 {aluno}", key=f"btn_aluno_{tipo_busca}_{abs(hash(aluno))}_{i}", on_click=navegar_para, args=("Autor", aluno))
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.write(f"**Documentos {'Orientados' if tipo_busca == 'Orientador' else 'Co-orientados'} ({len(docs)}):**")
            
            # Agrupar documentos do Orientador por Macrotema
            from collections import defaultdict
            docs_por_mt = defaultdict(list)
            for d in docs:
                docs_por_mt[d.get('macrotema', 'Multidisciplinar / Transversal')].append(d)
                
            with st.expander(f"📚 Ver lista de {len(docs)} documentos"):
                for mt, docs_mt in docs_por_mt.items():
                    st.markdown(f"**🏷️ {mt}**")
                    for i, d in enumerate(docs_mt):
                        # Chave criptográfica infalível usando hash do título inteiro
                        chave_unica = f"btn_{tipo_busca}_{abs(hash(d['titulo']))}_{i}"
                        st.button(f"📄 {d['titulo']}", key=chave_unica, on_click=navegar_para, args=("Documento", d['titulo']))
            
        elif tipo_busca == "Palavra-chave":
            docs = [d for d in dados_completos if termo_ativo in d.get('palavras_chave', [])]
            gerar_tabela_macrotemas_perfil(docs, dados_completos)
            
            with st.expander(f"📚 Ver Lista Completa de Documentos Associados ({len(docs)})"):
                for i, d in enumerate(docs): 
                    chave_unica = f"btn_pk_{abs(hash(d['titulo']))}_{i}"
                    st.button(f"📄 {d['titulo']}", key=chave_unica, on_click=navegar_para, args=("Documento", d['titulo']))
            
        elif tipo_busca == "Macrotema":
            docs = [d for d in dados_completos if d.get('macrotema') == termo_ativo]
            gerar_tabela_entidades_por_macrotema(docs, dados_completos)
            
            with st.expander(f"📚 Explorar Teses e Dissertações da Categoria ({len(docs)})"):
                for i, d in enumerate(docs): 
                    chave_unica = f"btn_mt_{abs(hash(d['titulo']))}_{i}"
                    st.button(f"📄 {d['titulo']}", key=chave_unica, on_click=navegar_para, args=("Documento", d['titulo']))
                    
    with col_sna:
        metricas = sna_global.get(termo_ativo, {})
        if metricas:
            st.success(f"Cluster: {metricas.get('Comunidade')} | Rank: #{metricas.get('Ranking Global')}")
            st.metric("Grau (Conexões)", metricas.get('Grau Absoluto'))
            st.metric("Betweenness", f"{metricas.get('Betweenness', 0):.4f}")
            st.metric("Closeness", f"{metricas.get('Closeness', 0):.4f}") 
            
    if tipo_busca in ["Orientador", "Co-orientador", "Palavra-chave", "Macrotema"] and 'docs' in locals() and docs:
        
        titulo_secao = "Orientações" if tipo_busca in ["Orientador", "Co-orientador"] else "Documentos Associados"
        st.markdown(f"### 📈 Evolução Histórica ({titulo_secao})")
        
        tem_multiplos_ppgs = len(st.session_state.get('programas_selecionados_lista', [])) > 1
        cols_graf = st.columns(4) if tem_multiplos_ppgs else st.columns(3)
        agrupar_niveis = cols_graf[0].radio("Visão dos Níveis:", ["Separar Teses e Dissertações", "Agrupar tudo (Total)"], horizontal=True, key="agrup_niv_perfil")
        modo_analise = cols_graf[1].radio("Modo de Análise:", ["Visão Geral (Volume)", "Análise por Macrotemas"], horizontal=True, key="modo_ana_perfil")
        tipo_grafico = cols_graf[2].radio("Tipo de Gráfico:", ["Barras", "Linhas"], horizontal=True, key="tipo_graf_perfil")
        
        separar_ppg_hist = False
        if tem_multiplos_ppgs:
            separar_ppg_hist = cols_graf[3].radio("Separar por PPG:", ["Não", "Sim"], horizontal=True, key="agrup_ppg_perfil") == "Sim"
        
        df_docs = pd.DataFrame(docs)
        if not df_docs.empty and 'ano' in df_docs.columns:
            df_docs['ano'] = pd.to_numeric(df_docs['ano'], errors='coerce')
            df_docs = df_docs.dropna(subset=['ano'])
            df_docs['ano'] = df_docs['ano'].astype(int)
            
            if not df_docs.empty:
                if 'nivel_academico' not in df_docs.columns:
                    df_docs['nivel_academico'] = 'Outros'
                else:
                    df_docs['nivel_academico'] = df_docs['nivel_academico'].fillna('Outros')
                    
                if 'macrotema' not in df_docs.columns:
                    df_docs['macrotema'] = 'Multidisciplinar / Transversal'
                else:
                    df_docs['macrotema'] = df_docs['macrotema'].fillna('Multidisciplinar / Transversal')
                
                graf_func = px.bar if tipo_grafico == "Barras" else px.line
                barmode_kw = dict(barmode='stack') if tipo_grafico == "Barras" else dict()
                marker_kw = dict() if tipo_grafico == "Barras" else dict(markers=True)
                
                label_y = "Orientações" if tipo_busca in ["Orientador", "Co-orientador"] else "Documentos"
                
                # Configura facet_col se separar_ppg_hist for verdadeiro
                facet_kws = dict(facet_col='programa_origem', facet_col_wrap=2) if separar_ppg_hist else dict()
                groupby_cols = ['ano']
                if separar_ppg_hist:
                    groupby_cols.append('programa_origem')
                    df_docs['programa_origem'] = df_docs['programa_origem'].fillna('Desconhecido')
                
                if modo_analise == "Visão Geral (Volume)":
                    if agrupar_niveis == "Agrupar tudo (Total)":
                        df_plot = df_docs.groupby(groupby_cols).size().reset_index(name='Volume')
                        fig = graf_func(df_plot, x='ano', y='Volume', title=f"{label_y} por Ano (Total)", template="plotly_dark", **marker_kw, **facet_kws)
                    else:
                        df_plot = df_docs.groupby(groupby_cols + ['nivel_academico']).size().reset_index(name='Volume')
                        fig = graf_func(df_plot, x='ano', y='Volume', color='nivel_academico', title=f"{label_y} por Ano e Nível Acadêmico", template="plotly_dark", **barmode_kw, **marker_kw, **facet_kws)
                else:
                    if agrupar_niveis == "Agrupar tudo (Total)":
                        df_plot = df_docs.groupby(groupby_cols + ['macrotema']).size().reset_index(name='Volume')
                        fig = graf_func(df_plot, x='ano', y='Volume', color='macrotema', title=f"{label_y} por Ano e Macrotema", template="plotly_dark", **barmode_kw, **marker_kw, **facet_kws)
                    else:
                         df_docs['Nível/Tema'] = df_docs['nivel_academico'] + " - " + df_docs['macrotema']
                         df_plot = df_docs.groupby(groupby_cols + ['Nível/Tema']).size().reset_index(name='Volume')
                         fig = graf_func(df_plot, x='ano', y='Volume', color='Nível/Tema', title=f"{label_y} por Ano, Nível e Macrotema", template="plotly_dark", **barmode_kw, **marker_kw, **facet_kws)
                
                fig.update_layout(xaxis_title="Ano", yaxis_title="Quantidade", xaxis=dict(tickmode='linear', dtick=1))
                # Ajuste para facet annotations se houver
                if separar_ppg_hist:
                    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                st.plotly_chart(fig, use_container_width=True)
                
        st.markdown(f"### ☁️ Nuvem de Palavras ({titulo_secao})")
        
        col_nuvem1, col_nuvem2 = st.columns(2)
        modo_nuvem = col_nuvem1.selectbox("Fonte de Dados para Nuvem:", ["Tudo Combinado (Título + Resumo + Palavras-Chave)", "Apenas Palavras-chave", "Apenas Título", "Apenas Resumo"])
        separar_nuvem_ppg = False
        if tem_multiplos_ppgs:
            separar_nuvem_ppg = col_nuvem2.radio("Separar Nuvem por PPG:", ["Não", "Sim"], horizontal=True, key="sep_nuvem_ppg_perfil") == "Sim"

        stopwords = set(['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'uma', 'para', 'com', 'não', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'das', 'à', 'seu', 'sua', 'ou', 'nos', 'já', 'eu', 'também', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'sem', 'mesmo', 'aos', 'nas', 'me', 'esse', 'essa', 'num', 'nem', 'numa', 'pelos', 'pelas', 'este', 'esta', 'sobre', 'estudo', 'análise', 'proposta', 'uso', 'aplicação', 'desenvolvimento', 'modelo', 'sistema', 'avaliação', 'gestão', 'conhecimento', 'engenharia', 'objetivo', 'pesquisa', 'trabalho', 'resultados', 'método', 'foi', 'foram', 'são', 'ser', 'através', 'forma', 'apresenta', 'the', 'of', 'and', 'in', 'to', 'a', 'is', 'for', 'by', 'on', 'with', 'an', 'as', 'this', 'that', 'which', 'from', 'it', 'or', 'be', 'are', 'at', 'has', 'have', 'was', 'were', 'not', 'but', 'by'])
        
        def extrair_texto_docs(lista_docs):
            texto_completo = []
            for d in lista_docs:
                if "Resumo" in modo_nuvem or "Tudo Combinado" in modo_nuvem:
                    texto_completo.append(d.get('resumo', ''))
                if "Título" in modo_nuvem or "Tudo Combinado" in modo_nuvem:
                    texto_completo.append(d.get('titulo', ''))
                if "Palavras-chave" in modo_nuvem or "Tudo Combinado" in modo_nuvem:
                    texto_completo.append(" ".join(d.get('palavras_chave', [])))
            texto_str = " ".join([str(t) for t in texto_completo]).lower()
            return re.sub(r'[^\w\s]', '', texto_str)

        if separar_nuvem_ppg:
            docs_por_ppg = {}
            for d in docs:
                ppg = d.get('programa_origem', 'Desconhecido')
                if ppg not in docs_por_ppg:
                    docs_por_ppg[ppg] = []
                docs_por_ppg[ppg].append(d)
                
            abas_ppg = st.tabs(list(docs_por_ppg.keys()))
            for idx, ppg in enumerate(docs_por_ppg.keys()):
                with abas_ppg[idx]:
                    texto_limpo = extrair_texto_docs(docs_por_ppg[ppg])
                    palavras_nuvem = [p for p in texto_limpo.split() if p not in stopwords and len(p) > 2]
                    if palavras_nuvem:
                        freq_dict = dict(Counter(palavras_nuvem).most_common(100))
                        html_nuvem = renderizar_nuvem_interativa_html(freq_dict)
                        components.html(html_nuvem, height=480, scrolling=False)
                    else:
                        st.info("Palavras insuficientes para gerar a nuvem neste PPG.")
        else:
            texto_limpo = extrair_texto_docs(docs)
            palavras_nuvem = [p for p in texto_limpo.split() if p not in stopwords and len(p) > 2]
            if palavras_nuvem:
                freq_dict = dict(Counter(palavras_nuvem).most_common(100))
                html_nuvem = renderizar_nuvem_interativa_html(freq_dict)
                components.html(html_nuvem, height=480, scrolling=False)
            else:
                st.info("Palavras insuficientes para gerar a nuvem.")

  
    st.markdown("### 🌌 Órbita de Relacionamentos")
    
    col_orb1, col_orb2 = st.columns(2)
    grau_expansao = col_orb1.slider("Expansão do Grafo (Camadas de Profundidade):", 1, 3, 1)
    metodo_tamanho_ego = col_orb2.selectbox("Tamanho dos Nós na Órbita:", ["Tamanho Fixo", "Grau Absoluto", "Degree Centrality", "Betweenness"])
    
    with st.spinner("A mapear o ecossistema local em 3D/2D..."):
        nodes, edges = gerar_nodos_agraph(dados_completos, termo_ativo, grau_expansao, metodo_tamanho_ego, sna_global)
        
        if nodes and edges:
            config = Config(width="100%", height=600, directed=False, physics=True, hierarchical=False, nodeHighlightBehavior=True, highlightColor="#F1C40F", collapsible=False)
            retorno_clique = agraph(nodes=nodes, edges=edges, config=config)
            
            if retorno_clique and retorno_clique != termo_ativo:
                st.info(f"💡 Clicou no nó: **{retorno_clique}**. Pesquise por ele para ver os detalhes completos!")
        else:
            st.warning("Não foi possível gerar a órbita visual para este termo.")

st.markdown("---")

# --- MÓDULO DE MACROTEMAS ---
st.header("🧠 Análise Temática Estrutural")

sna_global = calcular_sna_global(dados_completos)

# --- NOVA TABELA ROBUSTA DE MACROTEMAS ---
O_total = len(dados_completos)
contagem_ori = Counter([d.get('orientador') for d in dados_completos if d.get('orientador')])
contagem_coori = Counter([co for d in dados_completos for co in d.get('co_orientadores', [])])

linhas_tabela = []
macrotemas_unicos = set([d.get('macrotema', 'Multidisciplinar / Transversal') for d in dados_completos])

for mt in macrotemas_unicos:
    docs_mt = [d for d in dados_completos if d.get('macrotema', 'Multidisciplinar / Transversal') == mt]
    O_k = len(docs_mt)
    
    teses = sum(1 for d in docs_mt if 'Tese' in d.get('nivel_academico', ''))
    dissertacoes = sum(1 for d in docs_mt if 'Disserta' in d.get('nivel_academico', ''))
    
    anos = [int(d['ano']) for d in docs_mt if d.get('ano') and str(d['ano']).isdigit()]
    ano_antigo = min(anos) if anos else "-"
    ano_recente = max(anos) if anos else "-"
    ano_modal = Counter(anos).most_common(1)[0][0] if anos else "-"
    
    def top_ql(entidades_na_mt, contagem_global):
        max_ql = -1
        top_ent = "-"
        contagem_local = Counter(entidades_na_mt)
        for ent, O_ik in contagem_local.items():
            O_i = contagem_global.get(ent, 0)
            if O_i > 0 and O_k > 0:
                ql = (O_ik / O_i) / (O_k / O_total)
                if ql > max_ql:
                    max_ql = ql
                    top_ent = ent
                elif ql == max_ql:
                    if O_ik > contagem_local.get(top_ent, 0): top_ent = ent
        return top_ent, max_ql

    oris_mt = [d.get('orientador') for d in docs_mt if d.get('orientador')]
    top_ori, ql_ori = top_ql(oris_mt, contagem_ori)
    
    cooris_mt = [co for d in docs_mt for co in d.get('co_orientadores', [])]
    top_coori, ql_coori = top_ql(cooris_mt, contagem_coori)
    
    mt_sna = sna_global.get(mt, {})
    
    linhas_tabela.append({
        "Macrotema": mt,
        "Docs": O_k,
        "Teses": teses,
        "Dissertações": dissertacoes,
        "Grau": mt_sna.get('Grau Absoluto', 0),
        "Betweenness": round(mt_sna.get('Betweenness', 0.0), 4),
        "Closeness": round(mt_sna.get('Closeness', 0.0), 4),
        "Especialista (Orientador)": f"{top_ori} (QL: {round(ql_ori,1)})" if top_ori != "-" else "-",
        "Especialista (Co-orientador)": f"{top_coori} (QL: {round(ql_coori,1)})" if top_coori != "-" else "-",
        "Início": ano_antigo,
        "Pico Modal": ano_modal,
        "Recente": ano_recente
    })
    
df_temas = pd.DataFrame(linhas_tabela).sort_values(by="Docs", ascending=False)
st.dataframe(df_temas, use_container_width=True, hide_index=True)

st.markdown("---")
# A partir daqui, o código de "# --- MOTOR DE BUSCA (EGO-GRAPH) ---" continua exatamente igual.

st.header("🗄️ Base de Dados Completa com Métricas SNA")

base_expandida = []
for d in dados_completos:
    row = d.copy()
    titulo = row.get('titulo')
    
    if isinstance(row.get('autores'), list): row['autores'] = ", ".join(row['autores'])
    if isinstance(row.get('co_orientadores'), list): row['co_orientadores'] = ", ".join(row['co_orientadores'])
    if isinstance(row.get('palavras_chave'), list): row['palavras_chave'] = ", ".join(row['palavras_chave'])
    
    metricas_doc = sna_global.get(titulo, {})
    row['Grau (SNA)'] = metricas_doc.get('Grau Absoluto', 0)
    row['Betweenness (SNA)'] = round(metricas_doc.get('Betweenness', 0.0), 4)
    row['Closeness (SNA)'] = round(metricas_doc.get('Closeness', 0.0), 4)
    row['Comunidade (SNA)'] = metricas_doc.get('Comunidade', 'N/A')
    row['Ranking Global (SNA)'] = metricas_doc.get('Ranking Global', 'N/A')
    
    base_expandida.append(row)

df_base_completa = pd.DataFrame(base_expandida)
colunas_principais = ['titulo', 'ano', 'nivel_academico', 'autores', 'orientador', 'co_orientadores', 'palavras_chave', 'macrotema', 'Grau (SNA)', 'Betweenness (SNA)', 'Closeness (SNA)', 'Comunidade (SNA)', 'Ranking Global (SNA)', 'resumo', 'url']
colunas_finais = [c for c in colunas_principais if c in df_base_completa.columns] + [c for c in df_base_completa.columns if c not in colunas_principais]

max_g = int(df_base_completa['Grau (SNA)'].max()) if not df_base_completa.empty else 100
max_b = float(df_base_completa['Betweenness (SNA)'].max()) if not df_base_completa.empty else 1.0
max_c = float(df_base_completa['Closeness (SNA)'].max()) if not df_base_completa.empty else 1.0

df_base_completa['nivel_academico'] = df_base_completa['nivel_academico'].astype('category')

if 'macrotema' in df_base_completa.columns:
    df_base_completa['macrotema'] = df_base_completa['macrotema'].astype('category')
if 'Comunidade (SNA)' in df_base_completa.columns:
    df_base_completa['Comunidade (SNA)'] = df_base_completa['Comunidade (SNA)'].astype('category')

st.dataframe(
    df_base_completa[colunas_finais],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Grau (SNA)": st.column_config.ProgressColumn("Grau (SNA)", min_value=0, max_value=max_g, format="%d"),
        "Betweenness (SNA)": st.column_config.ProgressColumn("Betweenness (SNA)", min_value=0, max_value=max_b, format="%.4f"),
        "Closeness (SNA)": st.column_config.ProgressColumn("Closeness (SNA)", min_value=0, max_value=max_c, format="%.4f")
    }
)
