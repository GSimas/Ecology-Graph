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
from sickle.oaiexceptions import NoRecordsMatch

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

@st.cache_data
def carregar_catalogo_programas():
    """Lê o catálogo de programas previamente validado e limpo."""
    try:
        with open('programas_ufsc.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("⚠️ Ficheiro 'programas_ufsc.json' não encontrado. Por favor, adicione-o à raiz do projeto.")
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

def realizar_extracao(set_spec, status_placeholder, nome_prog=""):
    # Aumentamos o timeout para 120s para dar tempo à UFSC de processar as "páginas" de resultados
    sickle = Sickle('https://repositorio.ufsc.br/oai/request', timeout=120)
    
    try: 
        records = sickle.ListRecords(metadataPrefix='oai_dc', set=set_spec)
    except NoRecordsMatch:
        status_placeholder.error(f"⚠️ O programa {nome_prog} está vazio ou fora do padrão.")
        return []
    except Exception as e:
        status_placeholder.error(f"⚠️ Erro de comunicação inicial: {e}")
        return []

    dados_extraidos = []
    titulos_vistos = set()
    iterator = iter(records)
    i = 0
    
    while True:
        # --- NÍVEL 1 DE PROTEÇÃO: REDE E PAGINAÇÃO ---
        try:
            record = next(iterator)
        except StopIteration:
            break # Fim natural: todos os documentos foram baixados
        except Exception as e:
            st.warning(f"⚠️ O servidor da UFSC interrompeu a ligação após {i} documentos do {nome_prog}. A guardar os dados já extraídos...")
            break # Se a rede cair, salva o que já tem e para a extração
            
        # --- NÍVEL 2 DE PROTEÇÃO: DOCUMENTO ISOLADO ---
        try:
            i += 1
            if i % 50 == 0:
                status_placeholder.info(f"⏳ [{nome_prog}] A extrair documentos... Já processados: **{i}**")

            # Ignora documentos deletados ou sem metadados válidos
            if record.header.deleted or not hasattr(record, 'metadata') or not record.metadata: 
                continue
                
            meta = record.metadata
            titulo = meta.get('title', [''])[0].strip()
            
            # Filtro de duplicados
            if not titulo or titulo in titulos_vistos: 
                continue
            titulos_vistos.add(titulo)
            
            autores = [normalizar_nome(a) for a in meta.get('creator', []) if a.strip()]
            ano_real = extrair_melhor_ano(meta.get('date', []))
            nivel = identificar_nivel(meta.get('type', []), titulo)
            
            contrib = [normalizar_nome(c) for c in meta.get('contributor', []) if "universidade" not in c.lower() and "ufsc" not in c.lower()]
            orientador = contrib[0] if len(contrib) > 0 else None
            co_orientadores = contrib[1:] if len(contrib) > 1 else []
            pks = list(set([normalizar_palavra_chave(pk) for pk in meta.get('subject', []) if pk]))
            
            descricoes = meta.get('description', [])
            resumo = max(descricoes, key=len) if descricoes else ""
            
            dados_extraidos.append({
                'titulo': titulo, 'nivel_academico': nivel, 'autores': autores,
                'orientador': orientador, 'co_orientadores': co_orientadores,
                'possui_coorientador': len(co_orientadores) > 0, 'palavras_chave': pks, 
                'ano': ano_real, 'resumo': resumo, 'programa_origem': nome_prog
            })
            
        except Exception:
            # SE UM DOCUMENTO ESPECÍFICO DER ERRO (ex: XML corrompido), 
            # IGNORA-O E PASSA AO PRÓXIMO EM VEZ DE PARAR TUDO!
            continue

    return dados_extraidos

@st.cache_data
def calcular_sna_global(dados):
    """Calcula as métricas SNA de toda a base de uma só vez para o Motor de Busca."""
    G = nx.Graph()
    for d in dados:
        doc = d.get('titulo')
        if not doc: continue
        G.add_node(doc, tipo='Documento')
        for a in d.get('autores', []):
            G.add_node(a, tipo='Autor')
            G.add_edge(doc, a)
        ori = d.get('orientador')
        if ori:
            G.add_node(ori, tipo='Orientador')
            G.add_edge(doc, ori)
        for co in d.get('co_orientadores', []):
            G.add_node(co, tipo='Co-orientador')
            G.add_edge(doc, co)
        for pk in d.get('palavras_chave', []):
            G.add_node(pk, tipo='Palavra-chave')
            G.add_edge(doc, pk)

    deg_cent = nx.degree_centrality(G)
    bet_cent = nx.betweenness_centrality(G)
    grau_abs = dict(G.degree())
    
    # Detecção de Comunidades (Louvain)
    try:
        comunidades = nx_comm.louvain_communities(G)
        mapa_comunidades = {node: i+1 for i, comm in enumerate(comunidades) for node in comm}
    except:
        mapa_comunidades = {}

    # Ranking Geral de Betweenness (Poder de Ponte)
    nodes_sorted_bet = sorted(bet_cent.items(), key=lambda x: x[1], reverse=True)
    rank_bet = {node: rank+1 for rank, (node, _) in enumerate(nodes_sorted_bet)}

    sna_dict = {}
    for node in G.nodes():
        sna_dict[node] = {
            'Grau Absoluto': grau_abs.get(node, 0),
            'Degree Centrality': deg_cent.get(node, 0),
            'Betweenness': bet_cent.get(node, 0),
            'Comunidade': mapa_comunidades.get(node, 'N/A'),
            'Ranking Global': rank_bet.get(node, 'N/A')
        }
    return sna_dict

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
    if fonte_nuvem == "Conceitos (Palavras-chave)":
        lista_c = []
        for lst in df_hist['palavras_chave']: lista_c.extend(lst)
        return dict(Counter(lista_c).most_common(100))
    else:
        # Define se vai ler Títulos ou Resumos
        if fonte_nuvem == "Resumos (Abstracts)":
            textos = df_hist.get('resumo', pd.Series()).dropna().astype(str).tolist()
        else:
            textos = df_hist['titulo'].dropna().astype(str).tolist()
            
        texto_completo = " ".join(textos).lower()
        texto_completo = re.sub(r'[^\w\s]', '', texto_completo)
        palavras = texto_completo.split()
        
        # Stopwords expandidas para limpar a linguagem processual de abstracts
        stopwords_pt = set(['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'uma', 'para', 'com', 'não', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'das', 'à', 'seu', 'sua', 'ou', 'nos', 'já', 'eu', 'também', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'sem', 'mesmo', 'aos', 'nas', 'me', 'esse', 'essa', 'num', 'nem', 'numa', 'pelos', 'pelas', 'este', 'esta', 'sobre', 'estudo', 'análise', 'proposta', 'uso', 'aplicação', 'desenvolvimento', 'modelo', 'sistema', 'avaliação', 'gestão', 'conhecimento', 'engenharia', 'objetivo', 'pesquisa', 'trabalho', 'resultados', 'método', 'foi', 'foram', 'são', 'ser', 'através', 'forma', 'apresenta'])
        
        palavras_limpas = [p for p in palavras if p not in stopwords_pt and len(p) > 2]
        return dict(Counter(palavras_limpas).most_common(100))

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
    
    # AGORA CARREGA INSTANTANEAMENTE DO JSON LOCAL
    colecoes_disponiveis = carregar_catalogo_programas()
    
    if colecoes_disponiveis:
        st.markdown("### Selecione os Programas para Analisar")
        
        programas_selecionados = st.multiselect("Pode selecionar múltiplos Programas de Pós-Graduação (PPGs):", list(colecoes_disponiveis.keys()))
        
        if programas_selecionados:
            st.markdown("#### ℹ️ Informações das Coleções Selecionadas")
            for prog in programas_selecionados:
                set_spec = colecoes_disponiveis[prog]
                
                # Monta o link Handle dinamicamente
                partes_id = set_spec.replace('col_', '').split('_')
                if len(partes_id) >= 2:
                    url_repositorio = f"https://repositorio.ufsc.br/handle/{partes_id[0]}/{partes_id[1]}"
                else:
                    url_repositorio = "https://repositorio.ufsc.br/"
                
                st.info(f"**{prog}**\n\n🔗 [Aceder à página original na UFSC]({url_repositorio}) | 🪪 ID Interno OAI: `{set_spec}`")
        
        with st.form("form_extracao"):
            st.warning("⚠️ A extração ao vivo agrupa todos os programas. Selecionar muitos programas exigirá mais tempo de download.")
            btn_extrair = st.form_submit_button("Iniciar Extração ao Vivo", type="primary")
            
        status_box = st.empty()
        
        if btn_extrair:
            if not programas_selecionados:
                status_box.error("Selecione pelo menos um programa para continuar.")
            else:
                dados_agregados = []
                for prog in programas_selecionados:
                    status_box.info(f"🚀 Iniciando conexão OAI-PMH com: {prog}...")
                    set_spec_alvo = colecoes_disponiveis[prog]
                    # Passamos o set_spec real encontrado no JSON
                    dados = realizar_extracao(set_spec_alvo, status_box, nome_prog=prog)
                    dados_agregados.extend(dados)
                
                if dados_agregados:
                    status_box.success(f"✅ Extração Global concluída! {len(dados_agregados)} documentos combinados.")
                    st.session_state['dados_completos'] = dados_agregados
                    
                    if len(programas_selecionados) == 1:
                        st.session_state['nome_programa'] = programas_selecionados[0]
                    else:
                        st.session_state['nome_programa'] = f"Análise Multidisciplinar ({len(programas_selecionados)} Programas)"
                        
                    st.rerun() 
    else:
        st.stop()


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
st.markdown("Plataforma de inteligência bibliométrica para mapeamento de redes acadêmicas, evolução histórica e análise topológica estrutural do conhecimento.")

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


# === MOTOR DE BUSCA E NAVEGAÇÃO EM GRAFO (SEARCH ENGINE) ===
st.header("🔍 Motor de Busca e Dossiê (Search Engine)")
st.markdown("Navegue pela rede! **Clique em qualquer autor, orientador ou documento nos resultados** para viajar através do conhecimento.")

# 1. Configuração da Memória de Navegação (Session State)
if 'busca_tipo' not in st.session_state:
    st.session_state['busca_tipo'] = "Documento"
if 'busca_termo' not in st.session_state:
    st.session_state['busca_termo'] = None

# Callback que muda o foco da pesquisa quando um botão é clicado
def navegar_para(novo_tipo, novo_termo):
    st.session_state['busca_tipo'] = novo_tipo
    st.session_state['busca_termo'] = novo_termo

# Calcula a inteligência da rede em background
sna_global = calcular_sna_global(dados_completos)

# 2. Interface Principal de Busca
tipo_busca = st.radio("O que deseja procurar?", ["Documento", "Autor", "Orientador", "Co-orientador", "Palavra-chave"], horizontal=True, key="busca_tipo")

opcoes_busca = []
if tipo_busca == "Documento": opcoes_busca = [d['titulo'] for d in dados_completos]
elif tipo_busca == "Autor": opcoes_busca = list(autores_set)
elif tipo_busca == "Orientador": opcoes_busca = list(orientadores_set)
elif tipo_busca == "Co-orientador": opcoes_busca = list(coorientadores_set)
elif tipo_busca == "Palavra-chave": opcoes_busca = list(keywords_set)

# Se mudarmos o tipo de busca (ex: de Autor para Documento), reseta o termo atual
if st.session_state['busca_termo'] not in opcoes_busca:
    st.session_state['busca_termo'] = None

# Encontra o índice do termo atual para a caixa de pesquisa manter o valor correto
idx_termo = sorted(opcoes_busca).index(st.session_state['busca_termo']) if st.session_state['busca_termo'] in opcoes_busca else None

termo_selecionado = st.selectbox(
    f"Selecione ou digite o nome do(a) {tipo_busca.lower()}:", 
    sorted(opcoes_busca), 
    index=idx_termo,
    placeholder=f"Escreva aqui para pesquisar..."
)

# Se o utilizador pesquisar manualmente na caixa, atualizamos o sistema
if termo_selecionado != st.session_state['busca_termo']:
    st.session_state['busca_termo'] = termo_selecionado
    st.rerun()

termo_ativo = st.session_state['busca_termo']

# 3. Exibição dos Detalhes e Hiperligações
if termo_ativo:
    st.markdown("### 📑 Resultado da Análise")
    col_info, col_sna = st.columns([2, 1])
    
    with col_info:
        st.markdown(f"#### Perfil: {tipo_busca}")
        st.info(f"**{termo_ativo}**")
        st.markdown("---")
        
        # --- LÓGICA DE DETALHES CLICÁVEIS ---
        if tipo_busca == "Documento":
            doc = next((d for d in dados_completos if d['titulo'] == termo_ativo), None)
            if doc:
                st.write(f"**Ano:** {doc.get('ano', 'N/A')} | **Nível:** {doc.get('nivel_academico', 'N/A')} | **Programa:** {doc.get('programa_origem', 'N/A')}")
                
                st.write("**Autor(es):**")
                for i, a in enumerate(doc.get('autores', [])): 
                    st.button(f"👤 {a}", key=f"doc_aut_{i}_{a}", on_click=navegar_para, args=("Autor", a))
                    
                if doc.get('orientador'):
                    st.write("**Orientador:**")
                    st.button(f"🏫 {doc['orientador']}", key=f"doc_ori_{doc['orientador']}", on_click=navegar_para, args=("Orientador", doc['orientador']))
                    
                if doc.get('co_orientadores'):
                    st.write("**Co-orientador(es):**")
                    for i, co in enumerate(doc.get('co_orientadores', [])): 
                        st.button(f"🤝 {co}", key=f"doc_co_{i}_{co}", on_click=navegar_para, args=("Co-orientador", co))
                        
                st.write("**Palavras-chave:**")
                for i, pk in enumerate(doc.get('palavras_chave', [])): 
                    st.button(f"💡 {pk}", key=f"doc_pk_{i}_{pk}", on_click=navegar_para, args=("Palavra-chave", pk))
                    
                with st.expander("Ler Resumo (Abstract)"):
                    st.write(doc.get('resumo', 'Resumo não disponível.'))
                    
        elif tipo_busca == "Autor":
            docs = [d for d in dados_completos if termo_ativo in d.get('autores', [])]
            oris = set([d.get('orientador') for d in docs if d.get('orientador')])
            co_oris = set([co for d in docs for co in d.get('co_orientadores', [])])
            
            if oris:
                st.write("**Orientadores que teve:**")
                for i, o in enumerate(oris): st.button(f"🏫 {o}", key=f"aut_ori_{i}_{o}", on_click=navegar_para, args=("Orientador", o))
            if co_oris:
                st.write("**Co-orientadores:**")
                for i, co in enumerate(co_oris): st.button(f"🤝 {co}", key=f"aut_co_{i}_{co}", on_click=navegar_para, args=("Co-orientador", co))
            
            st.write(f"**Documentos de Autoria ({len(docs)}):**")
            for i, d in enumerate(docs): 
                st.button(f"📄 {d['titulo']} ({d.get('ano')})", key=f"aut_doc_{i}_{d['titulo']}", on_click=navegar_para, args=("Documento", d['titulo']))
            
        elif tipo_busca == "Orientador":
            docs = [d for d in dados_completos if d.get('orientador') == termo_ativo]
            pupilos = set([a for d in docs for a in d.get('autores', [])])
            
            st.write(f"**Pupilos (Autores orientados - {len(pupilos)}):**")
            for i, p in enumerate(pupilos): 
                st.button(f"👤 {p}", key=f"ori_pup_{i}_{p}", on_click=navegar_para, args=("Autor", p))
            
            st.write(f"**Trabalhos Orientados ({len(docs)}):**")
            for i, d in enumerate(docs): 
                st.button(f"📄 {d['titulo']} ({d.get('ano')})", key=f"ori_doc_{i}_{d['titulo']}", on_click=navegar_para, args=("Documento", d['titulo']))
                
        elif tipo_busca == "Co-orientador":
            docs = [d for d in dados_completos if termo_ativo in d.get('co_orientadores', [])]
            st.write(f"**Trabalhos Co-orientados ({len(docs)}):**")
            for i, d in enumerate(docs): 
                st.button(f"📄 {d['titulo']} ({d.get('ano')})", key=f"coori_doc_{i}_{d['titulo']}", on_click=navegar_para, args=("Documento", d['titulo']))
                
        elif tipo_busca == "Palavra-chave":
            docs = [d for d in dados_completos if termo_ativo in d.get('palavras_chave', [])]
            st.write(f"**Aparece em {len(docs)} documentos:**")
            for i, d in enumerate(docs): 
                st.button(f"📄 {d['titulo']} ({d.get('ano')})", key=f"pk_doc_{i}_{d['titulo']}", on_click=navegar_para, args=("Documento", d['titulo']))

    with col_sna:
        st.markdown("#### 🕸️ Métricas de Rede (SNA)")
        metricas = sna_global.get(termo_ativo)
        
        if metricas:
            st.info(f"**Ranking de Influência:** #{metricas['Ranking Global']}")
            st.success(f"**Comunidade (Cluster):** {metricas['Comunidade']}")
            
            st.metric("Grau Absoluto (Conexões)", metricas['Grau Absoluto'])
            st.metric("Betweenness (Ponto de Ponte)", f"{metricas['Betweenness']:.4f}")
            st.metric("Degree Centrality (Volume)", f"{metricas['Degree Centrality']:.4f}")
            
            st.caption("A Comunidade indica o agrupamento matemático/temático a que este nó pertence na rede.")
        else:
            st.warning("Métricas SNA não disponíveis isoladamente.")

st.markdown("---")



# === SEÇÃO 1: GRAFO INTERATIVO GERAL ===
st.header("🕸️ 1. Topologia e Grafo Interativo")
niveis_sel_grafo = st.multiselect("Nível Acadêmico (Grafo):", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_grafo")
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
        niveis_sel_tabela = st.multiselect("Nível Acadêmico:", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_tabela")
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
        niveis_sel_hist = st.multiselect("Nível Acadêmico:", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_hist")
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
        niveis_sel_nuvem = st.multiselect("Nível Acadêmico:", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_nuvem")
    with col_n_filt2:
        orientador_sel_nuvem = st.multiselect("Orientador(es):", options=orientadores_disponiveis, default=[], help="Deixe em branco para considerar todos.")
    with col_n_filt3:
        anos_sel_nuvem = st.slider("Intervalo de Anos:", min_ano_global, max_ano_global, (min_ano_global, max_ano_global), 1, key="ano_nuvem")

    fonte_nuvem = st.radio("Base de texto:", ["Conceitos (Palavras-chave)", "Títulos dos Documentos", "Resumos (Abstracts)"], horizontal=True)
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
        niveis_sel_co = st.multiselect("Nível Acadêmico:", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_co")
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
