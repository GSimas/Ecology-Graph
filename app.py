import streamlit as st
import streamlit.components.v1 as components
from streamlit_agraph import agraph, Node, Edge, Config
import networkx as nx
import networkx.algorithms.community as nx_comm
import pandas as pd
import json
import re
import unicodedata
from collections import Counter
from sickle import Sickle
from sickle.oaiexceptions import NoRecordsMatch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import google.generativeai as genai

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

def aplicar_macrotemas(dados, api_key, num_topicos=12):
    # 1. Configuração da API
    genai.configure(api_key=api_key)
    
    # Apontando para o modelo exato detectado no seu log de erro
    model = genai.GenerativeModel('gemini-2.5-flash') 
    
    # 2. SUPER-LIMPEZA (Mantemos a mesma peneira rigorosa que construímos)
    sujeira_academica = [
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com", "não", "uma", "os", "no", "se", "na", 
        "por", "mais", "as", "dos", "como", "mas", "foi", "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser",
        "neste", "esta", "está", "este", "pelo", "pela", "seus", "suas", "nas", "aos", "meu", "sua", "através",
        "the", "of", "and", "in", "to", "for", "with", "on", "at", "by", "from", "an", "is", "it", "this", "that",
        "study", "analysis", "based", "using", "results", "work", "research", "paper", "thesis", "dissertation",
        "analise", "estudo", "desenvolvimento", "proposta", "metodo", "processo", "sistema", "modelo", "projeto",
        "utilização", "uso", "efeito", "avaliação", "verificação", "experimental", "numérica", "aplicação",
        "sobre", "entre", "quando", "onde", "qual", "quais", "abstract", "resumo", "palavras", "chave"
    ]

    textos = []
    for doc in dados:
        # Peso Triplo para o Título + Keywords + Resumo
        bruto = f"{(doc.get('titulo', '') + ' ') * 3} {' '.join(doc.get('palavras_chave', []))} {doc.get('resumo', '')}"
        limpo = re.sub(r'[^a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ\s]', ' ', bruto).lower()
        textos.append(limpo)

    # 3. VETORIZAÇÃO E NMF (Matemática Pura)
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words=sujeira_academica, max_features=800)
    
    try:
        tfidf_matrix = vectorizer.fit_transform(textos)
        nmf_model = NMF(n_components=num_topicos, random_state=42, init='nndsvd')
        nmf_matrix = nmf_model.fit_transform(tfidf_matrix)
        feature_names = vectorizer.get_feature_names_out()
    except Exception as e:
        st.error(f"Erro na vetorização (amostra pode ser muito pequena ou similar): {e}")
        return dados

    # Montar contexto para o Gemini
    clusters = []
    for idx, topic in enumerate(nmf_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-8:-1]]
        clusters.append(f"Grupo {idx+1}: {', '.join(top_words)}")
    
    contexto = "\n".join(clusters)

    # 4. Prompt para o Gemini (Focado em rigor acadêmico e síntese transdisciplinar)
    prompt_humanizado = f"""Você é um especialista em epistemologia e taxonomia acadêmica.
Abaixo estão {num_topicos} grupos de palavras-chave extraídas de agrupamentos matemáticos (NMF) de teses e dissertações.
Sua missão é batizar cada grupo com um nome definitivo e que represente com precisão essa subárea do conhecimento.

Diretrizes rigorosas:
- Crie títulos com no máximo 4 palavras.
- NÃO use palavras genéricas como 'Estudo', 'Análise', 'Área de', 'Aplicações', 'Correlatas'.
- Vá direto ao ponto. Exemplo: Se ler 'soldagem, laser, liga, tensão', responda 'Tecnologias de Soldagem' ou 'Metalurgia e Soldagem a Laser'.
- Retorne APENAS uma lista estritamente numerada de 1 a {num_topicos}, sem introduções ou conclusões.

GRUPOS DE PALAVRAS:
{contexto}"""

    # 5. Geração e Tratamento
    try:
        response = model.generate_content(
            prompt_humanizado,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                temperature=0.4,
            )
        )
        texto_resposta = response.text.strip()
        
        # Limpeza da lista retornada
        respostas = texto_resposta.split('\n')
        nomes_finais = [re.sub(r'^\d+[\.\s\-]+', '', r).strip().replace('*', '') for r in respostas if len(r) > 3]
        
        # Validação de segurança
        if len(nomes_finais) < num_topicos:
            raise ValueError(f"Gemini retornou apenas {len(nomes_finais)} nomes. Esperados: {num_topicos}")

    except Exception as e:
        st.error(f"Erro na API Gemini: {e}")
        # Plano B (Extração direta)
        nomes_finais = []
        for c in clusters:
            palavras = c.split(': ')[1].split(', ')
            nomes_finais.append(f"{palavras[0].title()} e {palavras[1].title()}")

    # 6. ATRIBUIÇÃO DOS RÓTULOS AOS DOCUMENTOS
    for i, doc in enumerate(dados):
        top_idx = nmf_matrix[i].argmax()
        doc['macrotema'] = nomes_finais[top_idx] if top_idx < len(nomes_finais) else "Interseções Multidisciplinares"

    return dados

# --- FUNÇÕES DE BACKEND (EXTRAÇÃO E BUSCA) ---
@st.cache_data
def carregar_catalogo_programas():
    try:
        with open('programas_ufsc.json', 'r', encoding='utf-8') as f: return json.load(f)
    except FileNotFoundError:
        st.error("⚠️ Ficheiro 'programas_ufsc.json' não encontrado na raiz.")
        return {}

def extrair_melhor_ano(lista_datas):
    if not lista_datas: return None
    anos = [int(m) for d in lista_datas for m in re.findall(r'\b(19\d{2}|20\d{2})\b', str(d))]
    return str(min(anos)) if anos else None

def normalizar_nome(nome):
    return ''.join(c for c in unicodedata.normalize('NFD', nome) if unicodedata.category(c) != 'Mn').strip().title() if nome else ""

def normalizar_palavra_chave(pk):
    return ''.join(c for c in unicodedata.normalize('NFD', pk.lower().strip()) if unicodedata.category(c) != 'Mn') if pk else ""

def identificar_nivel(tipos, titulo=""):
    tipos_str = " ".join(tipos).lower()
    if 'doctoral' in tipos_str or 'tese' in tipos_str or 'tese' in titulo.lower(): return 'Tese (Doutorado)'
    if 'master' in tipos_str or 'disserta' in tipos_str or 'disserta' in titulo.lower(): return 'Dissertação (Mestrado)'
    return 'Outros'

def realizar_extracao(set_spec, status_placeholder, nome_prog=""):
    sickle = Sickle('https://repositorio.ufsc.br/oai/request', timeout=120)
    try: records = sickle.ListRecords(metadataPrefix='oai_dc', set=set_spec)
    except: return []
    dados_extraidos, titulos_vistos, iterator, i = [], set(), iter(records), 0
    
    while True:
        try: record = next(iterator)
        except StopIteration: break
        except Exception: break
            
        try:
            i += 1
            if i % 50 == 0: status_placeholder.info(f"⏳ [{nome_prog}] A extrair documentos... Já processados: **{i}**")
            if record.header.deleted or not hasattr(record, 'metadata') or not record.metadata: continue
            meta = record.metadata
            titulo = meta.get('title', [''])[0].strip()
            if not titulo or titulo in titulos_vistos: continue
            titulos_vistos.add(titulo)
            
            autores = [normalizar_nome(a) for a in meta.get('creator', []) if a.strip()]
            ano_real = extrair_melhor_ano(meta.get('date', []))
            nivel = identificar_nivel(meta.get('type', []), titulo)
            contrib = [normalizar_nome(c) for c in meta.get('contributor', []) if "ufsc" not in c.lower() and "universidade" not in c.lower()]
            orientador = contrib[0] if len(contrib) > 0 else None
            co_orientadores = contrib[1:] if len(contrib) > 1 else []
            pks = list(set([normalizar_palavra_chave(pk) for pk in meta.get('subject', []) if pk]))
            descricoes = meta.get('description', [])
            resumo = max(descricoes, key=len) if descricoes else ""
            
            identificadores = meta.get('identifier', [])
            url_doc = next((link for link in identificadores if str(link).startswith('http')), "")
            
            dados_extraidos.append({
                'titulo': titulo, 
                'nivel_academico': nivel, 
                'autores': autores, 
                'orientador': orientador, 
                'co_orientadores': co_orientadores, 
                'possui_coorientador': len(co_orientadores) > 0, 
                'palavras_chave': pks, 
                'ano': ano_real, 
                'resumo': resumo, 
                'programa_origem': nome_prog,
                'url': url_doc 
            })
        except Exception: continue
    return dados_extraidos

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
        
        # Injeção do Macrotema no Grafo
        mt = d.get('macrotema')
        if mt: G.add_node(mt, tipo='Macrotema'); G.add_edge(doc, mt)

    deg_cent, bet_cent, grau_abs = nx.degree_centrality(G), nx.betweenness_centrality(G), dict(G.degree())
    try: mapa_comunidades = {node: i+1 for i, comm in enumerate(nx_comm.louvain_communities(G)) for node in comm}
    except: mapa_comunidades = {}
    rank_bet = {node: rank+1 for rank, (node, _) in enumerate(sorted(bet_cent.items(), key=lambda x: x[1], reverse=True))}

    return {node: {'Grau Absoluto': grau_abs.get(node, 0), 'Degree Centrality': deg_cent.get(node, 0), 'Betweenness': bet_cent.get(node, 0), 'Comunidade': mapa_comunidades.get(node, 'N/A'), 'Ranking Global': rank_bet.get(node, 'N/A')} for node in G.nodes()}

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
                
        # Estilização estendida para incluir Macrotema
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

# --- INTERFACE DE EXTRAÇÃO ---
if 'dados_completos' not in st.session_state:
    st.title("🔌 Conexão Direta: Repositório Institucional UFSC")
    colecoes_disponiveis = carregar_catalogo_programas()
    
    if colecoes_disponiveis:
        st.markdown("### Selecione os Programas para Analisar")
        programas_selecionados = st.multiselect("Pode selecionar múltiplos Programas (PPGs):", list(colecoes_disponiveis.keys()))
        
        if programas_selecionados:
            for prog in programas_selecionados:
                set_spec = colecoes_disponiveis[prog]
                url = f"https://repositorio.ufsc.br/handle/{set_spec.split('_')[-1]}" if '_' in set_spec else "https://repositorio.ufsc.br/"
                st.info(f"**{prog}**\n\n🔗 [Aceder à página original na UFSC]({url}) | 🪪 ID Interno OAI: `{set_spec}`")
        
        with st.form("form_extracao"):
            btn_extrair = st.form_submit_button("Iniciar Extração ao Vivo", type="primary")
            
        status_box = st.empty()
        if btn_extrair and programas_selecionados:
            dados_agregados = []
            for prog in programas_selecionados:
                dados_agregados.extend(realizar_extracao(colecoes_disponiveis[prog], status_box, nome_prog=prog))
            
            if dados_agregados:
                status_box.success("✅ Extração Concluída!")
                st.session_state.update({
                    'dados_completos': dados_agregados, 
                    'nome_programa': prog if len(programas_selecionados)==1 else f"Análise Multidisciplinar ({len(programas_selecionados)} Programas)",
                    'macrotemas_computados': False
                })
                st.rerun() 
    st.stop()

# --- DASHBOARD PRINCIPAL ---
dados_completos = st.session_state['dados_completos']
st.title("🌌 Ecologia do Conhecimento")
st.subheader(f"Base: {st.session_state['nome_programa']}")

if st.sidebar.button("🔄 Nova Extração", type="primary"):
    del st.session_state['dados_completos']
    st.session_state['macrotemas_computados'] = False
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

st.markdown("---")

# --- MÓDULO DE MACROTEMAS ---
st.header("🧠 Análise Temática Estrutural")

if not st.session_state['macrotemas_computados']:
    st.info("Os documentos extraídos ainda não possuem categorização temática.")
    
    if st.button("Computar Macrotemas Agora", type="primary"):
        try:
            # Puxa a chave do Gemini em vez do Groq
            minha_chave_gemini = st.secrets["GEMINI_API_KEY"]
            
            with st.spinner("Analisando ecossistema de dados com NMF e IA do Google Gemini..."):
                st.session_state['dados_completos'] = aplicar_macrotemas(
                    dados_completos, 
                    api_key=minha_chave_gemini
                )
                st.session_state['macrotemas_computados'] = True
                
                calcular_sna_global.clear()
                gerar_nodos_agraph.clear()
                st.rerun()
                
        except KeyError:
            st.error("❌ Erro: 'GEMINI_API_KEY' não encontrada nos Secrets do Streamlit.")
            st.info("💡 Certifique-se de que adicionou a chave no painel do Streamlit Cloud em 'Settings' -> 'Secrets'.")
            
else:
    # Exibir Tabela de Macrotemas
    todos_temas = [d.get('macrotema', 'Multidisciplinar / Transversal') for d in dados_completos]
    contagem_temas = Counter(todos_temas)
    df_temas = pd.DataFrame(contagem_temas.items(), columns=["Macrotema", "Quantidade de Documentos"]).sort_values(by="Quantidade de Documentos", ascending=False)
    
    col_tabela, col_vazia = st.columns([2, 1])
    with col_tabela:
        st.dataframe(df_temas, use_container_width=True, hide_index=True)

st.markdown("---")

# --- MOTOR DE BUSCA (EGO-GRAPH) ---
st.header("🔍 Motor de Busca e Dossiê")

sna_global = calcular_sna_global(dados_completos)

# Define as opções de busca baseadas no estado da aplicação
opcoes_busca = ["Documento", "Autor", "Orientador", "Co-orientador", "Palavra-chave"]
if st.session_state['macrotemas_computados']:
    opcoes_busca.append("Macrotema")

tipo_busca = st.radio("Procurar por Entidade:", opcoes_busca, horizontal=True, key="busca_tipo")

# Prepara a lista de opções para o Selectbox
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
                
            # TAG do Macrotema clicável
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
            for d in docs: st.button(f"📄 {d['titulo']}", on_click=navegar_para, args=("Documento", d['titulo']))
        elif tipo_busca == "Orientador":
            docs = [d for d in dados_completos if d.get('orientador') == termo_ativo]
            for d in docs: st.button(f"📄 {d['titulo']}", on_click=navegar_para, args=("Documento", d['titulo']))
        elif tipo_busca == "Palavra-chave":
            docs = [d for d in dados_completos if termo_ativo in d.get('palavras_chave', [])]
            for d in docs: st.button(f"📄 {d['titulo']}", on_click=navegar_para, args=("Documento", d['titulo']))
        elif tipo_busca == "Macrotema":
            docs = [d for d in dados_completos if d.get('macrotema') == termo_ativo]
            st.write(f"**Documentos encontrados na categoria ({len(docs)}):**")
            for d in docs: st.button(f"📄 {d['titulo']}", key=f"btn_mt_{d['titulo']}", on_click=navegar_para, args=("Documento", d['titulo']))

    with col_sna:
        metricas = sna_global.get(termo_ativo, {})
        if metricas:
            st.success(f"Cluster: {metricas.get('Comunidade')} | Rank: #{metricas.get('Ranking Global')}")
            st.metric("Grau (Conexões)", metricas.get('Grau Absoluto'))
            st.metric("Betweenness", f"{metricas.get('Betweenness', 0):.4f}")

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
