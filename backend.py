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
import requests
import urllib.parse
from neo4j import GraphDatabase
from pyvis.network import Network
from streamlit_agraph import Node, Edge
import plotly.express as px
import pandas as pd
import numpy as np
from scipy import stats
import io
import itertools
import math
import time
import scipy.stats as sps

from app_config import get_gemini_api_key, get_neo4j_credentials
from gemini_utils import DEFAULT_FAST_MODELS, DEFAULT_TEXT_MODELS, generate_content, response_text

def _estimar_gamma_lei_potencia(degrees: list) -> float:
    """
    Estimador MLE (máxima verossimilhança) do expoente γ da lei de potência.
    Robusto a redes degeneradas (graus iguais, nós isolados, amostra pequena).
    Retorna 0.0 em vez de lançar exceção para qualquer caso edge.

    Fórmula: γ = 1 + n / Σ ln(xi / x_min)
    Referência: Clauset, Shalizi & Newman (2009), SIAM Review.
    """
    degrees_validos = [d for d in degrees if isinstance(d, (int, float)) and d > 0]

    if len(degrees_validos) < 10:
        return 0.0  # amostra insuficiente para estimativa confiável

    d_min = min(degrees_validos)
    if d_min < 1:
        return 0.0  # graus devem ser ≥ 1 para log ser definido

    try:
        log_sum = sum(math.log(d / d_min) for d in degrees_validos)
        if log_sum <= 0:
            return 0.0  # rede degenerada: todos os graus iguais → log_sum = 0
        return 1.0 + len(degrees_validos) / log_sum
    except (ValueError, ZeroDivisionError, OverflowError):
        return 0.0


# --- CONSTANTE GLOBAL: Stopwords para análise lexical ---
# Unificado para PT e EN. Inclui termos acadêmicos genéricos que poluem nuvens e redes.
STOPWORDS_ACADEMICAS: frozenset = frozenset({
    # Português — conectivos e artigos
    'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'uma', 'para', 'com',
    'não', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas',
    'ao', 'das', 'à', 'seu', 'sua', 'ou', 'nos', 'já', 'eu', 'também', 'pelo',
    'pela', 'até', 'isso', 'ela', 'entre', 'sem', 'mesmo', 'aos', 'nas', 'me',
    'esse', 'essa', 'num', 'nem', 'numa', 'pelos', 'pelas', 'este', 'esta',
    'sob', 'perspectiva', 'frente', 'partir', 'baseado',
    # Português — termos acadêmicos genéricos
    'sobre', 'estudo', 'análise', 'proposta', 'uso', 'aplicação', 'desenvolvimento',
    'modelo', 'sistema', 'avaliação', 'gestão', 'conhecimento', 'engenharia',
    'objetivo', 'pesquisa', 'trabalho', 'resultados', 'método', 'foi', 'foram',
    'são', 'ser', 'através', 'forma', 'apresenta',
    # Inglês — conectivos
    'the', 'of', 'and', 'in', 'to', 'a', 'is', 'for', 'by', 'on', 'with', 'an',
    'as', 'this', 'that', 'which', 'from', 'it', 'or', 'be', 'are', 'at', 'has',
    'have', 'was', 'were', 'not', 'but',
})


@st.cache_data(show_spinner=False)
def gerar_grafo_ecologia_memes_agraph(dados_lista, min_coocorrencia=1, fonte_memes="Artefatos Extraídos"):
    import networkx as nx
    import itertools
    from streamlit_agraph import Node, Edge
    import pandas as pd
    import numpy as np
    import scipy.stats as sps
    import re
    import unicodedata
    
    # Stopwords para limpar os títulos no método Tradicional
    stopwords = STOPWORDS_ACADEMICAS

    def remover_acentos(texto):
        if not isinstance(texto, str): return ""
        return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

    stopwords_norm = {remover_acentos(w) for w in stopwords}
    
    G_completo = nx.Graph()
    mapa_docs = {}
    
    # 1. Constrói a Rede de Co-ocorrência Integral
    for d in dados_lista:
        titulo = d.get('titulo', 'Sem Título')
        artefatos_doc = []
        
        # LÓGICA A: Inteligência Artificial (Artefatos)
        if fonte_memes == "Artefatos Extraídos":
            onto = d.get('ontologia_ia', {})
            if isinstance(onto, dict):
                teorias = onto.get('teorias_e_modelos', [])
                ferramentas = onto.get('ferramentas_e_artefatos', [])
                metodos = onto.get('metodos_e_tecnicas', [])
                
                def safe_list(l): return [str(x).strip().title() for x in l if str(x).strip()] if isinstance(l, list) else []
                artefatos_doc = list(set(safe_list(teorias) + safe_list(ferramentas) + safe_list(metodos)))
                
        # LÓGICA B: Tradicional (Palavras-chave e Títulos)
        else:
            memes = set()
            pks = d.get('palavras_chave', [])
            if not isinstance(pks, list): pks = [pks] if pd.notna(pks) else []
            memes.update([str(p).strip().title() for p in pks if str(p).strip()])
            
            titulo_norm = remover_acentos(str(titulo).lower())
            palavras_titulo = re.findall(r'\b[a-z]{3,}\b', titulo_norm)
            memes.update([p.title() for p in palavras_titulo if p not in stopwords_norm])
            artefatos_doc = list(memes)
            
        # Rastreia os documentos associados a cada termo
        for a in artefatos_doc:
            if a not in mapa_docs: mapa_docs[a] = set()
            mapa_docs[a].add(titulo)
            
        if len(artefatos_doc) > 1:
            for a1, a2 in itertools.combinations(artefatos_doc, 2):
                if G_completo.has_edge(a1, a2): G_completo[a1][a2]['weight'] += 1
                else: G_completo.add_edge(a1, a2, weight=1)
        elif len(artefatos_doc) == 1:
            G_completo.add_node(artefatos_doc[0])
            
    # Trava de segurança: Se não houver nada, retorna vazio sem quebrar
    if G_completo.number_of_nodes() == 0:
        return [], [], pd.DataFrame(), {}, {}

    # 2. MATEMÁTICA DA REDE COMPLETA (Estatísticas Globais)
    deg_cent_full = nx.degree_centrality(G_completo)
    bet_cent_full = nx.betweenness_centrality(G_completo, weight='weight')
    clo_cent_full = nx.closeness_centrality(G_completo)
    grau_abs_full = dict(G_completo.degree())
    
    label_coluna = "Artefato (IA)" if fonte_memes == "Artefatos Extraídos" else "Termo/Conceito (Tradicional)"
    
    df_nos = pd.DataFrame({
        label_coluna: list(G_completo.nodes()),
        'Grau Absoluto': [grau_abs_full.get(n, 0) for n in G_completo.nodes()],
        'Grau (Degree)': [deg_cent_full.get(n, 0) for n in G_completo.nodes()],
        'Betweenness': [bet_cent_full.get(n, 0) for n in G_completo.nodes()],
        'Closeness': [clo_cent_full.get(n, 0) for n in G_completo.nodes()],
        'Documentos Associados': [", ".join(list(mapa_docs.get(n, []))) for n in G_completo.nodes()]
    }).sort_values('Grau Absoluto', ascending=False)
    
    # 3. MÉTRICAS TOTAIS (Cards)
    degrees = list(grau_abs_full.values())
    try: pr = nx.pagerank(G_completo, weight='weight')
    except Exception: pr = {n: 0 for n in G_completo.nodes()}
    
    net_metrics = {
        'densidade': nx.density(G_completo),
        'eficiencia': nx.global_efficiency(G_completo),
        'clustering': nx.average_clustering(G_completo, weight='weight'),
        'links_mean': np.mean(degrees),
        'links_std': np.std(degrees),
        'links_min': np.min(degrees),
        'links_max': np.max(degrees),
        'pr_avg': np.mean(list(pr.values())),
        'entropia': -sum((pd.Series(degrees).value_counts(normalize=True)) * np.log2(pd.Series(degrees).value_counts(normalize=True))),
        'ev_avg': 0, 
        'constraint_avg': 0,
        'redundancia': 1 - nx.global_efficiency(G_completo)
    }
    
    # 4. FILTRO DE EXIBIÇÃO VISUAL (Slider)
    G_display = G_completo.copy()
    edges_to_remove = [(u, v) for u, v, data in G_display.edges(data=True) if data['weight'] < min_coocorrencia]
    G_display.remove_edges_from(edges_to_remove)
    G_display.remove_nodes_from(list(nx.isolates(G_display)))

    nodes, edges = [], []
    max_grau_abs = max(degrees) if degrees else 1
    
    # Estilização Inteligente: IA é roxo, Tradicional é Verde
    cor_base = "rgba(155, 89, 182, 0.4)" if fonte_memes == "Artefatos Extraídos" else "rgba(46, 204, 113, 0.4)"
    cor_hover = "#F39C12" if fonte_memes == "Artefatos Extraídos" else "#27AE60"
    
    cor_dinamica_no = {
        "background": cor_base, "border": cor_base,
        "hover": {"background": cor_hover, "border": cor_hover},
        "highlight": {"background": cor_hover, "border": cor_hover}
    }
    
    for node in G_display.nodes():
        tamanho = 15 + (grau_abs_full[node] / max_grau_abs) * 25
        hover_text = f"🧬 Termo: {node}\n🔗 Grau Absoluto: {grau_abs_full[node]}\n🌉 Betweenness Global: {bet_cent_full[node]:.4f}"
        
        nodes.append(Node(
            id=node, label=node, size=tamanho, title=hover_text, color=cor_dinamica_no, 
            font={'color': 'white', 'strokeWidth': 4, 'strokeColor': '#1E1E1E'}
        ))
        
    cor_dinamica_aresta = {"color": "rgba(127, 140, 141, 0.2)", "hover": cor_hover, "highlight": cor_hover}
        
    for u, v, data in G_display.edges(data=True):
        edges.append(Edge(source=u, target=v, value=data['weight'], color=cor_dinamica_aresta))
        
    gamma = _estimar_gamma_lei_potencia(degrees)
    spearman, _ = sps.spearmanr(list(deg_cent_full.values()), list(bet_cent_full.values()))
    
    net_maturity = {
        'gamma': gamma,
        'spearman': spearman if not np.isnan(spearman) else 0,
        'assortatividade': nx.degree_assortativity_coefficient(G_completo),
        'rich_club': 0 
    }

    return nodes, edges, df_nos, net_metrics, net_maturity
    

def configurar_gemini():
    """Confere se há chave disponível via env ou Streamlit secrets."""
    return bool(get_gemini_api_key())


def _chamar_gemini_com_retry(prompt: str, model_candidates, temperature: float,
                              response_mime_type: str,
                              max_tentativas: int = 3,
                              delay_base: float = 2.0):
    """
    Wrapper com backoff exponencial para chamadas à API do Gemini.
    Tentativa 1: imediata
    Tentativa 2: aguarda delay_base segundos (2s por padrão)
    Tentativa 3: aguarda delay_base * 2 segundos (4s por padrão)
    Lança a última exceção se todas as tentativas falharem.
    """
    ultima_excecao = None
    for tentativa in range(max_tentativas):
        try:
            return generate_content(
                prompt=prompt,
                model_candidates=model_candidates,
                temperature=temperature,
                response_mime_type=response_mime_type,
            )
        except Exception as e:
            ultima_excecao = e
            if tentativa < max_tentativas - 1:
                espera = delay_base * (2 ** tentativa)
                time.sleep(espera)
    raise ultima_excecao


def extrair_artefatos_llm(titulo, resumo):
    """Envia um prompt estruturado ao Gemini forçando um retorno em JSON."""
    if not titulo or not resumo:
        return None, "Documento sem resumo válido"
        
    prompt = f"""
    Extraia as entidades específicas que foram objeto central de desenvolvimento, análise ou uso metodológico no texto abaixo.
    Não extraia palavras genéricas. Busque ferramentas, métodos, algoritmos, frameworks ou teorias.
    
    REGRA CRÍTICA DE PADRONIZAÇÃO (NOME CANÔNICO):
    Padronize os nomes encontrados similares para a sua forma mais comum, curta ou sigla.
    Por exemplo: se o texto diz "Microscópio Eletrônico de Varredura" ou "Microscopia Eletrônica de Varredura", escolha apenas um nome e os agrupe.
    Agrupe variações do mesmo conceito sob um único nome guarda-chuva consolidado.
    
    O RETORNO DEVE SER EXATAMENTE UM JSON COM AS SEGUINTES CHAVES:
    {{
        "teorias_e_modelos": ["teoria 1", "modelo A"],
        "ferramentas_e_artefatos": ["ferramenta 1", "software B"],
        "metodos_e_tecnicas": ["método 1", "técnica C"]
    }}
    Se não houver itens para uma categoria, retorne uma lista vazia [].
    
    Título: {titulo}
    Resumo: {resumo}
    """
    
    try:
        response = _chamar_gemini_com_retry(
            prompt=prompt,
            model_candidates=DEFAULT_FAST_MODELS,
            temperature=0.2,
            response_mime_type="application/json",
        )

        dicionario_ontologia = json.loads(response_text(response))
        
        if isinstance(dicionario_ontologia, list) and len(dicionario_ontologia) > 0 and isinstance(dicionario_ontologia[0], dict):
            # Extraímos apenas o texto de dentro de cada 'entity' (ou qualquer chave que ela inventar)
            lista_achatada = []
            for item in dicionario_ontologia:
                # Pega o primeiro valor de qualquer chave que existir no dicionário interno
                valor = list(item.values())[0] if item.values() else None
                if valor: lista_achatada.append(str(valor))
            
            dicionario_ontologia = {
                'teorias_e_modelos': [],
                'ferramentas_e_artefatos': lista_achatada,
                'metodos_e_tecnicas': []
            }

        # CASO B: A IA mandou uma lista simples de strings ["A", "B"]
        elif isinstance(dicionario_ontologia, list):
            dicionario_ontologia = {
                'teorias_e_modelos': [],
                'ferramentas_e_artefatos': [str(x) for x in dicionario_ontologia],
                'metodos_e_tecnicas': []
            }
            
        # CASO C: Não veio um dicionário (prevenção total)
        elif not isinstance(dicionario_ontologia, dict):
            dicionario_ontologia = {}

        # --- FIM DA BLINDAGEM ---
            
        # Garantir que as chaves padrão existam para o restante do código
        for chave in ['teorias_e_modelos', 'ferramentas_e_artefatos', 'metodos_e_tecnicas']:
            if chave not in dicionario_ontologia or not isinstance(dicionario_ontologia[chave], list):
                dicionario_ontologia[chave] = []
            
        return dicionario_ontologia, None
        
    except Exception as e:
        # Pega a mensagem exata de erro da Google (ex: API Limit, Erro de Auth, etc)
        erro_msg = str(e)
        if 'response' in locals() and hasattr(response, 'text'):
            erro_msg += f" | Resposta bruta: {response.text[:100]}"
        return None, erro_msg

def processar_lote_ontologia(dados_completos, tamanho_lote=10, barra_progresso=None, status_texto=None):
    """Processa o lote com cálculo de tempo estimado de conclusão (ETC)."""
    if not configurar_gemini():
        if status_texto:
            status_texto.error("Erro: GEMINI_API_KEY não encontrada em variáveis de ambiente ou secrets.")
        return 0, sum(1 for d in dados_completos if 'ontologia_ia' not in d and d.get('resumo'))
    
    fila_processamento = [d for d in dados_completos if 'ontologia_ia' not in d and d.get('resumo') and str(d.get('resumo')).strip() != ""]
    lote_atual = fila_processamento[:tamanho_lote]
    
    if not lote_atual:
        return 0, len(fila_processamento)
        
    processados = 0
    erros = []
    tempo_inicio_lote = time.time() # Registro do início
    
    for i, doc in enumerate(lote_atual):
        # --- CÁLCULO DE TEMPO ESTIMADO ---
        if i > 0:
            tempo_decorrido = time.time() - tempo_inicio_lote
            media_por_doc = tempo_decorrido / i
            docs_restantes = len(lote_atual) - i
            eta_segundos = docs_restantes * media_por_doc
            
            minutos, segundos = divmod(int(eta_segundos), 60)
            eta_str = f" | ⏳ Restam aprox. **{minutos}m {segundos}s**"
        else:
            eta_str = " | ⏳ Calculando tempo..."

        if status_texto:
            status_texto.markdown(f"🤖 Lendo {i+1}/{len(lote_atual)}: *{doc.get('titulo', '')[:50]}...*{eta_str}")
        # ---------------------------------
            
        ontologia, erro = extrair_artefatos_llm(doc.get('titulo'), doc.get('resumo'))
        
        if ontologia:
            doc['ontologia_ia'] = ontologia
            processados += 1
        elif erro:
            erros.append(f"**{doc.get('titulo', '')[:30]}...**: {erro}")
            
        if barra_progresso:
            barra_progresso.progress((i + 1) / len(lote_atual))
            
        # DELAY DE SEGURANÇA: 4 segundos entre requisições
        if i < len(lote_atual) - 1:
            time.sleep(4) 
            
    if erros and status_texto:
        with st.expander(f"⚠️ {len(erros)} falhas na IA. Clique para ver."):
            for e in erros[:10]: st.error(e)
                
    return processados, len(fila_processamento) - processados

# --- FUNÇÕES DE BACK-END (AVANÇADO) ---
@st.cache_data
def preparar_dataframe(dados_lista):
    """Prepara o DataFrame para as análises avançadas (Burst, Memética, etc)."""
    df = pd.DataFrame(dados_lista)
    df['Ano'] = pd.to_numeric(df.get('ano'), errors='coerce')
    df = df.dropna(subset=['Ano'])
    df['Ano'] = df['Ano'].astype(int)
    df['nivel_academico'] = df.get('nivel_academico', 'Outros').fillna('Outros')
    return df

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

@st.cache_data
def calcular_metricas_memeticas(df_base, fonte_memes="Palavras-chave"):
    import re
    import unicodedata 
    
    if df_base.empty: 
        return pd.DataFrame(), pd.DataFrame(), 0, 0, pd.DataFrame(), pd.DataFrame()

    stopwords = STOPWORDS_ACADEMICAS

    def remover_acentos(texto):
        if not isinstance(texto, str): return ""
        return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

    stopwords_norm = {remover_acentos(w) for w in stopwords}

    def extrair_memes_completos(row):
        memes = set()
        
        # LÓGICA 1: Abordagem Tradicional
        if fonte_memes == "Palavras-chave":
            pks = row.get('palavras_chave', [])
            if not isinstance(pks, list):
                pks = [pks] if pd.notna(pks) else []
            memes.update([remover_acentos(str(p).lower().strip()) for p in pks if str(p).strip()])
            
            titulo_norm = remover_acentos(str(row.get('titulo', '')).lower())
            palavras_titulo = re.findall(r'\b[a-z]{3,}\b', titulo_norm)
            memes.update([p for p in palavras_titulo if p not in stopwords_norm])
            
        # LÓGICA 2: Abordagem IA (Artefatos)
        elif fonte_memes == "Artefatos Extraídos":
            onto = row.get('ontologia_ia', {})
            # Garante que é um dicionário antes de extrair
            if isinstance(onto, dict):
                teorias = onto.get('teorias_e_modelos', [])
                ferramentas = onto.get('ferramentas_e_artefatos', [])
                metodos = onto.get('metodos_e_tecnicas', [])
                
                # Previne erros caso a IA tenha retornado uma string em vez de lista
                if not isinstance(teorias, list): teorias = [teorias]
                if not isinstance(ferramentas, list): ferramentas = [ferramentas]
                if not isinstance(metodos, list): metodos = [metodos]
                
                todos_artefatos = teorias + ferramentas + metodos
                # Limpa e adiciona (mantemos o texto original sem quebrar palavras para preservar nomes compostos)
                memes.update([str(a).strip().capitalize() for a in todos_artefatos if str(a).strip()])

        return list(memes)

    df_copy = df_base.copy()
    df_copy['memes_todos'] = df_copy.apply(extrair_memes_completos, axis=1)
    
    df_explodido = df_copy.explode('memes_todos')
    df_explodido = df_explodido[df_explodido['memes_todos'].notna()]
    df_explodido = df_explodido[df_explodido['memes_todos'].astype(str).str.strip() != '']
    df_explodido['meme'] = df_explodido['memes_todos']

    if df_explodido.empty:
        return pd.DataFrame(), pd.DataFrame(), 0, 0, pd.DataFrame(), pd.DataFrame()

    fecundidade = df_explodido.groupby('meme')['titulo'].nunique().reset_index(name='fecundidade')
    
    df_mortos = fecundidade[fecundidade['fecundidade'] == 1][['meme']].rename(columns={'meme': 'Memes Mortos (1 Aparição)'})
    df_vivos = fecundidade[fecundidade['fecundidade'] > 1].sort_values('fecundidade', ascending=False).rename(columns={'meme': 'Memes Sobreviventes', 'fecundidade': 'Nº de Aparições'})

    mortalidade_count = len(df_mortos)
    sobreviventes_count = len(df_vivos)

    longevidade = df_explodido.groupby('meme').agg(
        ano_nascimento=('Ano', 'min'),
        ano_extincao=('Ano', 'max'),
        total_aparicoes=('titulo', 'nunique')
    ).reset_index()
    
    longevidade['tempo_vida_anos'] = longevidade['ano_extincao'] - longevidade['ano_nascimento']
    longevidade_valida = longevidade[longevidade['total_aparicoes'] > 1].copy()

    return fecundidade, longevidade_valida, mortalidade_count, sobreviventes_count, df_mortos, df_vivos

# =========================================================================
# --- NOVO LÓGICA SANKEY TEMPORAL (PALAVRAS-CHAVE) ---
# =========================================================================
@st.cache_data
def preparar_sankey_temporal(dados_lista, top_n, p1_range, p2_range, p3_range):
    """Mapeia a evolução temática (Bibliometrix Style) cruzando palavras-chave 
    através dos pesquisadores (Orientadores/Autores) que transitaram entre os períodos."""
    df = pd.DataFrame(dados_lista)
    df['Ano'] = pd.to_numeric(df.get('ano'), errors='coerce')
    
    # Extrai os anos iniciais e finais de cada período
    p1_y = (p1_range[0].year, p1_range[1].year)
    p2_y = (p2_range[0].year, p2_range[1].year)
    p3_y = (p3_range[0].year, p3_range[1].year)

    def get_period_df(df_base, years):
        return df_base[(df_base['Ano'] >= years[0]) & (df_base['Ano'] <= years[1])]

    df_p1 = get_period_df(df, p1_y)
    df_p2 = get_period_df(df, p2_y)
    df_p3 = get_period_df(df, p3_y)

    # Identifica o Top N de palavras de cada período isoladamente
    def get_top_kw(df_p, n):
        if df_p.empty or 'palavras_chave' not in df_p.columns: return []
        return df_p.explode('palavras_chave')['palavras_chave'].value_counts().head(n).index.tolist()

    top_kw1 = get_top_kw(df_p1, top_n)
    top_kw2 = get_top_kw(df_p2, top_n)
    top_kw3 = get_top_kw(df_p3, top_n)

    if not top_kw1 or not top_kw2 or not top_kw3: return [], [], [], []

    # Cria as etiquetas (nodes) únicas para cada período
    lbl_p1 = str(p1_y[0])
    lbl_p2 = str(p2_y[0])
    lbl_p3 = str(p3_y[0])
    
    nodes_labels = [f"{kw} ({lbl_p1})" for kw in top_kw1] + \
                   [f"{kw} ({lbl_p2})" for kw in top_kw2] + \
                   [f"{kw} ({lbl_p3})" for kw in top_kw3]
                   
    mapping = {label: i for i, label in enumerate(nodes_labels)}
    fluxos = []

    # Extrai o "DNA do Pesquisador": Quais palavras ele usou em um determinado período?
    def get_researcher_kws(df_period, top_kws):
        df_exp = df_period.explode('palavras_chave')
        df_exp = df_exp[df_exp['palavras_chave'].isin(top_kws)]
        
        researcher_kws = {}
        for _, row in df_exp.iterrows():
            kw = row['palavras_chave']
            ori = row.get('orientador')
            autores = row.get('autores', [])
            if isinstance(autores, str): autores = [autores]
            elif not isinstance(autores, list): autores = []
            
            # Une orientador e aluno como "vetores" de transferência de conhecimento
            researchers = [ori] + autores if ori else autores
            for r in researchers:
                if pd.notna(r) and str(r).strip() != "":
                    if r not in researcher_kws: researcher_kws[r] = set()
                    researcher_kws[r].add(kw)
        return researcher_kws

    rk_p1 = get_researcher_kws(df_p1, top_kw1)
    rk_p2 = get_researcher_kws(df_p2, top_kw2)
    rk_p3 = get_researcher_kws(df_p3, top_kw3)

    # Conecta as palavras-chave se os mesmos pesquisadores publicaram nelas entre os períodos
    def calc_flow(rk_src, rk_tgt, label_src, label_tgt):
        # Encontra quem publicou em ambos os períodos
        shared_researchers = set(rk_src.keys()).intersection(set(rk_tgt.keys()))
        links_dict = {}
        
        for r in shared_researchers:
            for ks in rk_src[r]:
                for kt in rk_tgt[r]:
                    pair = (f"{ks} ({label_src})", f"{kt} ({label_tgt})")
                    # Incrementa o "peso" do fluxo baseado na repetição do caminho
                    links_dict[pair] = links_dict.get(pair, 0) + 1
        
        for (src_node, tgt_node), val in links_dict.items():
            fluxos.append({'src': src_node, 'tgt': tgt_node, 'val': val})

    # Calcula os fluxos P1 -> P2 e P2 -> P3
    calc_flow(rk_p1, rk_p2, lbl_p1, lbl_p2)
    calc_flow(rk_p2, rk_p3, lbl_p2, lbl_p3)

    if not fluxos: return nodes_labels, [], [], []

    df_fluxos = pd.DataFrame(fluxos)
    
    # Mapeia os textos de volta para os índices numéricos exigidos pelo Plotly Sankey
    sources = df_fluxos['src'].map(mapping)
    targets = df_fluxos['tgt'].map(mapping)
    values = df_fluxos['val']

    return nodes_labels, sources.tolist(), targets.tolist(), values.tolist()# =========================================================================

def renderizar_nuvem_interativa_html_exploracao(word_freq_dict):
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

@st.cache_resource
def gerar_nodos_coocorrencia_agraph(dados_recorte, min_coocorrencia=1):
    G = nx.Graph()
    for d in dados_recorte:
        pks = d.get('palavras_chave', [])
        for pk in pks:
            if G.has_node(pk): G.nodes[pk]['count'] += 1
            else: G.add_node(pk, count=1, tipo='Conceito')
        for pk1, pk2 in itertools.combinations(pks, 2):
            if G.has_edge(pk1, pk2): G[pk1][pk2]['weight'] += 1
            else: G.add_edge(pk1, pk2, weight=1)

    G.remove_edges_from([(u, v) for u, v, attrs in G.edges(data=True) if attrs['weight'] < min_coocorrencia])
    G.remove_nodes_from(list(nx.isolates(G)))

    nodes, edges = [], []
    
    # Configuração da fonte com contorno (Halo)
    config_fonte = {"color": "black", "strokeWidth": 3, "strokeColor": "white"}
    
    for node, attrs in G.nodes(data=True):
        tam = min(10 + (attrs['count'] * 1.5), 50)
        nodes.append(Node(
            id=node, 
            label=node, 
            size=tam, 
            color='#2ECC71', 
            shape='dot', 
            title=f"{node}\nOcorrências: {attrs['count']}",
            font=config_fonte # <--- ADICIONADO AQUI
        ))

    for u, v, attrs in G.edges(data=True):
        # Transparência ajustada (150,150,150) garante contraste no preto (#1E1E1E) e no branco (#FFFFFF)
        edges.append(Edge(source=u, target=v, width=attrs['weight']*0.5, color="rgba(150, 150, 150, 0.6)", title=f"Co-ocorrências: {attrs['weight']}"))

    return nodes, edges

@st.cache_data
def preparar_csv_exportacao(dados):
    df = pd.DataFrame(dados)
    for col in ['autores', 'co_orientadores', 'palavras_chave']:
        if col in df.columns: df[col] = df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def calcular_metricas_complexas(dados):
    G = nx.Graph()
    for d in dados:
        doc = d.get('titulo')
        if not doc: continue
        G.add_node(doc, tipo='Documento')
        for a in d.get('autores', []): G.add_edge(doc, a)
        ori = d.get('orientador')
        if ori: G.add_edge(doc, ori)
        for pk in d.get('palavras_chave', []): G.add_edge(doc, pk)

    if G.number_of_nodes() == 0: return {}

    densidade = nx.density(G)
    graus = [d for n, d in G.degree()]
    links_stats = {'media': np.mean(graus), 'min': np.min(graus), 'max': np.max(graus), 'std': np.std(graus)}
    eficiencia = nx.global_efficiency(G)
    redundancia = 1 - eficiencia 
    pk = np.array(nx.degree_histogram(G))
    pk = pk / pk.sum()
    pk = pk[pk > 0]
    entropia = -np.sum(pk * np.log2(pk))
    clustering = nx.average_clustering(G)
    pagerank_dict = nx.pagerank(G)
    eigen_dict = nx.eigenvector_centrality(G, max_iter=1000, weight=None)
    constraint_dict = nx.constraint(G) 

    return {
        'densidade': densidade, 'links': links_stats, 'eficiencia': eficiencia, 'redundancia': redundancia,
        'entropia': entropia, 'clustering': clustering, 'pagerank_avg': np.mean(list(pagerank_dict.values())),
        'eigen_avg': np.mean(list(eigen_dict.values())), 'constraint_avg': np.mean(list(constraint_dict.values())),
        'n_nos': G.number_of_nodes()
    }

def preparar_exportacao_grafo(G, formato):
    output = io.BytesIO()
    G_export = G.copy()
    if formato == "GEXF (Gephi)":
        nx.write_gexf(G_export, output, encoding='utf-8')
        return output.getvalue(), "grafo_ufsc.gexf"
    elif formato == "GraphML":
        nx.write_graphml(G_export, output, encoding='utf-8')
        return output.getvalue(), "grafo_ufsc.graphml"
    elif formato == "JSON (Node-Link)":
        data = nx.node_link_data(G_export)
        return json.dumps(data, ensure_ascii=False).encode('utf-8'), "grafo_ufsc.json"


def obter_frequencias_texto(df_hist, fonte_nuvem):
    if fonte_nuvem == "Conceitos (Palavras-chave)":
        lista_c = []
        for lst in df_hist['palavras_chave']: lista_c.extend(lst)
        return dict(Counter(lista_c).most_common(100))
    else:
        textos = df_hist.get('resumo', pd.Series()).dropna().astype(str).tolist() if fonte_nuvem == "Resumos (Abstracts)" else df_hist['titulo'].dropna().astype(str).tolist()
        texto_completo = " ".join(textos).lower()
        texto_completo = re.sub(r'[^\w\s]', '', texto_completo)
        palavras = texto_completo.split()
        stopwords_pt = set(['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'uma', 'para', 'com', 'não', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'das', 'à', 'seu', 'sua', 'ou', 'nos', 'já', 'eu', 'também', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'sem', 'mesmo', 'aos', 'nas', 'me', 'esse', 'essa', 'num', 'nem', 'numa', 'pelos', 'pelas', 'este', 'esta', 'sobre', 'estudo', 'análise', 'proposta', 'uso', 'aplicação', 'desenvolvimento', 'modelo', 'sistema', 'avaliação', 'gestão', 'conhecimento', 'engenharia', 'objetivo', 'pesquisa', 'trabalho', 'resultados', 'método', 'foi', 'foram', 'são', 'ser', 'através', 'forma', 'apresenta'])
        palavras_limpas = [p for p in palavras if p not in stopwords_pt and len(p) > 2]
        return dict(Counter(palavras_limpas).most_common(100))


@st.cache_resource
def gerar_nodos_globais_agraph(dados_recorte, metodo_cor="Original (Categoria)", metodo_tamanho="Tamanho Fixo"):
    G = nx.Graph()
    for tese in dados_recorte:
        doc_id = tese['titulo']
        G.add_node(doc_id, label=doc_id[:30], tipo='Documento', nivel=tese.get('nivel_academico', 'N/A'), ano=tese.get('ano', 'N/A'))
        if tese.get('orientador'): 
            G.add_node(tese['orientador'], label=tese['orientador'], tipo='Orientador')
            G.add_edge(tese['orientador'], doc_id)
        for pk in tese.get('palavras_chave', []): 
            G.add_node(pk, label=pk, tipo='Conceito')
            G.add_edge(doc_id, pk)

    deg_cent, bet_cent, grau_abs = nx.degree_centrality(G), nx.betweenness_centrality(G), dict(G.degree())
    max_deg, max_bet, max_abs = max(deg_cent.values() or [1]), max(bet_cent.values() or [1]), max(grau_abs.values() or [1])

    comunidades = nx_comm.louvain_communities(G)
    legendas_comunidades = []
    paleta = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6']
    
    for i, comm in enumerate(comunidades):
        cor_com = paleta[i % len(paleta)]
        legendas_comunidades.append({"id": i+1, "cor": cor_com, "tamanho": len(comm)})
        for node in comm: 
            G.nodes[node]['color'] = cor_com
            G.nodes[node]['community'] = i + 1

    nodes_agraph, edges_agraph = [], []
    for node, attrs in G.nodes(data=True):
        tipo = attrs.get('tipo', 'Desconhecido')
        grau_atual = grau_abs.get(node, 0)
        
        if metodo_tamanho == "Grau Absoluto": tam = 10 + (grau_atual / max_abs) * 40
        elif metodo_tamanho == "Degree Centrality": tam = 10 + (deg_cent.get(node, 0) / max_deg) * 40
        elif metodo_tamanho == "Betweenness": tam = 10 + (bet_cent.get(node, 0) / max_bet) * 40
        else: tam = 20
        
        cor_final = attrs.get('color', ('#E74C3C' if tipo == 'Documento' else '#F39C12' if tipo == 'Orientador' else '#2ECC71'))
        formato = 'star' if tipo == 'Orientador' else 'square' if tipo == 'Documento' else 'dot'
        
# Configuração da fonte com contorno (Halo) para leitura em Light/Dark Mode
        config_fonte = {"color": "black", "strokeWidth": 3, "strokeColor": "white"}
        
        nodes_agraph.append(Node(
            id=node, 
            label=attrs['label'], 
            size=tam, 
            color=cor_final, 
            shape=formato, 
            title=f"{node}\nTipo: {tipo}",
            font=config_fonte # <--- ADICIONADO AQUI
        ))
    for u, v in G.edges():
        # Cor neutra universal e linha levemente mais espessa para visualização em ambos os temas
        edges_agraph.append(Edge(source=u, target=v, color="#95A5A6", width=0.8))

    return nodes_agraph, edges_agraph, legendas_comunidades, G


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
    closeness_cent = nx.closeness_centrality(G) # NOVO CÁLCULO
    
    lista = []
    for node, attrs in G.nodes(data=True):
        lista.append({
            'Entidade (Nó)': node,
            'Categoria': attrs.get('tipo', 'Desconhecido'),
            'Grau Absoluto': G.degree(node),
            'Degree Centrality': degree_cent.get(node, 0),
            'Betweenness': betweenness_cent.get(node, 0),
            'Closeness': closeness_cent.get(node, 0) # NOVA MÉTRICA
        })
    return pd.DataFrame(lista)


@st.cache_data
def preparar_dados_base_df(dados):
    df = pd.DataFrame(dados)
    df['Ano'] = pd.to_numeric(df.get('ano'), errors='coerce')
    df = df.dropna(subset=['Ano'])
    df['Ano'] = df['Ano'].astype(int)
    df['nivel_academico'] = df.get('nivel_academico', 'Outros').fillna('Outros')
    df['titulo'] = df.get('titulo', '').fillna('')
    df['orientador'] = df.get('orientador', 'Não informado').fillna('Não informado')
    return df


@st.cache_data(show_spinner=False)
def calcular_maturidade_rede(dados_completos, _sna_global):
    """Calcula os indicadores avançados de maturidade e robustez topológica."""
    
    # Reconstrói as conexões de forma ultrarrápida (sem atributos, só a estrutura)
    G = nx.Graph()
    for d in dados_completos:
        doc = d.get('titulo')
        if not doc: continue
        G.add_node(doc)
        for a in d.get('autores', []): G.add_edge(doc, a)
        ori = d.get('orientador')
        if ori: G.add_edge(doc, ori)
        for pk in d.get('palavras_chave', []): G.add_edge(doc, pk)
        mt = d.get('macrotema')
        if mt: G.add_edge(doc, mt)

    # Remove self-loops (nós conectados a si mesmos) pois quebram o algoritmo Rich-Club
    G.remove_edges_from(nx.selfloop_edges(G))

    # 1. Assortatividade (Quem se conecta com quem?)
    try:
        assortatividade = nx.degree_assortativity_coefficient(G)
    except Exception:
        assortatividade = 0.0

    # 2. Rich-Club Coefficient (Os Hubs se conversam?)
    try:
        rc = nx.rich_club_coefficient(G, normalized=False)
        # Pega a probabilidade de conexão apenas para o "Clube" dos top 20% maiores nós
        k_max = max(rc.keys()) if rc else 1
        k_target = int(k_max * 0.8)
        chaves_validas = [k for k in rc.keys() if k >= k_target]
        rich_club_val = rc[chaves_validas[0]] if chaves_validas else list(rc.values())[-1]
    except Exception:
        rich_club_val = 0.0

    # 3. Expoente Gamma da Lei de Potência (O grau de monopolização)
    try:
        graus = [d for n, d in G.degree() if d > 0]
        contagem = Counter(graus)
        k = np.array(list(contagem.keys()))
        Pk = np.array(list(contagem.values())) / len(graus)

        # Filtra k > 1 para focar na "cauda longa" da distribuição (onde a magia acontece)
        mask = k > 1
        log_k = np.log10(k[mask])
        log_Pk = np.log10(Pk[mask])

        if len(log_k) > 1:
            slope, _, _, _, _ = stats.linregress(log_k, log_Pk)
            gamma = abs(slope)
        else:
            gamma = 0.0
    except Exception:
        gamma = 0.0

    # 4. Correlação de Spearman (A barriga da curva 3D: Brokers vs Hubs)
    try:
        # Puxamos os dados já calculados para economizar 100% do processamento
        graus_list = [v.get('Grau Absoluto', 0) for v in _sna_global.values()]
        bet_list = [v.get('Betweenness', 0) for v in _sna_global.values()]
        spearman_rho, _ = stats.spearmanr(graus_list, bet_list)
    except Exception:
        spearman_rho = 0.0

    return {
        "Assortatividade": assortatividade,
        "Rich_Club": rich_club_val,
        "Gamma": gamma,
        "Spearman": spearman_rho
    }



@st.cache_data(show_spinner=False)
def plotar_grafico_3d_sna(sna_global, tipo_alvo, termo_destaque=None):
    """Gera um Scatter 3D topológico. Se termo_destaque for None, mostra a massa global."""
    dados_filtrados = []
    
    for node, metrics in sna_global.items():
        tipo_node = metrics.get('Tipo', '')
        
        # Filtro de Categoria
        if tipo_alvo in ['Orientador', 'Co-orientador']:
            if tipo_node not in ['Orientador', 'Co-orientador']: continue
        elif tipo_node != tipo_alvo:
            continue
            
        status = '🎯 Alvo Selecionado' if termo_destaque and node == termo_destaque else 'Ecossistema'
        
        dados_filtrados.append({
            'Item': node,
            'Grau': metrics.get('Grau Absoluto', 0),
            'Betweenness': metrics.get('Betweenness', 0.0),
            'Closeness': metrics.get('Closeness', 0.0),
            'Comunidade': str(metrics.get('Comunidade', 'N/A')),
            'Status': status
        })
        
    df = pd.DataFrame(dados_filtrados)
    if df.empty: return None
    
    # Limite de pontos para fluidez WebGL
    if len(df) > 2000:
        df = df.sort_values('Grau', ascending=False).head(2000)
        
    df['Tamanho'] = df['Grau'].apply(lambda x: max(x, 2))
    if termo_destaque:
        df.loc[df['Status'] == '🎯 Alvo Selecionado', 'Tamanho'] = df['Tamanho'].max() * 3 + 10

    fig = px.scatter_3d(
        df, x='Grau', y='Betweenness', z='Closeness',
        color='Comunidade', hover_name='Item',
        symbol='Status' if termo_destaque else None,
        size='Tamanho', size_max=30, opacity=0.7
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_title='Grau (Conexões)',
            yaxis_title='Betweenness (Ponte)',
            zaxis_title='Closeness (Proximidade)'
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

@st.cache_data(show_spinner=False)
def construir_perfis_similaridade(dados_lista):
    """
    Pré-computa o 'DNA acadêmico' de todas as entidades para comparação Jaccard.
    Separado de calcular_similares_rede para ser cacheado independentemente.
    Cache hit nas chamadas subsequentes — O(1) em vez de O(n) por busca.
    """
    perfis = {}
    tipos = {}
    niveis_docs = {}

    def add_feature(entidade, feature, tipo_entidade):
        if entidade not in perfis:
            perfis[entidade] = set()
            tipos[entidade] = tipo_entidade
        perfis[entidade].add(feature)

    for d in dados_lista:
        doc = d.get('titulo')
        if not doc: continue

        niveis_docs[doc] = d.get('nivel_academico', 'Outros')

        autores = d.get('autores', [])
        ori = d.get('orientador')
        cooris = d.get('co_orientadores', [])
        pks = d.get('palavras_chave', [])
        mt = d.get('macrotema')

        todas_entidades_doc = autores + ([ori] if ori else []) + cooris + pks + ([mt] if mt else [])

        # DNA do Documento: Todas as pessoas e conceitos atrelados a ele
        for ent in todas_entidades_doc:
            if ent: add_feature(doc, ent, 'Documento')

        # DNA do Autor: Seus orientadores, palavras-chave e temas que costuma escrever
        for a in autores:
            if ori: add_feature(a, ori, 'Autor')
            for pk in pks: add_feature(a, pk, 'Autor')
            if mt: add_feature(a, mt, 'Autor')

        # DNA dos Professores: Seus parceiros de banca, palavras-chave e temas dos alunos
        if ori:
            for pk in pks: add_feature(ori, pk, 'Orientador')
            if mt: add_feature(ori, mt, 'Orientador')
            for co in cooris: add_feature(ori, co, 'Orientador')

        for co in cooris:
            for pk in pks: add_feature(co, pk, 'Co-orientador')
            if mt: add_feature(co, mt, 'Co-orientador')
            if ori: add_feature(co, ori, 'Co-orientador')

        # DNA dos Conceitos: Outras palavras usadas juntas, macrotema e orientadores
        for pk in pks:
            if mt: add_feature(pk, mt, 'Palavra-chave')
            for pk2 in pks:
                if pk != pk2: add_feature(pk, pk2, 'Palavra-chave')

        # DNA do Macrotema: Suas palavras-chave base e orientadores principais
        if mt:
            for pk in pks: add_feature(mt, pk, 'Macrotema')
            if ori: add_feature(mt, ori, 'Macrotema')

    return perfis, tipos, niveis_docs


@st.cache_data
def calcular_similares_rede(termo_foco, tipo_busca, dados_completos):
    """Calcula os nós mais próximos usando o Índice de Jaccard (Similaridade de Vizinhança)."""

    perfis, tipos, niveis_docs = construir_perfis_similaridade(dados_completos)

    if termo_foco not in perfis:
        return {}
        
    perfil_foco = perfis[termo_foco]
    resultados = []
    
    # 2. Calcula a Similaridade (Índice de Jaccard) contra todo o resto da rede
    for node, features in perfis.items():
        if node == termo_foco: continue
        
        tipo_node = tipos[node]
        # Filtros de performance: só compara maçã com maçã
        if tipo_busca == 'Documento' and tipo_node != 'Documento': continue
        if tipo_busca == 'Autor' and tipo_node != 'Autor': continue
        if tipo_busca in ['Orientador', 'Co-orientador'] and tipo_node not in ['Orientador', 'Co-orientador']: continue
        if tipo_busca == 'Palavra-chave' and tipo_node != 'Palavra-chave': continue
        if tipo_busca == 'Macrotema' and tipo_node != 'Macrotema': continue
        
        intersecao_itens = perfil_foco.intersection(features)
        intersecao = len(intersecao_itens)
        if intersecao == 0: continue # Sem nada em comum = Similaridade 0
        
        uniao = len(perfil_foco.union(features))
        jaccard = intersecao / uniao
        
        resultados.append({
            'Item': node,
            'Similaridade (%)': round(jaccard * 100, 2),
            'Qtd. Traços': intersecao,
            'Traços em Comum': ", ".join(sorted(list(intersecao_itens))),
            'Nível': niveis_docs.get(node, 'Outros') if tipo_node == 'Documento' else None,
            'Tipo': tipo_node
        })
        
    # 3. Empacota os Top 5 de cada subcategoria
    retorno = {}
    if tipo_busca == 'Documento':
        teses = sorted([x for x in resultados if 'Tese' in str(x['Nível'])], key=lambda x: x['Similaridade (%)'], reverse=True)[:5]
        diss = sorted([x for x in resultados if 'Disserta' in str(x['Nível'])], key=lambda x: x['Similaridade (%)'], reverse=True)[:5]
        retorno['Teses'] = teses
        retorno['Dissertações'] = diss
    elif tipo_busca == 'Autor':
        retorno['Autores'] = sorted(resultados, key=lambda x: x['Similaridade (%)'], reverse=True)[:5]
    elif tipo_busca in ['Orientador', 'Co-orientador']:
        retorno['Professores'] = sorted(resultados, key=lambda x: x['Similaridade (%)'], reverse=True)[:5]
    elif tipo_busca == 'Palavra-chave':
        retorno['Palavras-chave'] = sorted(resultados, key=lambda x: x['Similaridade (%)'], reverse=True)[:5]
    elif tipo_busca == 'Macrotema':
        retorno['Macrotemas'] = sorted(resultados, key=lambda x: x['Similaridade (%)'], reverse=True)[:5]
        
    return retorno


# Conexão com o Banco de Grafos
@st.cache_resource
def conectar_neo4j():
    uri, user, pwd = get_neo4j_credentials(required=True)
    return GraphDatabase.driver(uri, auth=(user, pwd))

def extrair_subgrafo_neo4j(driver, nome_programa, limite=50):
    """Consulta diretamente no Neo4j o ecossistema de um PPG específico."""
    
    # Esta query Cypher traz o Documento, o Orientador e as Palavras-chave dele
    query = """
    MATCH (p:Programa {nome: $programa})<-[:VINCULADO_A]-(d:Documento)
    OPTIONAL MATCH (o:Orientador)-[:ORIENTOU]->(d)
    OPTIONAL MATCH (d)-[:ABORDA]->(c:Conceito)
    RETURN d.titulo AS documento, d.ano AS ano, 
           o.nome AS orientador, 
           collect(c.nome) AS conceitos
    LIMIT $limite
    """
    
    nodes_agraph = []
    edges_agraph = []
    nos_processados = set()
    
    config_fonte = {"color": "black", "strokeWidth": 3, "strokeColor": "white"}
    
    with driver.session() as session:
        resultados = session.run(query, programa=nome_programa, limite=limite)
        
        for record in resultados:
            doc = record["documento"]
            ori = record["orientador"]
            conceitos = record["conceitos"]
            
            # 1. Cria o Nó do Documento
            if doc not in nos_processados:
                nodes_agraph.append(Node(id=doc, label=doc[:30], size=20, color="#E74C3C", shape="square", font=config_fonte))
                nos_processados.add(doc)
            
            # 2. Cria o Nó do Orientador e a Aresta
            if ori and ori not in nos_processados:
                nodes_agraph.append(Node(id=ori, label=ori, size=35, color="#F39C12", shape="star", font=config_fonte))
                nos_processados.add(ori)
                edges_agraph.append(Edge(source=ori, target=doc, color="#95A5A6", width=1))
            
            # 3. Cria os Nós dos Conceitos e as Arestas
            for c in conceitos:
                if c and c not in nos_processados:
                    nodes_agraph.append(Node(id=c, label=c, size=15, color="#2ECC71", shape="dot", font=config_fonte))
                    nos_processados.add(c)
                edges_agraph.append(Edge(source=doc, target=c, color="#95A5A6", width=0.5))
                
    return nodes_agraph, edges_agraph

def navegar_para(novo_tipo, novo_termo): 
    st.session_state.update({'busca_tipo': novo_tipo, 'busca_termo': novo_termo})

# --- FUNÇÕES DE BACKEND (EXTRAÇÃO E BUSCA) ---
def _normalizar_nivel(nivel) -> str:
    """
    Normaliza o campo nivel_academico para um dos 4 valores canônicos:
    'Teses', 'Dissertações', 'TCC', 'Outros'.
    Case-insensitive. Robusto a variações de nomenclatura.
    """
    if not nivel:
        return 'Outros'
    nivel_str = str(nivel).lower().strip()
    if 'tese' in nivel_str or 'doutor' in nivel_str or 'thesis' in nivel_str:
        return 'Teses'
    if 'disserta' in nivel_str or 'mestrado' in nivel_str or 'dissertation' in nivel_str or 'master' in nivel_str:
        return 'Dissertações'
    if 'tcc' in nivel_str or 'conclus' in nivel_str or 'gradua' in nivel_str:
        return 'TCC'
    return 'Outros'

def _extrair_entidades_por_tipo(doc, tipo):
    if tipo == "Orientador":
        ori = doc.get("orientador")
        return [ori] if ori else []
    if tipo == "Co-orientador":
        return list(set([co for co in doc.get("co_orientadores", []) if co]))
    if tipo == "Palavra-chave":
        return list(set([pk for pk in doc.get("palavras_chave", []) if pk]))
    if tipo == "Macrotema":
        mt = doc.get("macrotema", "Multidisciplinar / Transversal")
        return [mt]
    return []

def _renderizar_tabela_ql(resumo_df, titulo):
    if resumo_df.empty:
        return

    st.write(titulo)
    max_tot = int(resumo_df["Total"].max()) if not resumo_df.empty else 100

    def color_ql(val):
        try:
            v = float(val)
            if v > 1:
                return "color: #00FF00; font-weight: bold;"
            if v < 1:
                return "color: #FF4B4B;"
            return "color: #F8E71C;"
        except Exception:
            return ""

    styler = resumo_df.style.map(color_ql, subset=["Valor QL"])
    st.dataframe(
        styler,
        width='stretch',
        hide_index=True,
        column_config={
            "Total": st.column_config.ProgressColumn("Total", min_value=0, max_value=max_tot, format="%d"),
            "Teses": st.column_config.ProgressColumn("Teses", min_value=0, max_value=max_tot, format="%d"),
            "Dissertações": st.column_config.ProgressColumn("Dissertações", min_value=0, max_value=max_tot, format="%d"),
            "Outros": st.column_config.ProgressColumn("Outros", min_value=0, max_value=max_tot, format="%d"),
            "Valor QL": st.column_config.NumberColumn("Valor QL", format="%.2f"),
        },
    )
    st.caption("*Nota: QL > 1 indica especialização acima da média global da base.*")
    st.markdown("<br>", unsafe_allow_html=True)

def gerar_tabela_ql_cruzado_perfil(docs_perfil, dados_totais, tipos_alvo, titulo="**📊 Frequência Temática e Especialização (QL):**"):
    """Gera tabela de QL para entidades-alvo de um perfil (Orientador/Co/PK/Macrotema)."""
    if not docs_perfil or not dados_totais or not tipos_alvo:
        return

    total_docs_global = len(dados_totais)
    total_docs_perfil = len(docs_perfil)
    if total_docs_global == 0 or total_docs_perfil == 0:
        return

    contagens_globais = {}
    for tipo in tipos_alvo:
        c = Counter()
        for d in dados_totais:
            for entidade in _extrair_entidades_por_tipo(d, tipo):
                c[entidade] += 1
        contagens_globais[tipo] = c

    linhas = []
    for tipo in tipos_alvo:
        agregados = {}
        for d in docs_perfil:
            nivel = _normalizar_nivel(d.get("nivel_academico"))
            for entidade in _extrair_entidades_por_tipo(d, tipo):
                if entidade not in agregados:
                    agregados[entidade] = {"Teses": 0, "Dissertações": 0, "Outros": 0}
                agregados[entidade][nivel] += 1

        for entidade, conts in agregados.items():
            total_local = conts["Teses"] + conts["Dissertações"] + conts["Outros"]
            total_global_entidade = contagens_globais[tipo].get(entidade, 0)
            if total_global_entidade == 0:
                ql = 0.0
            else:
                ql = (total_local / total_docs_perfil) / (total_global_entidade / total_docs_global)

            linhas.append({
                "Entidade": entidade,
                "Tipo": tipo,
                "Teses": conts["Teses"],
                "Dissertações": conts["Dissertações"],
                "Outros": conts["Outros"],
                "Total": total_local,
                "Valor QL": round(ql, 2),
            })

    if not linhas:
        return

    resumo = pd.DataFrame(linhas)
    resumo = resumo[["Entidade", "Tipo", "Teses", "Dissertações", "Outros", "Total", "Valor QL"]]
    resumo = resumo.sort_values(by=["Tipo", "Valor QL", "Total"], ascending=[True, False, False])
    _renderizar_tabela_ql(resumo, titulo)

@st.cache_data(show_spinner=False)
def gerar_base_boxplot_ql(dados_lista, tipo_entidade, entidades):
    """Gera valores de QL por nível acadêmico para alimentar boxplot."""
    if not dados_lista or not entidades:
        return pd.DataFrame()

    docs_validos = [d for d in dados_lista if d.get("titulo")]
    if not docs_validos:
        return pd.DataFrame()

    total_docs = len(docs_validos)
    docs_por_nivel = {"Teses": 0, "Dissertações": 0, "Outros": 0}
    for d in docs_validos:
        docs_por_nivel[_normalizar_nivel(d.get("nivel_academico"))] += 1

    linhas = []
    for entidade in entidades:
        docs_entidade = []
        for d in docs_validos:
            if entidade in _extrair_entidades_por_tipo(d, tipo_entidade):
                docs_entidade.append(d)

        total_entidade = len(docs_entidade)
        if total_entidade == 0:
            continue

        local_por_nivel = {"Teses": 0, "Dissertações": 0, "Outros": 0}
        for d in docs_entidade:
            local_por_nivel[_normalizar_nivel(d.get("nivel_academico"))] += 1

        for nivel in ["Teses", "Dissertações", "Outros"]:
            base_global_nivel = docs_por_nivel.get(nivel, 0)
            if base_global_nivel == 0:
                ql = 0.0
            else:
                ql = (local_por_nivel[nivel] / total_entidade) / (base_global_nivel / total_docs)
            linhas.append({
                "Entidade": entidade,
                "Nível": nivel,
                "Valor QL": round(ql, 4),
            })

    return pd.DataFrame(linhas)

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
        except Exception: return ''

    styler = resumo.style.map(color_ql, subset=['QL (Especialização)'])
    
    st.dataframe(
        styler, 
        width='stretch', 
        hide_index=True,
        column_config={
            "Total": st.column_config.ProgressColumn("Total", min_value=0, max_value=max_tot, format="%d"),
            "Teses": st.column_config.ProgressColumn("Teses", min_value=0, max_value=max_tot, format="%d"),
            "Dissertações": st.column_config.ProgressColumn("Dissertações", min_value=0, max_value=max_tot, format="%d"),
            "QL (Especialização)": st.column_config.NumberColumn("QL (Especialização)", format="%.2f")
        }
    )
    st.markdown("<br>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False) # Desativamos o spinner padrão para usar nossa barra customizada
def calcular_sna_global(dados):
    # 1. Inicia a barra de progresso no topo
    progresso_texto = "🚀 Iniciando análise de rede complexa..."
    barra = st.progress(0, text=progresso_texto)
    
    G = nx.Graph()
    total_docs = len(dados)
    
    # 2. Construção do Grafo (20% do progresso)
    for i, d in enumerate(dados):
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
            
        for pk in d.get('palavras_chave', []): 
            G.add_node(pk, tipo='Palavra-chave')
            G.add_edge(doc, pk)
            
        mt = d.get('macrotema')
        if mt: 
            G.add_node(mt, tipo='Macrotema')
            G.add_edge(doc, mt)
            
        # Atualiza a barra durante a montagem (a cada 500 docs para não travar a UI)
        if i % 500 == 0:
            percentual = int((i / total_docs) * 20)
            barra.progress(percentual, text=f"📦 Mapeando Ecossistema: {i}/{total_docs} documentos processados...")

    # 3. Cálculo de Degree (30% do progresso)
    barra.progress(30, text="📐 Calculando Centralidade de Grau (Conexões diretas)...")
    deg_cent = nx.degree_centrality(G)
    grau_abs = dict(G.degree())
    
    # 4. Cálculo de Betweenness (60% do progresso - Parte mais pesada)
    barra.progress(45, text="🌉 Calculando Betweenness (Intermediação)... Isso pode levar alguns segundos...")
    total_nos = G.number_of_nodes()
    if total_nos > 1500:
        k_aproximado = min(250, max(50, int(math.sqrt(total_nos) * 4)))
        bet_cent = nx.betweenness_centrality(G, k=k_aproximado, seed=42)
    else:
        bet_cent = nx.betweenness_centrality(G)
    
    # 5. Cálculo de Closeness (80% do progresso)
    barra.progress(75, text="🎯 Calculando Closeness (Proximidade Central)...")
    close_cent = nx.closeness_centrality(G)

    # 5.5 Cálculo de Clustering (Densidade da Vizinhança) - NOVO
    barra.progress(80, text="🕸️ Calculando Densidade Local (Clustering)...")
    clust_cent = nx.clustering(G)
    
    # 6. Identificação de Comunidades (90% do progresso)
    barra.progress(85, text="🏘️ Detectando Clusters e Comunidades (Algoritmo de Louvain)...")
    try:
        mapa_comunidades = {node: i+1 for i, comm in enumerate(nx_comm.louvain_communities(G)) for node in comm}
    except Exception:
        mapa_comunidades = {}
        
    # 7. Finalização e Ranking (100%)
    barra.progress(95, text="📊 Gerando rankings e consolidando métricas SNA...")
    rank_bet = {node: rank+1 for rank, (node, _) in enumerate(sorted(bet_cent.items(), key=lambda x: x[1], reverse=True))}

    resultado = {node: {
        'Tipo': G.nodes[node].get('tipo', 'Desconhecido'),
        'Grau Absoluto': grau_abs.get(node, 0), 
        'Degree Centrality': deg_cent.get(node, 0), 
        'Betweenness': bet_cent.get(node, 0), 
        'Closeness': close_cent.get(node, 0),
        'Clustering': clust_cent.get(node, 0), # <--- NOVA LINHA
        'Comunidade': mapa_comunidades.get(node, 'N/A'), 
        'Ranking Global': rank_bet.get(node, 'N/A')
    } for node in G.nodes()}

    # Remove a barra da tela ao finalizar
    barra.empty()
    
    return resultado

@st.cache_resource
def _construir_grafo_historico(dados_lista: tuple) -> nx.Graph:
    """
    Constrói o grafo completo com metadado de ano em cada aresta.
    Cacheada separadamente de gerar_orbita_local para reutilizar entre frames de animação.

    Recebe dados_lista como tuple de tuplas (para ser hashável pelo st.cache_resource).
    """
    G = nx.Graph()
    for d in dados_lista:
        titulo, ano_raw, orientador, autores, cooris, pks, macrotema = d
        try:
            ano = int(ano_raw) if ano_raw and str(ano_raw).isdigit() else 0
        except (ValueError, TypeError):
            ano = 0

        if not titulo:
            continue

        G.add_node(titulo, tipo='Documento', ano=ano)

        for a in autores:
            if a:
                G.add_node(a, tipo='Autor')
                G.add_edge(titulo, a, ano=ano)

        if orientador:
            G.add_node(orientador, tipo='Orientador')
            G.add_edge(titulo, orientador, ano=ano)

        for co in cooris:
            if co:
                G.add_node(co, tipo='Co-orientador')
                G.add_edge(titulo, co, ano=ano)

        for pk in pks:
            if pk:
                G.add_node(pk, tipo='Conceito')
                G.add_edge(titulo, pk, ano=ano)

        if macrotema:
            G.add_node(macrotema, tipo='Macrotema')
            G.add_edge(titulo, macrotema, ano=ano)

    return G


@st.cache_resource
def gerar_orbita_local(termo_foco, tipo_busca, profundidade=1, _sna_global=None, metodo_tamanho="Tamanho Fixo", ano_limite=2026, dados_completos=None):
    """Motor in-memory de renderização visual (Fallback sem Neo4j para TCCs)."""
    if not dados_completos: return [], []

    # Constrói ou recupera do cache o grafo completo datado
    G_completo = _construir_grafo_historico(tuple(
        (d.get('titulo', ''), d.get('ano', 0), d.get('orientador', ''),
         tuple(d.get('autores', [])), tuple(d.get('co_orientadores', [])),
         tuple(d.get('palavras_chave', [])), d.get('macrotema', ''))
        for d in dados_completos
    ))

    # Filtra o grafo pelo ano_limite de forma eficiente
    arestas_no_periodo = [
        (u, v) for u, v, data in G_completo.edges(data=True)
        if data.get('ano', 0) <= ano_limite
    ]

    if not arestas_no_periodo:
        return [], []

    G = G_completo.edge_subgraph(arestas_no_periodo).copy()

    if termo_foco not in G:
        return [], []

    # 2. Extrai o Ego-Graph de acordo com a profundidade
    ego_net = nx.ego_graph(G, termo_foco, radius=profundidade)
    
    # 3. Mapeamento Estético (Idêntico ao Neo4j)
    nodes_agraph = []
    edges_agraph = []
    config_fonte = {"color": "white", "strokeWidth": 4, "strokeColor": "#1E1E1E", "face": "sans-serif", "size": 14}
    
    for node, attrs in ego_net.nodes(data=True):
        tipo = attrs.get('tipo', 'Desconhecido')
        
        tam = 20
        if node == termo_foco: tam = 45
        elif _sna_global and metodo_tamanho != "Tamanho Fixo":
            valor_metrica = _sna_global.get(node, {}).get(metodo_tamanho, 0.1)
            tam = 15 + (valor_metrica * 30)
            
        cor_base = '#9B59B6'
        if tipo == 'Documento': cor_base = '#E74C3C'
        elif tipo == 'Autor': cor_base = '#3498DB'
        elif tipo in ['Orientador', 'Co-orientador']: cor_base = '#F39C12'
        elif tipo == 'Conceito': cor_base = '#2ECC71'
            
        cor = '#FFFFFF' if node == termo_foco else cor_base
        formato = 'diamond' if node == termo_foco else ('star' if tipo in ['Orientador', 'Co-orientador'] else 'square' if tipo == 'Documento' else 'triangle' if tipo == 'Conceito' else 'dot')
        
        rotulo = str(node)[:30] + "..." if len(str(node)) > 30 and tipo == 'Documento' else str(node)
        nodes_agraph.append(Node(id=node, label=rotulo, size=tam, color=cor, title=f"{node}\\nTipo: {tipo}", shape=formato, font=config_fonte))
        
    for u, v in ego_net.edges():
        edges_agraph.append(Edge(source=u, target=v, color="#95A5A6", width=1.0))
        
    return nodes_agraph, edges_agraph

@st.cache_resource
def gerar_orbita_neo4j(_driver, termo_foco, tipo_busca, profundidade=1, _sna_global=None, metodo_tamanho="Tamanho Fixo", ano_limite=2026, titulos_validos=None):
    """
    Motor de renderização visual que extrai o subgrafo do Neo4j com filtro temporal e de PPG.
    """
    label_map = {
        "Documento": ("Documento", "titulo"),
        "Autor": ("Autor", "nome"),
        "Orientador": ("Orientador", "nome"),
        "Co-orientador": ("Orientador", "nome"),
        "Palavra-chave": ("Conceito", "nome")
    }
    
    # ✅ Filtro para garantir que o documento pertença ao PPG selecionado
    filtro_titulos = "AND d.titulo IN $titulos_validos" if titulos_validos else ""
    
    if tipo_busca == "Macrotema":
        query = f"""
        MATCH path = (n:Documento {{macrotema: $termo}})-[*1..{profundidade}]-(m)
        WHERE all(d IN nodes(path) WHERE NOT d:Documento OR (toInteger(d.ano) <= $ano_limite {filtro_titulos}))
        RETURN path LIMIT 300
        """
    else:
        label, prop = label_map.get(tipo_busca, ("Documento", "titulo"))
        query = f"""
        MATCH path = (n:{label} {{{prop}: $termo}})-[*1..{profundidade}]-(m)
        WHERE all(d IN nodes(path) WHERE NOT d:Documento OR (toInteger(d.ano) <= $ano_limite {filtro_titulos}))
        RETURN path LIMIT 300
        """
        
    nodes_agraph = []
    edges_agraph = []
    nos_processados = set()
    arestas_processadas = set()
    
    config_fonte = {"color": "white", "strokeWidth": 4, "strokeColor": "#1E1E1E", "face": "sans-serif", "size": 14}
    
    try:
        with _driver.session() as session:
            lista_para_banco = list(titulos_validos) if titulos_validos else None

            resultados = session.run(query, termo=termo_foco, ano_limite=int(ano_limite), titulos_validos=lista_para_banco)
            for record in resultados:
                path = record["path"]
                for node in path.nodes:
                    node_id = node.get("titulo") or node.get("nome")
                    if not node_id or node_id in nos_processados: continue
                    
                    labels = list(node.labels)
                    tipo = labels[0] if labels else "Desconhecido"
                    
                    tam = 20
                    if node_id == termo_foco: tam = 45
                    elif _sna_global and metodo_tamanho != "Tamanho Fixo":
                        valor_metrica = _sna_global.get(node_id, {}).get(metodo_tamanho, 0.1)
                        tam = 15 + (valor_metrica * 30)
                    
                    cor = '#FFFFFF' if node_id == termo_foco else ('#E74C3C' if tipo == 'Documento' else '#3498DB' if tipo == 'Autor' else '#F39C12' if tipo == 'Orientador' else '#2ECC71' if tipo == 'Conceito' else '#9B59B6')
                    formato = 'diamond' if node_id == termo_foco else ('star' if tipo == 'Orientador' else 'square' if tipo == 'Documento' else 'triangle' if tipo == 'Conceito' else 'dot')
                    
                    rotulo = str(node_id)[:30] + "..." if len(str(node_id)) > 30 and tipo == 'Documento' else str(node_id)
                    nodes_agraph.append(Node(id=node_id, label=rotulo, size=tam, color=cor, title=f"{node_id}\nTipo: {tipo}", shape=formato, font=config_fonte))
                    nos_processados.add(node_id)
                    
                for rel in path.relationships:
                    n1 = rel.start_node.get("titulo") or rel.start_node.get("nome")
                    n2 = rel.end_node.get("titulo") or rel.end_node.get("nome")
                    if n1 in nos_processados and n2 in nos_processados:
                        edge_id = f"{n1}-{n2}"
                        if edge_id not in arestas_processadas:
                            edges_agraph.append(Edge(source=n1, target=n2, color="#95A5A6", width=1.0))
                            arestas_processadas.add(edge_id)
    except Exception as e:
        st.error(f"Erro na extração: {e}")
    return nodes_agraph, edges_agraph

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
        except Exception: return ''
        
    styler2 = resumo.style.map(color_ql2, subset=['QL (Especialização)'])
    
    st.dataframe(
        styler2, 
        width='stretch', 
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
        response = generate_content(
            prompt=prompt,
            api_key=api_key,
            model_candidates=DEFAULT_TEXT_MODELS,
            temperature=0.3,
        )
        return response_text(response).strip().replace('**', '').replace('"', '')
    except Exception as e:
        return f"Não foi possível gerar a síntese dinâmica no momento. (Aviso: {e})"


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
def _normalizar_documentos(dados: list) -> list:
    """
    Garante tipos consistentes em todos os documentos.
    Chamada pelos loaders para evitar bugs de tipo downstream.
    Modifica a lista in-place para economizar memória e retorna ela.
    """
    for d in dados:
        # Ano: int quando possível, None caso contrário
        ano_raw = d.get('ano')
        if ano_raw is not None:
            try:
                d['ano'] = int(str(ano_raw).strip())
            except (ValueError, TypeError):
                d['ano'] = None

        # Listas: garante list, nunca None
        for campo in ('autores', 'co_orientadores', 'palavras_chave'):
            if not isinstance(d.get(campo), list):
                d[campo] = []

        # Strings: garante string vazia em vez de None
        for campo in ('titulo', 'orientador', 'macrotema', 'resumo',
                      'programa_origem', 'nivel_academico', 'url'):
            if d.get(campo) is None:
                d[campo] = ''

        # Palavras-chave: remove itens None ou string vazia
        d['palavras_chave'] = [
            pk for pk in d['palavras_chave']
            if pk and str(pk).strip()
        ]
    return dados


@st.cache_data(show_spinner="📚 Carregando base de teses e dissertações...", ttl=3600)
def carregar_base_consolidada():
    try:
        with gzip.open('base_consolidada_ufsc.json.gz', 'rt', encoding='utf-8') as f:
            dados = json.load(f)
        return _normalizar_documentos(dados)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        st.error(f"⚠️ Arquivo 'base_consolidada_ufsc.json.gz' está corrompido: {e}")
        return []
    except Exception as e:
        st.error(f"⚠️ Erro inesperado ao carregar base consolidada: {e}")
        return []


@st.cache_data(show_spinner="📚 Carregando base de TCCs...", ttl=3600)
def carregar_base_tcc():
    try:
        with gzip.open('base_tcc_ufsc.json.gz', 'rt', encoding='utf-8') as f:
            dados = json.load(f)
        return _normalizar_documentos(dados)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        st.error(f"⚠️ Arquivo 'base_tcc_ufsc.json.gz' está corrompido: {e}")
        return []
    except Exception as e:
        st.error(f"⚠️ Erro inesperado ao carregar base de TCCs: {e}")
        return []


# --- FUNÇÃO GERADORA DO MAPA TEMÁTICO (BIBLIOMETRIX STYLE) ---
def plotar_mapa_tematico(df_plot, x_col, y_col, size_col, label_col, title):
    # Usamos a média para desenhar a cruz dos quadrantes no centro de massa dos dados
    x_mid = df_plot[x_col].mean()
    y_mid = df_plot[y_col].mean()

    fig = px.scatter(
        df_plot, x=x_col, y=y_col, size=size_col, text=label_col, color=label_col,
        size_max=70, title=title
    )
    
    # Ajuste visual das bolhas e rótulos
    fig.update_traces(
        textposition='top center', 
        marker=dict(opacity=0.7, line=dict(width=1, color='White')),
        textfont=dict(size=11, color='white')
    )

    # Linhas tracejadas dividindo os quadrantes
    fig.add_vline(x=x_mid, line_dash="dash", line_color="#BDC3C7", line_width=2)
    fig.add_hline(y=y_mid, line_dash="dash", line_color="#BDC3C7", line_width=2)

    # Coordenadas extremas para posicionar as legendas dos quadrantes
    x_min, x_max = df_plot[x_col].min(), df_plot[x_col].max()
    y_min, y_max = df_plot[y_col].min(), df_plot[y_col].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05

    # Textos fixos nos quatro cantos
    annotations = [
        dict(x=x_max, y=y_max, text="<b>Temas Motores</b><br>(Alta Centralidade e Densidade)", showarrow=False, font=dict(color="#BDC3C7", size=13), xanchor="right", yanchor="top"),
        dict(x=x_min, y=y_max, text="<b>Temas de Nicho</b><br>(Alta Densidade, Baixa Centralidade)", showarrow=False, font=dict(color="#BDC3C7", size=13), xanchor="left", yanchor="top"),
        dict(x=x_min, y=y_min, text="<b>Temas Emergentes / Declínio</b><br>(Baixa Centralidade e Densidade)", showarrow=False, font=dict(color="#BDC3C7", size=13), xanchor="left", yanchor="bottom"),
        dict(x=x_max, y=y_min, text="<b>Temas Básicos</b><br>(Alta Centralidade, Baixa Densidade)", showarrow=False, font=dict(color="#BDC3C7", size=13), xanchor="right", yanchor="bottom")
    ]

    # Desativa grid padrão para dar aspecto de mapa estratégico
    fig.update_layout(
        annotations=annotations,
        showlegend=False,
        xaxis_title="Grau de Relevância (Betweenness Centrality)",
        yaxis_title="Grau de Desenvolvimento (Densidade / Volume)",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        height=750,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


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

@st.cache_data
def carregar_catalogo_tcc_frontend():
    try:
        with open('mapa_colecoes_tcc.json', 'r', encoding='utf-8') as f: 
            return json.load(f)
    except FileNotFoundError:
        return []

def calcular_raridade_semantica(doc_pks, contagem_pks_global, total_docs):
    """Calcula o IDF médio das palavras-chave de um documento."""
    if not doc_pks or total_docs == 0: return 0.0
    idfs = []
    for pk in doc_pks:
        freq = contagem_pks_global.get(pk, 1) # O mínimo é 1 pra evitar divisão por 0
        idf = math.log10(total_docs / freq)
        idfs.append(idf)
    return sum(idfs) / len(idfs)