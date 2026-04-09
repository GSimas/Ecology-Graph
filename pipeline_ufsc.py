import json
import re
import unicodedata
import gzip
from collections import defaultdict
from sickle import Sickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import requests
import difflib
import time
from neo4j import GraphDatabase

from app_config import get_gemini_api_key, get_neo4j_credentials
from gemini_utils import DEFAULT_TEXT_MODELS, generate_content, response_text

# --- MOTOR DE INGESTÃO NEO4J ---
def popular_banco_grafos(dados, uri="neo4j://localhost:7687", user="neo4j", password="sua_senha_aqui"):
    print("\n--- 🌐 INICIANDO MIGRAÇÃO PARA O NEO4J ---")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    # Consultas Cypher para criar o ecossistema
    query_estrutural = """
    UNWIND $batch AS doc
    
    // 1. Cria a Grande Área e o Programa
    MERGE (eco:Ecossistema {nome: doc.ecossistema_afinidade})
    MERGE (p:Programa {nome: doc.programa_origem})
    MERGE (p)-[:PERTENCE_A]->(eco)
    
    // 2. Cria o Documento (Tese/Dissertação)
    MERGE (d:Documento {titulo: doc.titulo})
    SET d.ano = doc.ano, 
        d.nivel = doc.nivel_academico, 
        d.macrotema = doc.macrotema,
        d.resumo = doc.resumo
        
    MERGE (d)-[:VINCULADO_A]->(p)
    
    // 3. Conecta Autores e Conceitos
    FOREACH (autor_nome IN doc.autores |
        MERGE (a:Autor {nome: autor_nome})
        MERGE (a)-[:ESCREVEU]->(d)
    )
    
    FOREACH (pk IN doc.palavras_chave |
        MERGE (c:Conceito {nome: pk})
        MERGE (d)-[:ABORDA]->(c)
    )
    """
    
    # Query separada para o Orientador (para evitar nós em branco se vier nulo)
    query_orientador = """
    UNWIND $batch AS doc
    WITH doc WHERE doc.orientador IS NOT NULL AND doc.orientador <> ''
    MATCH (d:Documento {titulo: doc.titulo})
    MERGE (o:Orientador {nome: doc.orientador})
    MERGE (o)-[:ORIENTOU]->(d)
    """

    try:
        with driver.session() as session:
            # Dica de ouro: Cria índices para a injeção ser ultra rápida
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:Documento) ON (d.titulo)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (c:Conceito) ON (c.nome)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (o:Orientador) ON (o.nome)")
            
            print("   [+] Injetando Nós e Relacionamentos no Banco...")
            session.run(query_estrutural, batch=dados)
            session.run(query_orientador, batch=dados)
            print("   [+] Injeção concluída com sucesso! O Ecossistema está vivo.")
    except Exception as e:
        print(f"   [!] Erro ao conectar ou injetar no Neo4j: {e}")
    finally:
        driver.close()

# --- INTEGRAÇÃO OFICIAL CAPES SUCUPIRA ---
def carregar_catalogo_capes_ufsc():
    """Baixa todos os programas da UFSC direto da CAPES para classificação oficial."""
    print("📡 Baixando taxonomia oficial da CAPES (id-ies: 4362)...")
    url = "https://apigw-proxy.capes.gov.br/observatorio/data/observatorio/ppg"
    headers = {"Accept": "application/json"}
    programas_capes = {}
    page = 0
    
    while True:
        try:
            resposta = requests.get(url, params={"query": "id-ies:(4362)", "page": page, "size": 100}, headers=headers, timeout=15)
            if resposta.status_code != 200: break
            
            dados = resposta.json()
            resultados = dados if isinstance(dados, list) else dados.get('content', dados.get('data', []))
            if not resultados: break
            
            for ppg in resultados:
                nome_capes = ppg.get("nome", "").strip().upper()
                nome_norm = ''.join(c for c in unicodedata.normalize('NFD', nome_capes) if unicodedata.category(c) != 'Mn')
                # Salvamos a Grande Área como o Ecossistema principal
                programas_capes[nome_norm] = ppg.get("nomeAreaConhecimento", "Multidisciplinar / Sem Área")
            
            if len(resultados) < 100: break
            page += 1
        except Exception as e:
            print(f" [!] Erro ao conectar na CAPES: {e}")
            break
            
    print(f"✅ Catálogo CAPES carregado com {len(programas_capes)} programas.")
    return programas_capes

def obter_ecossistema_capes(nome_prog, catalogo_capes):
    """Encontra a Grande Área do PPG na base da CAPES usando Inteligência de Strings."""
    if not catalogo_capes: return "Multidisciplinar / Transversal"
    
    # Validação para não processar strings de formato incorreto
    if not isinstance(nome_prog, str):
        return "Multidisciplinar / Transversal"
        
    nome_limpo = nome_prog.replace("Programa de Pós-Graduação em ", "").replace("Programa de Pós-Graduação ", "").replace("PPG em ", "").strip().upper()
    nome_norm_busca = ''.join(c for c in unicodedata.normalize('NFD', nome_limpo) if unicodedata.category(c) != 'Mn')
    
    # 1. Busca Exata
    if nome_norm_busca in catalogo_capes:
        return catalogo_capes[nome_norm_busca]
        
    # 2. Busca Fuzzy (Aproximada)
    chaves = list(catalogo_capes.keys())
    matches = difflib.get_close_matches(nome_norm_busca, chaves, n=1, cutoff=0.65)
    if matches:
        return catalogo_capes[matches[0]]
        
    # 3. Interseção Semântica
    palavras_busca = set([p for p in nome_norm_busca.split() if len(p) > 2])
    for key_capes, grande_area in catalogo_capes.items():
        palavras_capes = set([p for p in key_capes.split() if len(p) > 2])
        if len(palavras_busca.intersection(palavras_capes)) >= max(1, len(palavras_busca) - 1):
            return grande_area
            
    # Se o programa for novo demais e não estiver na CAPES ainda
    return "Multidisciplinar / Transversal"

# --- FUNÇÕES DE LIMPEZA E FORMATAÇÃO ---
def carregar_catalogo_tcc():
    """Lê o arquivo local de mapeamento das coleções de TCC da UFSC."""
    try:
        with open('mapa_colecoes_tcc.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ ERRO: Arquivo 'mapa_colecoes_tcc.json' não encontrado.")
        return []

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

# --- MOTOR DE EXTRAÇÃO OAI-PMH ---
def realizar_extracao(set_spec, nome_prog=""):
    sickle = Sickle('https://repositorio.ufsc.br/oai/request', timeout=120)
    try: 
        records = sickle.ListRecords(metadataPrefix='oai_dc', set=set_spec)
    except Exception as e: 
        print(f"  [!] Erro ao acessar a coleção OAI {set_spec}: {e}")
        return []
        
    dados_extraidos, titulos_vistos, iterator, i = [], set(), iter(records), 0
    
    while True:
        try: record = next(iterator)
        except StopIteration: break
        except Exception: break
            
        try:
            i += 1
            if i % 200 == 0: print(f"  ...lendo documentos, já processados: {i}")
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
                'palavras_chave': pks, 
                'ano': ano_real, 
                'resumo': resumo, 
                'programa_origem': nome_prog,
                'url': url_doc 
            })
        except Exception: continue
        
    return dados_extraidos

def aplicar_macrotemas(dados, api_key):
    import re
    import time
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    sujeira_academica = list(set([ # Uso de set() para remover qualquer duplicata acidental
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com", "não", "uma", "os", "no", "se", "na", 
        "por", "mais", "as", "dos", "como", "mas", "foi", "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser",
        "neste", "esta", "está", "este", "pelo", "pela", "seus", "suas", "nas", "aos", "meu", "sua", "através",
        "the", "of", "and", "in", "to", "for", "with", "on", "at", "by", "from", "an", "is", "it", "this", "that",
        "study", "analysis", "based", "using", "results", "work", "research", "paper", "thesis", "dissertation",
        "analise", "estudo", "desenvolvimento", "proposta", "metodo", "processo", "sistema", "modelo", "projeto",
        "utilização", "uso", "efeito", "avaliação", "verificação", "experimental", "numérica", "aplicação",
        "sobre", "entre", "quando", "onde", "qual", "quais", "abstract", "resumo", "palavras", "chave", "keywords", 
        "introduction", "conclusion", "methodology", "are", "were", "was", "have", "has", "been", "can", "could", "would"
    ]))

    # OTIMIZAÇÃO 2: Pré-compilação do Regex fora do loop
    padrao_limpeza = re.compile(r'[^a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ\s]')
    textos = []
    
    for doc in dados:
        bruto = f"{(doc.get('titulo', '') + ' ') * 3} {' '.join(doc.get('palavras_chave', []))} {doc.get('resumo', '')}"
        limpo = padrao_limpeza.sub(' ', bruto).lower()
        textos.append(limpo)

    # OTIMIZAÇÃO 1: ngram_range=(1, 2) para capturar bigramas (ex: "machine learning")
    # Aumentei max_features para 1500 para dar espaço aos novos termos compostos
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words=sujeira_academica, max_features=1500, ngram_range=(1, 2))
    
    try:
        tfidf_matrix = vectorizer.fit_transform(textos)
    except Exception as e:
        print(f" [!] Erro na vetorização matemática: {e}")
        return dados

    print("   [+] Calculando a quantidade ótima de clusters matemáticos...")
    min_k = 4
    max_k = min(20, max(5, tfidf_matrix.shape[0] // 10)) 
    
    melhor_k = min_k
    melhor_score = -1
    
    if tfidf_matrix.shape[0] > min_k:
        for k in range(min_k, max_k + 1):
            # OTIMIZAÇÃO 3: n_init='auto' deixa o algoritmo mais rápido nas versões recentes do scikit-learn
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(tfidf_matrix)
            score = silhouette_score(tfidf_matrix, labels)
            
            if score > melhor_score:
                melhor_score = score
                melhor_k = k
    else:
        melhor_k = max(2, tfidf_matrix.shape[0] // 2)
        
    print(f"   [+] Quantidade ótima definida: {melhor_k} macrotemas (Score de Silhueta: {melhor_score:.3f})")
    num_topicos = melhor_k

    try:
        nmf_model = NMF(n_components=num_topicos, random_state=42, init='nndsvd', max_iter=500)
        nmf_matrix = nmf_model.fit_transform(tfidf_matrix)
        feature_names = vectorizer.get_feature_names_out()
    except Exception as e:
        print(f" [!] Erro no NMF: {e}")
        return dados

    clusters = []
    for idx, topic in enumerate(nmf_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-8:-1]]
        clusters.append(f"Grupo {idx+1}: {', '.join(top_words)}")
    
    contexto = "\n".join(clusters)

    prompt_humanizado = f"""Você é um especialista em epistemologia e taxonomia acadêmica.
Abaixo estão {num_topicos} grupos de palavras-chave extraídas de agrupamentos matemáticos (NMF) de trabalhos acadêmicos.
Sua missão é batizar cada grupo com um nome definitivo e que represente com precisão essa subárea do conhecimento.

Diretrizes rigorosas:
- Crie títulos com no máximo 4 palavras.
- NÃO use palavras genéricas como 'Estudo', 'Análise', 'Área de', 'Aplicações', 'Correlatas'.
- Vá direto ao ponto. Exemplo: Se ler 'soldagem, laser, liga, tensão', responda 'Tecnologias de Soldagem'.
- Retorne APENAS uma lista estritamente numerada de 1 a {num_topicos}, sem introduções ou conclusões.

GRUPOS DE PALAVRAS:
{contexto}"""

    max_tentativas = 3
    nomes_finais = []
    
    for tentativa in range(max_tentativas):
        try:
            from gemini_utils import generate_content, response_text, DEFAULT_TEXT_MODELS # Ajuste o import conforme sua estrutura
            
            response = generate_content(
                prompt=prompt_humanizado,
                api_key=api_key,
                model_candidates=DEFAULT_TEXT_MODELS,
                temperature=0.4,
                candidate_count=1,
            )
            respostas = response_text(response).strip().split('\n')
            nomes_finais = [re.sub(r'^\d+[\.\s\-]+', '', r).strip().replace('*', '') for r in respostas if len(r) > 3]
            
            if len(nomes_finais) < num_topicos:
                raise ValueError(f"Gemini retornou apenas {len(nomes_finais)} nomes. Esperados: {num_topicos}")
                
            break 
            
        except Exception as e:
            erro_str = str(e)
            if "429" in erro_str or "Quota" in erro_str or "exhausted" in erro_str.lower():
                tempo_espera = 60
                print(f"   [⏳] Limite da API atingido. Pausando {tempo_espera}s... (Tentativa {tentativa + 1}/{max_tentativas})")
                time.sleep(tempo_espera)
            else:
                print(f"   [!] Erro inesperado na API Gemini ({e}).")
                break 

    if not nomes_finais or len(nomes_finais) < num_topicos:
        print("   [!] Aplicando Fallback de nomes (extração matemática direta)...")
        nomes_finais = []
        for c in clusters:
            palavras = c.split(': ')[1].split(', ')
            # Fallback seguro contra clusters com menos de 2 palavras
            nome_fallback = f"{palavras[0].title()} e {palavras[1].title()}" if len(palavras) > 1 else palavras[0].title()
            nomes_finais.append(nome_fallback)

    # Aplica os nomes aos documentos E CALCULA A PUREZA
    for i, doc in enumerate(dados):
        top_idx = nmf_matrix[i].argmax()
        pesos_doc = nmf_matrix[i]
        
        doc['macrotema'] = nomes_finais[top_idx] if top_idx < len(nomes_finais) else "Interseções Multidisciplinares"
        
        # 1. CÁLCULO DA PUREZA TEMÁTICA (0 a 100%)
        soma_pesos = pesos_doc.sum()
        pureza = (pesos_doc[top_idx] / soma_pesos) * 100 if soma_pesos > 0 else 0.0
        doc['pureza_nmf'] = round(pureza, 2)
        
        if pureza >= 85.0: doc['perfil_pesquisa'] = "Altamente Especializado"
        elif pureza >= 60.0: doc['perfil_pesquisa'] = "Especialista"
        elif pureza >= 40.0: doc['perfil_pesquisa'] = "Híbrido / Fronteira"
        else: doc['perfil_pesquisa'] = "Generalista / Transversal"

    return dados

# --- ORQUESTRAÇÃO DO PIPELINE ---
def executar_pipeline_diario():
    print("==================================================")
    print("🚀 INICIANDO PIPELINE DE ECOSSISTEMAS - UFSC")
    print("==================================================")
    
    api_key = get_gemini_api_key()
    if not api_key:
        print("❌ ERRO: GEMINI_API_KEY não encontrada em variáveis de ambiente ou secrets.")
        return

    try:
        with open('programas_ufsc.json', 'r', encoding='utf-8') as f:
            catalogo_programas = json.load(f)
    except FileNotFoundError:
        print("❌ ERRO: Arquivo 'programas_ufsc.json' não encontrado.")
        return

    # 1. Extração e Agrupamento por Afinidade (VIA CAPES PARA PPG)
    print("\n--- INICIANDO COLETA OAI-PMH (PPGs) ---")
    
    # NOVO: Baixa o catálogo oficial antes de começar a extrair os teses
    catalogo_capes_oficial = carregar_catalogo_capes_ufsc()
    
    documentos_por_ecossistema = defaultdict(list)
    
    for prog, set_spec in catalogo_programas.items():
        print(f"-> Minerando: {prog}")
        dados_prog = realizar_extracao(set_spec, nome_prog=prog)
        
        if dados_prog:
            # NOVO: Classifica o ecossistema usando a base do governo
            ecossistema = obter_ecossistema_capes(prog, catalogo_capes_oficial)
            
            # Padroniza a capitalização do nome da área (ex: "ENGENHARIAS" vira "Engenharias")
            ecossistema = ecossistema.title() 
            
            for doc in dados_prog:
                doc['ecossistema_afinidade'] = ecossistema
            
            documentos_por_ecossistema[ecossistema].extend(dados_prog)
            print(f"   [+] {len(dados_prog)} docs alocados na Área: {ecossistema}")

    if not documentos_por_ecossistema:
        print("\n❌ ERRO FATAL: Nenhum documento extraído.")
        return

    # 2. Computação Semântica por Ecossistema
    print("\n--- INICIANDO COMPUTAÇÃO SEMÂNTICA POR ECOSSISTEMA ---")
    dados_finais_consolidados = []
    
    for ecossistema, docs_eco in documentos_por_ecossistema.items():
        print(f"\n🧠 Analisando Ecossistema: {ecossistema} ({len(docs_eco)} documentos)")
        
        docs_tematizados = aplicar_macrotemas(docs_eco, api_key=api_key)
        dados_finais_consolidados.extend(docs_tematizados)
        
        # Respiro entre as áreas para esfriar a API
        time.sleep(5)

    # 3. Salvamento dos Dados de PPGs
    nome_arquivo_saida = 'base_consolidada_ufsc.json.gz'
    print(f"\n--- COMPACTANDO E SALVANDO DADOS PPG ({nome_arquivo_saida}) ---")
    try:
        with gzip.open(nome_arquivo_saida, 'wt', encoding='utf-8') as f:
            json.dump(dados_finais_consolidados, f, ensure_ascii=False)
        print("✅ Arquivo GZIP gerado com sucesso!")
    except Exception as e:
        print(f"❌ ERRO ao salvar o arquivo: {e}")

    # =========================================================================
    # FLUXO DA GRADUAÇÃO (TCCs) - EXTRAÇÃO
    # =========================================================================
    catalogo_tcc = carregar_catalogo_tcc()
    if catalogo_tcc:
        print("\n--- INICIANDO COLETA OAI-PMH (TCCs / Graduação) ---")
        documentos_tcc_por_ecossistema = defaultdict(list)
        
        for tcc in catalogo_tcc:
            curso = tcc.get('curso')
            set_spec = tcc.get('setSpec')
            if not curso or not set_spec: continue
                
            print(f"-> Minerando Graduação: {curso}")
            dados_tcc = realizar_extracao(set_spec, nome_prog=curso)
            
            if dados_tcc:
            
                nome_curso_limpo = curso.replace("TCC", "").strip()
                ecossistema_tcc = f"Graduação - {nome_curso_limpo}"
                
                for doc in dados_tcc:
                    doc['ecossistema_afinidade'] = ecossistema_tcc
                
                documentos_tcc_por_ecossistema[ecossistema_tcc].extend(dados_tcc)
                print(f"   [+] {len(dados_tcc)} TCCs alocados na Área: {ecossistema_tcc}.")
                
        if documentos_tcc_por_ecossistema:
            print("\n--- INICIANDO COMPUTAÇÃO SEMÂNTICA (TCCs) ---")
            dados_tcc_consolidados = []
            
            for ecos, docs in documentos_tcc_por_ecossistema.items():
                print(f"\n🧠 Analisando TCCs: {ecos} ({len(docs)} documentos)")
                docs_tematizados_tcc = aplicar_macrotemas(docs, api_key=api_key)
                dados_tcc_consolidados.extend(docs_tematizados_tcc)
                time.sleep(5)
                
            # Salvamento dos Dados de TCCs
            nome_arquivo_saida_tcc = 'base_tcc_ufsc.json.gz'
            print(f"\n--- COMPACTANDO E SALVANDO DADOS TCC ({nome_arquivo_saida_tcc}) ---")
            try:
                with gzip.open(nome_arquivo_saida_tcc, 'wt', encoding='utf-8') as f:
                    json.dump(dados_tcc_consolidados, f, ensure_ascii=False)
                print("✅ Arquivo GZIP de TCCs gerado com sucesso!")
            except Exception as e:
                print(f"❌ ERRO ao salvar o arquivo de TCCs: {e}")
    else:
        print("\n⚠️ Nenhum catálogo de TCCs encontrado ou válido. Pulando extração de graduação.")
    # =========================================================================

    # 4. Envio para o Banco de Grafos (apenas PPGs)
    neo4j_uri, neo4j_user, neo4j_password = get_neo4j_credentials()
    if neo4j_uri and neo4j_user and neo4j_password:
        popular_banco_grafos(
            dados_finais_consolidados,
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
        )
    else:
        print("⚠️ Neo4j não configurado. Pulando a etapa de injeção no banco.")

    print("\n==================================================")
    print("🎯 PIPELINE FINALIZADO COM SUCESSO!")
    print("==================================================")

if __name__ == "__main__":
    executar_pipeline_diario()
