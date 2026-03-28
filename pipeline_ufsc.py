import os
import json
import re
import unicodedata
from sickle import Sickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import google.generativeai as genai
import gzip

# --- FUNÇÕES DE LIMPEZA E FORMATAÇÃO ---
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

# --- MOTOR DE INTELIGÊNCIA SEMÂNTICA (NMF + GEMINI) ---
def aplicar_macrotemas(dados, api_key, num_topicos=12):
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('gemini-2.5-flash') 
    except Exception:
        model = genai.GenerativeModel('gemini-2.0-flash')

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
        bruto = f"{(doc.get('titulo', '') + ' ') * 3} {' '.join(doc.get('palavras_chave', []))} {doc.get('resumo', '')}"
        limpo = re.sub(r'[^a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ\s]', ' ', bruto).lower()
        textos.append(limpo)

    vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words=sujeira_academica, max_features=800)
    
    try:
        tfidf_matrix = vectorizer.fit_transform(textos)
        nmf_model = NMF(n_components=num_topicos, random_state=42, init='nndsvd')
        nmf_matrix = nmf_model.fit_transform(tfidf_matrix)
        feature_names = vectorizer.get_feature_names_out()
    except Exception as e:
        print(f" [!] Erro na vetorização matemática: {e}")
        return dados

    clusters = []
    for idx, topic in enumerate(nmf_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-8:-1]]
        clusters.append(f"Grupo {idx+1}: {', '.join(top_words)}")
    
    contexto = "\n".join(clusters)

    prompt_humanizado = f"""Você é um especialista em epistemologia e taxonomia acadêmica.
Abaixo estão {num_topicos} grupos de palavras-chave extraídas de agrupamentos matemáticos (NMF) de teses e dissertações.
Sua missão é batizar cada grupo com um nome definitivo e que represente com precisão essa subárea do conhecimento.

Diretrizes rigorosas:
- Crie títulos com no máximo 4 palavras.
- NÃO use palavras genéricas como 'Estudo', 'Análise', 'Área de', 'Aplicações', 'Correlatas'.
- Vá direto ao ponto. Exemplo: Se ler 'soldagem, laser, liga, tensão', responda 'Tecnologias de Soldagem'.
- Retorne APENAS uma lista estritamente numerada de 1 a {num_topicos}, sem introduções ou conclusões.

GRUPOS DE PALAVRAS:
{contexto}"""

    try:
        response = model.generate_content(
            prompt_humanizado,
            generation_config=genai.types.GenerationConfig(candidate_count=1, temperature=0.4)
        )
        respostas = response.text.strip().split('\n')
        nomes_finais = [re.sub(r'^\d+[\.\s\-]+', '', r).strip().replace('*', '') for r in respostas if len(r) > 3]
        
        if len(nomes_finais) < num_topicos:
            raise ValueError(f"Gemini retornou {len(nomes_finais)} nomes. Esperados: {num_topicos}")

    except Exception as e:
        print(f" [!] Erro na API Gemini ({e}). Utilizando Fallback direto.")
        nomes_finais = []
        for c in clusters:
            palavras = c.split(': ')[1].split(', ')
            nomes_finais.append(f"{palavras[0].title()} e {palavras[1].title()}")

    for i, doc in enumerate(dados):
        top_idx = nmf_matrix[i].argmax()
        doc['macrotema'] = nomes_finais[top_idx] if top_idx < len(nomes_finais) else "Interseções Multidisciplinares"

    return dados

# --- ORQUESTRAÇÃO DO PIPELINE ---
def executar_pipeline_diario():
    print("==================================================")
    print("🚀 INICIANDO PIPELINE DE EXTRAÇÃO - UFSC")
    print("==================================================")
    
    # 1. Checa a chave da API no ambiente
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERRO: A variável de ambiente 'GEMINI_API_KEY' não foi encontrada.")
        print("Certifique-se de exportar a chave antes de rodar o script.")
        return

    # 2. Carrega o Catálogo
    try:
        with open('programas_ufsc.json', 'r', encoding='utf-8') as f:
            catalogo_programas = json.load(f)
        print(f"✅ Catálogo carregado: {len(catalogo_programas)} programas encontrados.")
    except FileNotFoundError:
        print("❌ ERRO: Arquivo 'programas_ufsc.json' não encontrado na pasta atual.")
        return

    # 3. Fase de Extração
    dados_totais = []
    print("\n--- INICIANDO COLETA OAI-PMH ---")
    for prog, set_spec in catalogo_programas.items():
        print(f"-> Minerando: {prog} ({set_spec})")
        dados_prog = realizar_extracao(set_spec, nome_prog=prog)
        dados_totais.extend(dados_prog)
        print(f"   [+] Coletados {len(dados_prog)} documentos.")
        
    if not dados_totais:
        print("\n❌ ERRO FATAL: Nenhum documento foi extraído. Abortando.")
        return
        
    print(f"\n✅ COLETA CONCLUÍDA: {len(dados_totais)} documentos totais minerados.")

    # 4. Fase de Inteligência (Macrotemas)
    print("\n--- INICIANDO COMPUTAÇÃO SEMÂNTICA (NMF + GEMINI) ---")
    dados_finais = aplicar_macrotemas(dados_totais, api_key=api_key, num_topicos=15)
    print("✅ Macrotemas computados e atribuídos com sucesso.")

    # 5. Salvamento dos Dados (agora Compactado)
    nome_arquivo_saida = 'base_consolidada_ufsc.json.gz'
    print(f"\n--- COMPACTANDO E SALVANDO DADOS ({nome_arquivo_saida}) ---")
    try:
        # Usamos gzip.open com 'wt' (Write Text)
        with gzip.open(nome_arquivo_saida, 'wt', encoding='utf-8') as f:
            json.dump(dados_finais, f, ensure_ascii=False)
        print("✅ Arquivo GZIP gerado com sucesso. O tamanho caiu drasticamente!")
    except Exception as e:
        print(f"❌ ERRO ao salvar o arquivo: {e}")

    print("\n==================================================")
    print("🎯 PIPELINE FINALIZADO COM SUCESSO!")
    print("==================================================")

if __name__ == "__main__":
    executar_pipeline_diario()