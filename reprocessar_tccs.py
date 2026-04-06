import gzip
import json
import time
from collections import defaultdict

# Importamos as configurações e funções do seu pipeline existente
from app_config import get_gemini_api_key
from pipeline_ufsc import aplicar_macrotemas

def reprocessar_macrotemas_tcc():
    print("==================================================")
    print("🚀 INICIANDO REPROCESSAMENTO SEMÂNTICO (TCCs)")
    print("==================================================")
    
    api_key = get_gemini_api_key()
    if not api_key:
        print("❌ ERRO: GEMINI_API_KEY não encontrada.")
        return

    # 1. Carregar os dados que já foram extraídos do OAI-PMH
    print("📂 Carregando base_tcc_ufsc.json.gz...")
    try:
        with gzip.open('base_tcc_ufsc.json.gz', 'rt', encoding='utf-8') as f:
            dados_tcc = json.load(f)
        print(f"✅ Sucesso! {len(dados_tcc)} TCCs carregados em memória.")
    except FileNotFoundError:
        print("❌ ERRO: Arquivo base_tcc_ufsc.json.gz não encontrado.")
        return

    # 2. Re-agrupar os documentos pelo curso correto (Isolamento de Ecossistema)
    docs_por_curso = defaultdict(list)
    for doc in dados_tcc:
        # Pega o programa de origem (ex: "TCC Engenharia Elétrica")
        curso = doc.get('programa_origem', 'Desconhecido')
        
        # Limpa o nome para virar o ecossistema
        nome_curso_limpo = curso.replace("TCC", "").strip()
        ecossistema = f"Graduação - {nome_curso_limpo}"
        
        # Atualiza a tag no documento
        doc['ecossistema_afinidade'] = ecossistema 
        
        # Coloca no balde correto
        docs_por_curso[ecossistema].append(doc)

    # 3. Aplicar macrotemas de forma isolada (Curso a Curso)
    dados_reprocessados = []
    print("\n--- RE-CALCULANDO MACROTEMAS ---")
    
    for ecos, docs in docs_por_curso.items():
        print(f"\n🧠 Analisando: {ecos} ({len(docs)} documentos)")
        
        # Chama a sua função NMF + Gemini
        docs_tematizados = aplicar_macrotemas(docs, api_key=api_key)
        dados_reprocessados.extend(docs_tematizados)
        
        # Pausa para evitar Rate Limit 429 da API do Gemini
        time.sleep(5) 

    # 4. Salvar por cima do arquivo original
    print("\n--- COMPACTANDO E SALVANDO DADOS ---")
    try:
        with gzip.open('base_tcc_ufsc.json.gz', 'wt', encoding='utf-8') as f:
            json.dump(dados_reprocessados, f, ensure_ascii=False)
        print("🎯 SUCESSO ABSOLUTO! Arquivo atualizado.")
    except Exception as e:
        print(f"❌ ERRO ao salvar o arquivo: {e}")

if __name__ == "__main__":
    reprocessar_macrotemas_tcc()