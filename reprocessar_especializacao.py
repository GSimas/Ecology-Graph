import gzip
import json
import time
from collections import defaultdict
from app_config import get_gemini_api_key
from pipeline_ufsc import aplicar_macrotemas

def reprocessar_bases():
    api_key = get_gemini_api_key()
    arquivos = ['base_consolidada_ufsc.json.gz', 'base_tcc_ufsc.json.gz']
    
    for arquivo in arquivos:
        print(f"\n📂 Abrindo {arquivo}...")
        try:
            with gzip.open(arquivo, 'rt', encoding='utf-8') as f:
                dados = json.load(f)
                
            docs_por_eco = defaultdict(list)
            for doc in dados:
                eco = doc.get('ecossistema_afinidade', 'Desconhecido')
                docs_por_eco[eco].append(doc)
                
            dados_atualizados = []
            for eco, docs in docs_por_eco.items():
                print(f"🧠 Recalculando NMF para: {eco}")
                docs_tematizados = aplicar_macrotemas(docs, api_key=api_key)
                dados_atualizados.extend(docs_tematizados)
                time.sleep(3) # Respiro pra API do Gemini
                
            with gzip.open(arquivo, 'wt', encoding='utf-8') as f:
                json.dump(dados_atualizados, f, ensure_ascii=False)
            print(f"✅ {arquivo} salvo com a métrica Pureza NMF!")
            
        except FileNotFoundError:
            print(f"⚠️ Arquivo {arquivo} não encontrado, pulando...")

if __name__ == "__main__":
    reprocessar_bases()