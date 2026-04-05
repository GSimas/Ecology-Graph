import gzip
import json
from neo4j import GraphDatabase
import math

from app_config import get_neo4j_credentials

# ==========================================
# CONFIGURAÇÕES SEGURAS (Lendo o cofre)
# ==========================================
print("🔐 Carregando credenciais do Neo4j (env ou Streamlit secrets)...")
try:
    URI, USER, PASSWORD = get_neo4j_credentials(required=True)
except Exception as e:
    print(f"❌ ERRO ao carregar senhas: {e}")
    exit()

# ==========================================
# QUERIES CYPHER
# ==========================================
query_estrutural = """
UNWIND $batch AS doc

MERGE (eco:Ecossistema {nome: doc.ecossistema_afinidade})
MERGE (p:Programa {nome: doc.programa_origem})
MERGE (p)-[:PERTENCE_A]->(eco)

MERGE (d:Documento {titulo: doc.titulo})
SET d.ano = doc.ano, 
    d.nivel = doc.nivel_academico, 
    d.macrotema = doc.macrotema,
    d.resumo = doc.resumo
    
MERGE (d)-[:VINCULADO_A]->(p)

FOREACH (autor_nome IN doc.autores |
    MERGE (a:Autor {nome: autor_nome})
    MERGE (a)-[:ESCREVEU]->(d)
)

FOREACH (pk IN doc.palavras_chave |
    MERGE (c:Conceito {nome: pk})
    MERGE (d)-[:ABORDA]->(c)
)
"""

query_orientador = """
UNWIND $batch AS doc
WITH doc WHERE doc.orientador IS NOT NULL AND doc.orientador <> ''
MATCH (d:Documento {titulo: doc.titulo})
MERGE (o:Orientador {nome: doc.orientador})
MERGE (o)-[:ORIENTOU]->(d)
"""

# Função auxiliar para fatiar a lista gigante
def fatiar_lista(lista, tamanho_lote):
    for i in range(0, len(lista), tamanho_lote):
        yield lista[i:i + tamanho_lote]

def carregar_e_injetar():
    arquivo_gz = 'base_consolidada_ufsc.json.gz'
    
    print(f"\n📂 Abrindo arquivo local: {arquivo_gz}...")
    try:
        with gzip.open(arquivo_gz, 'rt', encoding='utf-8') as f:
            dados = json.load(f)
        total_docs = len(dados)
        print(f"✅ Arquivo lido! Total de documentos: {total_docs}")
    except Exception as e:
        print(f"❌ ERRO ao ler o arquivo: {e}")
        return

    print("\n🌐 Conectando ao Neo4j AuraDB...")
    try:
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        driver.verify_connectivity()
    except Exception as e:
        print(f"❌ ERRO FATAL DE CONEXÃO: {e}")
        return

    # Definimos lotes de 500 documentos para não estourar a memória do AuraDB Free
    TAMANHO_LOTE = 500
    total_lotes = math.ceil(total_docs / TAMANHO_LOTE)

    print(f"\n🚀 Iniciando injeção fragmentada ({total_lotes} lotes de {TAMANHO_LOTE} docs)...")
    try:
        with driver.session() as session:
            print("   -> Criando índices estruturais (Crucial para a velocidade do MERGE)...")
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:Documento) ON (d.titulo)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (c:Conceito) ON (c.nome)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (o:Orientador) ON (o.nome)")
            
            # Passo 1: Injeção Estrutural
            print("\n⏳ [ETAPA 1/2] Injetando Estrutura Base (Programas, Documentos, Autores, Conceitos)...")
            lote_atual = 1
            for lote in fatiar_lista(dados, TAMANHO_LOTE):
                session.run(query_estrutural, batch=lote)
                print(f"      Lote estrutural {lote_atual}/{total_lotes} concluído.")
                lote_atual += 1
                
            # Passo 2: Injeção de Orientadores
            print("\n⏳ [ETAPA 2/2] Conectando Orientadores aos Documentos...")
            lote_atual = 1
            for lote in fatiar_lista(dados, TAMANHO_LOTE):
                session.run(query_orientador, batch=lote)
                print(f"      Lote orientadores {lote_atual}/{total_lotes} concluído.")
                lote_atual += 1
            
            print("\n🎯 SUCESSO ABSOLUTO! O Ecossistema está mapeado no Neo4j.")
    except Exception as e:
        print(f"\n❌ Erro durante a injeção do lote: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    carregar_e_injetar()
