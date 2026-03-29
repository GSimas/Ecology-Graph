import re

with open('pages/1_Exploracao_Global.py', 'r', encoding='utf-8') as f:
    linhas = f.readlines()

inicio_idx = -1
for i, linha in enumerate(linhas):
    if "# --- SEÇÃO: MÉTRICAS DE ECOLOGIA PROFUNDA ---" in linha:
        inicio_idx = i
        break

if inicio_idx == -1:
    print("Seção não encontrada")
    exit(1)

topo = linhas[:inicio_idx]
metade_inferior = linhas[inicio_idx:]

blocos = []
bloco_atual = []
# 0 = Ecologia, 1 = Topologia, 2 = Estrutural, 3 = Historica, 4 = Lexicom, 5 = Co-ocorr, 6 = Exportacao
identificadores_blocos = [
    "# === SEÇÃO 1: GRAFO INTERATIVO GERAL ===",
    "# === SEÇÃO 2: ANÁLISE ESTRUTURAL (RANKING SNA) ===",
    "# === SEÇÃO 3: EVOLUÇÃO CRONOLÓGICA ===",
    "# === SEÇÃO 4: NUVEM DE PALAVRAS ===",
    "# === SEÇÃO 5: GRAFO DE CO-OCORRÊNCIA ===",
    "# === SEÇÃO 6: EXPORTAÇÃO DA BASE ==="
]

current_block_idx = 0
for linha in metade_inferior:
    if current_block_idx < len(identificadores_blocos) and identificadores_blocos[current_block_idx] in linha:
        blocos.append(bloco_atual)
        bloco_atual = []
        current_block_idx += 1
    
    # Check for extra 'st.markdown("---")' lines which were used as visual separators before and remove them
    if 'st.markdown("---")' in linha and not ('Glossário' in ''.join(bloco_atual[-1:]) or '#' in linha):
        # Allow if it's not simply a top level divider inside a tab
        pass
    bloco_atual.append(linha)
blocos.append(bloco_atual)

# Add tabs header to the top
novo_codigo = topo.copy()

tabs_header = """
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🧬 Métricas de Ecologia Profunda (SNA Avançado)",
    "🕸️ 1. Topologia e Grafo Interativo",
    "🏆 2. Análise Estrutural e Rankings (SNA)",
    "📈 3. Evolução Histórica (Temporal)",
    "☁️ 4. Lexicometria e Nuvem de Palavras",
    "🔗 5. Grafo de Co-ocorrência de Palavras",
    "📥 6. Exportação da Base de Dados Bruta"
])
"""
novo_codigo.append(tabs_header)

suffix = ""
for idx, bloco in enumerate(blocos):
    novo_codigo.append(f"\nwith tab{idx+1}:\n")
    for linha in bloco:
        if linha.strip() == 'st.markdown("---")' and "Glossário" not in ''.join(bloco):
            continue # Remover st.markdown("---") que eram divisores de seções antigas
            
        # Para evitar chaves de sessões repetidas
        if "key=\"niv_" in linha or "key=\"ano_" in linha:
            linha = linha.replace('key="', f'key="t{idx+1}_')
            
        novo_codigo.append("    " + linha if linha.strip() else "\n")

with open('pages/1_Exploracao_Global.py', 'w', encoding='utf-8') as f:
    f.writelines(novo_codigo)
print("Transformado com Sucesso!")
