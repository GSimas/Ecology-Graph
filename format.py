import codecs
import textwrap

with codecs.open('pages/1_Exploracao_Global.py', 'r', 'utf-8') as f:
    lines = f.readlines()

# We know the specific section headers
headers = [
    "# --- SEÇÃO: MÉTRICAS DE ECOLOGIA PROFUNDA ---",
    "# === SEÇÃO 1: GRAFO INTERATIVO GERAL ===",
    "# === SEÇÃO 2: ANÁLISE ESTRUTURAL (RANKING SNA) ===",
    "# === SEÇÃO 3: EVOLUÇÃO CRONOLÓGICA ===",
    "# === SEÇÃO 4: NUVEM DE PALAVRAS ===",
    "# === SEÇÃO 5: GRAFO DE CO-OCORRÊNCIA ===",
    "# === SEÇÃO 6: EXPORTAÇÃO DA BASE ==="
]

indices = []
for h in headers:
    idx = next(i for i, line in enumerate(lines) if h in line)
    indices.append(idx)
indices.append(len(lines))

blocks = []
for i in range(len(headers)):
    block_lines = lines[indices[i]:indices[i+1]]
    
    # Filter out horizontal rules between sections to keep UI clean inside tabs
    filtered = []
    for line in block_lines:
        if line.strip() == 'st.markdown("---")':
            # keep if it's the one above the glossary in tab 1
            if "Glossário" not in "".join(block_lines):
                continue
        filtered.append(line)
    
    indented = textwrap.indent("".join(filtered), "    ")
    blocks.append(f"with tab{i+1}:\n" + indented + "\n")

header_code = """
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

final_code = "".join(lines[:indices[0]]) + header_code + "".join(blocks)

with codecs.open('pages/1_Exploracao_Global.py', 'w', 'utf-8') as f:
    f.write(final_code)
