import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import networkx.algorithms.community as nx_comm
from pyvis.network import Network
import json
import re
from collections import Counter
import itertools  # CORRIGIDO: importação duplicada removida
import streamlit.components.v1 as components
from streamlit_agraph import agraph, Node, Edge, Config
import numpy as np
import scipy as sp
import io
import datetime
import time

from backend import (
    classificar_por_percentil,
    otimizar_parametros_foresight,
    validar_foresight_historico,
    preparar_radar_foresight,
    plotar_mapa_tematico,
    plotar_grafico_3d_sna,
    gerar_grafo_ecologia_memes_agraph,
    calcular_sna_global,
    calcular_maturidade_rede,
    renderizar_nuvem_interativa_html_exploracao,
    gerar_nodos_coocorrencia_agraph,
    preparar_csv_exportacao,
    calcular_metricas_complexas,
    preparar_exportacao_grafo,
    obter_frequencias_texto,
    gerar_nodos_globais_agraph,
    obter_dataframe_metricas,
    preparar_dados_base_df,
    renderizar_nuvem_interativa_html,
    preparar_dataframe,
    calcular_burt,
    calcular_metricas_memeticas,
    preparar_sankey_temporal,
    processar_lote_ontologia,
    gerar_base_boxplot_ql
)

# --- INICIALIZAÇÃO DEFENSIVA DE ESTADO ---
# CORRIGIDO: sna_global, df_grid e tipo_foresight_grid adicionados ao bloco
# defensivo para evitar recálculos custosos e perda de resultados no rerender.
chaves_necessarias = {
    'grafo_pronto': False,
    'tabela_pronta': False,
    'coocorrencia_pronta': False,
    'kpis_grafo': {'nos': 0, 'arestas': 0, 'legendas': []},
    'kpis_co': {'nos': 0, 'arestas': 0},
    'graf_glob_nodes': [],
    'graf_glob_edges': [],
    'co_nodes': [],
    'co_edges': [],
    'sna_global': None,           # NOVO: cache do cálculo global de SNA
    'df_grid': None,              # NOVO: persiste resultado do Grid Search
    'tipo_foresight_grid': None,  # NOVO: rastreia o tipo usado no último Grid Search
}

for chave, valor_padrao in chaves_necessarias.items():
    if chave not in st.session_state:
        st.session_state[chave] = valor_padrao

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Análises Avançadas & Globais",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)
with st.sidebar:
    st.image("ecograd - logo.png", width='stretch')
    st.markdown("---")
    st.markdown("### 🎓 Sobre o EcoGrad")
    st.markdown("O **EcoGrad** é uma plataforma analítica que mapeia e visualiza as redes de produção acadêmica da Pós-Graduação (Teses e Dissertações) e Graduação (TCCs).")
    st.markdown("### ⚙️ Como Funciona")
    st.markdown("Os documentos são processados usando algoritmos de redes complexas (Ciência de Redes) aliados a Inteligência Artificial. Isso desvela como pesquisadores, teorias e ferramentas se interconectam na academia.")
    st.markdown("### 🧭 Como Utilizar")
    st.markdown("1. **Selecione os cursos** e origens na aba principal.\n2. Navegue pelos **Dashboards** e visualize perfis no Motor de Busca.\n3. Explore os dados de um Autor, Orientador ou Conceito.\n4. Acesse **Análises Avançadas** para métricas detalhadas.")
    st.markdown("---")
    st.markdown("Desenvolvido por **Gustavo Simas**<br>[🔗 GitHub: GSimas](https://github.com/GSimas)", unsafe_allow_html=True)

# Estilização Adaptável (Light/Dark Mode)
st.markdown("""
    <style>
    /* Remove o fundo fixo para respeitar o tema do Streamlit */
    h1, h2, h3, h4 { color: #F39C12; font-family: 'Helvetica Neue', sans-serif; }
    
    /* Estilização das Métricas (Boxes) */
    [data-testid="stMetric"] {
        background-color: rgba(125, 125, 125, 0.08); /* Fundo sutil adaptável */
        border: 1px solid rgba(125, 125, 125, 0.2);
        padding: 15px;
        border-radius: 10px;
    }
    
    /* Garante que o rótulo (label) da métrica seja visível */
    [data-testid="stMetricLabel"] {
        color: var(--text-color);
        font-weight: bold;
    }

    button[kind="primary"] { 
        background-color: #2ECC71 !important; 
        color: white !important; 
        font-weight: bold !important; 
        border: none !important; 
    }
    </style>
""", unsafe_allow_html=True)

# --- PROTEÇÃO DE ESTADO ---
if 'dados_completos' not in st.session_state or not st.session_state['dados_completos']:
    st.warning("⚠️ Nenhuma base de dados carregada na memória.")
    st.info("Por favor, vá à página inicial e inicie a extração de um Programa de Pós-Graduação antes de usar as análises avançadas.")
    st.stop()

dados_completos = st.session_state['dados_completos']
dados_gerais = dados_completos  # Alias para compatibilidade com funções avançadas
nome_programa = st.session_state.get('nome_programa', 'N/A')
df_geral = preparar_dataframe(dados_gerais)

# --- PREPARAÇÃO DE VARIÁVEIS GLOBAIS PARA OS FILTROS DA EXPLORAÇÃO ---
niveis_disponiveis = sorted(list(set([d.get('nivel_academico', 'Não Classificado') for d in dados_completos])))
orientadores_disponiveis = sorted(list(set([d.get('orientador', 'Não informado') for d in dados_completos if d.get('orientador')])))
lista_todos_conceitos = []
for d in dados_completos:
    lista_todos_conceitos.extend(d.get('palavras_chave', []))
conceitos_unicos = sorted(list(set(lista_todos_conceitos)))
anos_disponiveis = [int(d.get('ano')) for d in dados_completos if d.get('ano') and str(d.get('ano')).isdigit()]
min_ano_global = min(anos_disponiveis) if anos_disponiveis else 2000
max_ano_global = max(anos_disponiveis) if anos_disponiveis else 2026


# =========================================================================
# INTERFACE PRINCIPAL
# =========================================================================

st.title(f"🧪 Ecologia do Conhecimento: Análises Avançadas")
st.subheader(f"Base de Dados: {nome_programa}")
st.markdown("---")

# Abas Principais (Incluindo a Exploração Global como a primeira)
tab_exploracao, tab_fluxos, tab_memes = st.tabs([
    "🔭 Exploração Global",
    "🌊 Fluxos",
    "🧬 Memética"
])

# =========================================================================
# ABA 1: EXPLORAÇÃO GLOBAL (Transformada em Sub-Módulos)
# =========================================================================
with tab_exploracao:
    st.markdown("### 🧬 Métricas de Redes Complexas Básicas")
    m_sna = calcular_metricas_complexas(dados_completos)

    if m_sna:
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Densidade da Rede", f"{m_sna['densidade']:.5f}")
        col_s2.metric("Eficiência Global", f"{m_sna['eficiencia']:.4f}")
        col_s3.metric("Entropia (H)", f"{m_sna['entropia']:.2f} bits")
        col_s4.metric("Clustering Médio", f"{m_sna['clustering']:.4f}")

        st.markdown("#### 📊 Estatísticas de Conectividade & 🧠 Influência e Estrutura (Médias)")
        with st.container():
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                st.markdown("**Conectividade (Links por Nó)**")
                c_l1, c_l2 = st.columns(2)
                c_l1.metric("Média de Links", f"{m_sna['links']['media']:.2f}")
                c_l2.metric("Desvio Padrão", f"{m_sna['links']['std']:.2f}")
                c_l3, c_l4 = st.columns(2)
                c_l3.metric("Mínimo", m_sna['links']['min'])
                c_l4.metric("Máximo", m_sna['links']['max'])
            with col_exp2:
                st.markdown("**Influência Estrutural**")
                c_i1, c_i2 = st.columns(2)
                c_i1.metric("PageRank Médio", f"{m_sna['pagerank_avg']:.6f}")
                c_i2.metric("Eigenvector Médio", f"{m_sna['eigen_avg']:.6f}")
                c_i3, c_i4 = st.columns(2)
                c_i3.metric("Restrição (Burt)", f"{m_sna['constraint_avg']:.4f}")
                c_i4.metric("Redundância", f"{m_sna['redundancia']:.4f}")

    with st.expander("📚 Glossário de Métricas"):
        st.markdown("""
        ### Topologia e Fluxo
        * **Densidade:** Proporção de conexões reais frente às possíveis. Indica quão "povoada" está a rede.
        * **Eficiência:** Mede quão fácil a informação viaja. Redes eficientes têm caminhos curtos entre quaisquer dois nós.
        * **Entropia da Rede ($H$):** Mede a diversidade e incerteza da distribuição de conexões. Uma entropia alta indica uma ecologia complexa e menos previsível.
        ### Centralidade e Poder
        * **PageRank:** Algoritmo do Google que mede a importância de um nó baseando-se na qualidade das suas conexões (não apenas quantidade).
        * **Eigenvector:** Mede a influência de um nó considerando que conexões com nós influentes valem mais.
        * **Restrição (Burt's Constraint):** Mede quanto um indivíduo está "preso" a um grupo. Baixa restrição indica um *Broker* (ponte entre diferentes saberes).
        ### Estrutura de Agrupamento
        * **Coeficiente de Agrupamento (Clustering):** Mede a probabilidade de dois vizinhos de um nó também estarem conectados entre si (formação de "bolhas").
        * **Redundância:** Indica o excesso de caminhos para a mesma informação. É o inverso da eficiência na otimização de fluxos.
        """)

    st.markdown("### 🧬 Métricas de Ecologia Profunda (SNA Avançado)")
    st.caption("Diagnóstico estrutural e físico da maturidade do ecossistema de conhecimento.")

    # CORRIGIDO: sna_global agora é cacheado em session_state para evitar
    # recálculo duplo quando o usuário navega para a aba Fluxos/Foresight.
    if st.session_state['sna_global'] is None:
        with st.spinner("Calculando leis de escala e correlações topológicas..."):
            st.session_state['sna_global'] = calcular_sna_global(dados_completos)

    sna_global = st.session_state['sna_global']
    maturidade = calcular_maturidade_rede(dados_completos, sna_global)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        gamma = maturidade['Gamma']
        if 2.0 <= gamma <= 3.0:
            status_g = "🟢 Ecossistema Saudável (Livre de Escala)"
        elif gamma < 2.0:
            status_g = "🔴 Alta Monopolização"
        else:
            status_g = "🟡 Rede Fragmentada (Aleatória)"
        st.metric("Lei de Potência (γ)", f"{gamma:.2f}", status_g, delta_color="off")

    with col_m2:
        spearman = maturidade['Spearman']
        if spearman > 0.95:
            status_s = "🔴 Hierarquia Rígida"
        elif spearman < 0.85:
            status_s = "🟢 Alto Fator de Inovação (Brokers)"
        else:
            status_s = "🟡 Equilíbrio Padrão"
        st.metric("Correlação de Spearman (ρ)", f"{spearman:.2f}", status_s, delta_color="off")

    with col_m3:
        assort = maturidade['Assortatividade']
        if assort > 0.1:
            status_a = "🟡 Endogâmica (Panelinhas)"
        elif assort < -0.1:
            status_a = "🟢 Expansiva / Interdisciplinar"
        else:
            status_a = "⚪ Estrutura Neutra"
        st.metric("Assortatividade (r)", f"{assort:.2f}", status_a, delta_color="off")

    with col_m4:
        rc = maturidade['Rich_Club']
        if rc > 0.3:
            status_rc = "🟢 Elite Coesa e Colaborativa"
        elif rc < 0.1:
            status_rc = "🔴 Hubs Isolados (Silos)"
        else:
            status_rc = "🟡 Colaboração Moderada"
        st.metric("Coeficiente Rich-Club (Φ)", f"{rc:.2%}", status_rc, delta_color="off")

    with st.expander("📖 Glossário de Maturidade: Como interpretar estes números?"):
        st.markdown("""
        Este painel mede a **Resiliência e o Fluxo de Conhecimento** baseados na física de Redes Complexas.

        **1. Lei de Potência (γ - Gamma): Mede o Equilíbrio de Liderança**
        A rede segue uma equação matemática $P(k) \\sim k^{-\\gamma}$.
        * **O que significa?** Avalia a dependência da rede em relação aos seus maiores pesquisadores/temas. 
        * **Diagnóstico:** O cenário "Ecossistema Saudável" (γ entre 2 e 3) indica que existem grandes líderes de pesquisa (hubs), mas o programa permite o surgimento de novos pesquisadores. Valores abaixo de 2 indicam uma rede monopolizada por pouquíssimos indivíduos.

        **2. Correlação de Spearman (ρ): Mede a Presença de Inovadores (Brokers)**
        Analisa a curva não-linear entre a quantidade de conexões (Grau) e o poder de ponte (Betweenness).
        * **O que significa?** Se a correlação for perto de **1.0**, as únicas "pontes" do programa são os grandes orientadores com centenas de alunos. Se a correlação for menor, o status é **Verde (Alto Fator de Inovação)**, pois significa que existem pesquisadores "iniciantes" ou temas de nicho atuando como pontes secretas (*Brokers*) unindo áreas que normalmente não conversariam.

        **3. Assortatividade (r): Mede a "Panelinha" vs "Interdisciplinaridade"**
        * **O que significa?** É a tendência dos nós se conectarem com nós parecidos. 
        * **Diagnóstico:** Valores muito positivos ($r > 0$) indicam **Endogenia/Panelinha**: os gigantes só trabalham com os gigantes, e os iniciantes só trabalham com os iniciantes. Valores negativos ($r < 0$) são excelentes para PPGs, pois caracterizam uma rede disassortativa, onde orientadores gigantes trazem alunos isolados para o centro da rede.

        **4. Coeficiente Rich-Club (Φ): Mede a Coesão da Elite**
        * **O que significa?** Avalia exclusivamente a "elite" do programa (o top 20% com mais conexões). Eles competem em silos ou colaboram?
        * **Diagnóstico:** Se Φ é alto, a "elite é coesa": os maiores orientadores participam das bancas uns dos outros e fundem os grandes temas. Se for baixo, indica que o PPG possui "caciques" que não dialogam entre si, gerando fragmentação política e científica.
        """)

    st.markdown("---")

    # --- MÓDULO DE MACROTEMAS ---
    st.header("🧠 Análise Temática Estrutural")

    # --- NOVA TABELA ROBUSTA DE MACROTEMAS ---
    O_total = len(dados_completos)
    contagem_ori = Counter([d.get('orientador') for d in dados_completos if d.get('orientador')])
    contagem_coori = Counter([co for d in dados_completos for co in d.get('co_orientadores', [])])

    linhas_tabela = []
    macrotemas_unicos = set([d.get('macrotema', 'Multidisciplinar / Transversal') for d in dados_completos])

    # CORRIGIDO: top_ql movida para fora do loop. Defini-la dentro do loop
    # a redefinia desnecessariamente a cada iteração (antipattern Python).
    def top_ql(entidades_na_mt, contagem_global, O_k, O_total):
        max_ql = -1
        top_ent = "-"
        contagem_local = Counter(entidades_na_mt)
        for ent, O_ik in contagem_local.items():
            O_i = contagem_global.get(ent, 0)
            if O_i > 0 and O_k > 0:
                ql = (O_ik / O_i) / (O_k / O_total)
                if ql > max_ql:
                    max_ql = ql
                    top_ent = ent
                elif ql == max_ql:
                    if O_ik > contagem_local.get(top_ent, 0):
                        top_ent = ent
        return top_ent, max_ql

    for mt in macrotemas_unicos:
        docs_mt = [d for d in dados_completos if d.get('macrotema', 'Multidisciplinar / Transversal') == mt]
        O_k = len(docs_mt)

        teses = sum(1 for d in docs_mt if 'Tese' in d.get('nivel_academico', ''))
        dissertacoes = sum(1 for d in docs_mt if 'Disserta' in d.get('nivel_academico', ''))

        anos = [int(d['ano']) for d in docs_mt if d.get('ano') and str(d['ano']).isdigit()]
        ano_antigo = min(anos) if anos else "-"
        ano_recente = max(anos) if anos else "-"
        ano_modal = Counter(anos).most_common(1)[0][0] if anos else "-"

        oris_mt = [d.get('orientador') for d in docs_mt if d.get('orientador')]
        top_ori, ql_ori = top_ql(oris_mt, contagem_ori, O_k, O_total)

        cooris_mt = [co for d in docs_mt for co in d.get('co_orientadores', [])]
        top_coori, ql_coori = top_ql(cooris_mt, contagem_coori, O_k, O_total)

        mt_sna = sna_global.get(mt, {})

        linhas_tabela.append({
            "Macrotema": mt,
            "Docs": O_k,
            "Teses": teses,
            "Dissertações": dissertacoes,
            "Grau": mt_sna.get('Grau Absoluto', 0),
            "Betweenness": round(mt_sna.get('Betweenness', 0.0), 4),
            "Closeness": round(mt_sna.get('Closeness', 0.0), 4),
            "Especialista (Orientador)": f"{top_ori} (QL: {round(ql_ori,1)})" if top_ori != "-" else "-",
            "Especialista (Co-orientador)": f"{top_coori} (QL: {round(ql_coori,1)})" if top_coori != "-" else "-",
            "Início": ano_antigo,
            "Pico Modal": ano_modal,
            "Recente": ano_recente
        })

    df_temas = pd.DataFrame(linhas_tabela).sort_values(by="Docs", ascending=False)

    # --- ABAS DA SEÇÃO TEMÁTICA ---
    tab_tm1, tab_tm2, tab_tm3, tab_tm4 = st.tabs([
        "📊 Tabela Geral",
        "🧭 Mapa Temático (Macrotemas)",
        "🧭 Mapa Temático (Palavras-chave)",
        "🧊 Espaço Topológico 3D"
    ])

    with tab_tm1:
        st.markdown("Consolidação dos dados por categoria metodológica.")
        st.dataframe(df_temas, width='stretch', hide_index=True)

    with tab_tm2:
        st.markdown("O mapa classifica os macrotemas em 4 quadrantes baseados em seu papel ecológico na rede.")
        if not df_temas.empty:
            fig_mapa_macro = plotar_mapa_tematico(
                df_plot=df_temas,
                x_col="Betweenness",
                y_col="Grau",
                size_col="Docs",
                label_col="Macrotema",
                title="Thematic Map - Macrotemas do PPG"
            )
            st.plotly_chart(fig_mapa_macro, width='stretch')

    with tab_tm3:
        st.markdown("Análise granulada dos micro-conceitos mais frequentes.")

        kw_data = []
        todas_pks = [pk for d in dados_completos for pk in d.get('palavras_chave', [])]
        kw_counter = Counter(todas_pks)
        top_kws = [kw for kw, count in kw_counter.most_common(40)]

        for kw in top_kws:
            kw_sna = sna_global.get(kw, {})
            kw_data.append({
                "Palavra-chave": kw,
                "Frequência": kw_counter[kw],
                "Betweenness": kw_sna.get('Betweenness', 0.0),
                "Grau": kw_sna.get('Grau Absoluto', 0)
            })

        df_kw = pd.DataFrame(kw_data)

        if not df_kw.empty:
            fig_mapa_kw = plotar_mapa_tematico(
                df_plot=df_kw,
                x_col="Betweenness",
                y_col="Grau",
                size_col="Frequência",
                label_col="Palavra-chave",
                title="Thematic Map - Top 40 Palavras-chave"
            )
            st.plotly_chart(fig_mapa_kw, width='stretch')

    with tab_tm4:
        st.markdown("### 🧊 Distribuição Espacial do Ecossistema")
        st.caption("Esta visualização tridimensional permite identificar a arquitetura da rede. Itens no topo do eixo **Betweenness** são os grandes intermediadores, enquanto o eixo **Closeness** revela os centros nervosos de disseminação.")

        categoria_3d = st.radio(
            "Selecione a Dimensão para Visualizar:",
            ["Documento", "Autor", "Orientador", "Palavra-chave", "Macrotema"],
            horizontal=True,
            key="selector_3d_global"
        )

        with st.spinner(f"Processando geometria 3D para {categoria_3d}s..."):
            fig_3d_global = plotar_grafico_3d_sna(sna_global, categoria_3d)

            if fig_3d_global:
                st.plotly_chart(fig_3d_global, width='stretch', height=800)
            else:
                st.warning(f"Não há dados suficientes para gerar o gráfico 3D de {categoria_3d}s.")

    st.markdown("---")
    opcao_exploracao = st.radio(
        "Navegue pelas ferramentas de Exploração Global:",
        ["🕸️ Grafo Interativo", "🏆 Análise Estrutural", "📈 Evolução Histórica", "☁️ Lexicometria", "🔗 Co-ocorrência", "📥 Exportação"],
        horizontal=True
    )

    if opcao_exploracao == "🕸️ Grafo Interativo":
        st.header("Topologia e Grafo Interativo")
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
                    btn_render_grafo = st.form_submit_button("Renderizar Grafo", width='stretch')

            if btn_render_grafo:
                with st.spinner("A construir a rede topológica visual..."):
                    nodes, edges, legendas, G_obj = gerar_nodos_globais_agraph(
                        dados_grafo[:n_registros_grafo],
                        metodo_cor=metodo_coloracao,
                        metodo_tamanho=metodo_tamanho
                    )
                    st.session_state['graf_glob_nodes'] = nodes
                    st.session_state['graf_glob_edges'] = edges
                    st.session_state['kpis_grafo'] = {'nos': len(nodes), 'arestas': len(edges), 'legendas': legendas}
                    st.session_state['G_atual'] = G_obj
                    st.session_state['grafo_pronto'] = True

            if st.session_state['grafo_pronto']:
                kpis = st.session_state['kpis_grafo']

                st.markdown("### 📥 Exportar Estrutura de Rede")
                col_ex1, col_ex2, col_ex3 = st.columns(3)
                G_para_exportar = st.session_state.get('G_atual')

                if G_para_exportar:
                    data_gexf, nome_gexf = preparar_exportacao_grafo(G_para_exportar, "GEXF (Gephi)")
                    col_ex1.download_button("📂 Exportar para Gephi", data=data_gexf, file_name=nome_gexf, width='stretch')
                    data_ml, nome_ml = preparar_exportacao_grafo(G_para_exportar, "GraphML")
                    col_ex2.download_button("💾 Exportar GraphML", data=data_ml, file_name=nome_ml, width='stretch')
                    data_json, nome_json = preparar_exportacao_grafo(G_para_exportar, "JSON (Node-Link)")
                    col_ex3.download_button("🌐 Exportar JSON Web", data=data_json, file_name=nome_json, width='stretch')

                config = Config(
                    width="100%",
                    height=650,
                    directed=False,
                    physics=True,
                    nodeHighlightBehavior=True,
                    highlightColor="#F1C40F",
                    interaction={"navigationButtons": True, "keyboard": True}
                )
                agraph(nodes=st.session_state['graf_glob_nodes'], edges=st.session_state['graf_glob_edges'], config=config)
        else:
            st.warning("Nenhum documento selecionado para o Grafo.")

    elif opcao_exploracao == "🏆 Análise Estrutural":
        st.header("Análise Estrutural e Rankings (SNA)")
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
            todas_metricas = ["Grau Absoluto", "Degree Centrality", "Betweenness", "Closeness"]

            with col_t3:
                cat_sel = st.multiselect("Categorias a exibir na tabela:", categorias_disp, default=["Orientador", "Conceito"])
            with col_t4:
                met_sel = st.multiselect("Métricas a exibir:", todas_metricas, default=["Grau Absoluto", "Betweenness", "Closeness"])
            with col_t5:
                met_ord = st.selectbox("Ordenar Ranking primariamente por:", met_sel if met_sel else todas_metricas)

            btn_render_tabela = st.form_submit_button("Processar e Atualizar Tabela", type="primary")

        if btn_render_tabela:
            if met_sel and cat_sel:
                dados_tab_filtrados = []
                for d in dados_completos[:n_registros_tabela]:
                    if d.get('nivel_academico', 'Outros') not in niveis_sel_tabela:
                        continue
                    ano_d = int(d.get('ano')) if d.get('ano') and str(d.get('ano')).isdigit() else None
                    if not ano_d or ano_d < anos_sel_tabela[0] or ano_d > anos_sel_tabela[1]:
                        continue
                    if conceitos_contexto:
                        pks_doc = set(d.get('palavras_chave', []))
                        if not any(c in pks_doc for c in conceitos_contexto):
                            continue
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
            if 'Degree Centrality' in df_exibicao.columns:
                df_exibicao['Degree Centrality'] = df_exibicao['Degree Centrality'].apply(lambda x: f"{x:.4f}")
            if 'Betweenness' in df_exibicao.columns:
                df_exibicao['Betweenness'] = df_exibicao['Betweenness'].apply(lambda x: f"{x:.4f}")
            if 'Closeness' in df_exibicao.columns:
                df_exibicao['Closeness'] = df_exibicao['Closeness'].apply(lambda x: f"{x:.4f}")

            st.dataframe(df_exibicao[colunas], width='stretch', hide_index=True)

    elif opcao_exploracao == "📈 Evolução Histórica":
        st.header("Evolução Histórica (Temporal)")
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
                    st.plotly_chart(fig, width='stretch')

    elif opcao_exploracao == "☁️ Lexicometria":
        st.header("Lexicometria e Nuvem de Palavras")
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

        if btn_render_nuvem:
            df_geral_base = preparar_dados_base_df(dados_completos)
            if not df_geral_base.empty:
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
                        html_nuvem = renderizar_nuvem_interativa_html_exploracao(freq_dict)
                        components.html(html_nuvem, height=480, scrolling=False)

    elif opcao_exploracao == "🔗 Co-ocorrência":
        st.header("Grafo de Co-ocorrência de Palavras")
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
                if d.get('nivel_academico', 'Outros') not in niveis_sel_co:
                    continue
                ano_d = int(d.get('ano')) if d.get('ano') and str(d.get('ano')).isdigit() else None
                if not ano_d or ano_d < anos_sel_co[0] or ano_d > anos_sel_co[1]:
                    continue
                if orientador_sel_co and d.get('orientador', 'Não informado') not in orientador_sel_co:
                    continue
                dados_co.append(d)

            if not dados_co:
                st.warning("Não há documentos nos filtros selecionados para a Co-ocorrência.")
            else:
                with st.spinner("A mapear co-ocorrências..."):
                    nodes, edges = gerar_nodos_coocorrencia_agraph(dados_co, min_coocorrencia=min_peso_co)
                    st.session_state['co_nodes'] = nodes
                    st.session_state['co_edges'] = edges
                    st.session_state['coocorrencia_pronta'] = True

        if st.session_state.get('coocorrencia_pronta'):
            kpis_co = st.session_state.get('kpis_co', {'nos': 0, 'arestas': 0})
            nodes_co = st.session_state.get('co_nodes', [])
            edges_co = st.session_state.get('co_edges', [])

            c1, c2, c3 = st.columns(3)
            c1.metric("Conceitos Interligados", kpis_co['nos'])
            c2.metric("Conexões Formadas", kpis_co['arestas'])
            c3.info("Dica: Use o mouse para zoom e arraste os nós para organizar a visão.")

            if not nodes_co:
                st.info("Gere o grafo para visualizar os dados.")
            else:
                config_co = Config(
                    width="100%",
                    height=650,
                    directed=False,
                    physics=True,
                    nodeHighlightBehavior=True,
                    highlightColor="#2ECC71",
                    collapsible=False
                )
                agraph(nodes=nodes_co, edges=edges_co, config=config_co)

    elif opcao_exploracao == "📥 Exportação":
        st.header("Exportação da Base de Dados Bruta")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            json_string = json.dumps(dados_completos, ensure_ascii=False, indent=4)
            st.download_button("📄 Baixar Base Completa (JSON)", file_name="base_ppg.json", mime="application/json", data=json_string)
        with col_b2:
            csv_bytes = preparar_csv_exportacao(dados_completos)
            st.download_button("📊 Baixar Base Completa (CSV)", file_name="base_ppg.csv", mime="text/csv", data=csv_bytes)

    st.markdown("---")

    # ==========================================================
    # 1. FUROS ESTRUTURAIS (BURT)
    # ==========================================================
    st.markdown("#### 🕳️ Furos Estruturais (Burt)")

    with st.expander("📖 O que é a Análise de Furos Estruturais (Burt)?"):
        st.markdown("""
        Esta métrica identifica pesquisadores que atuam como "pontes" entre grupos isolados de conhecimento.
        * **Restrição (Constraint):** Mede o quão "preso" o pesquisador está a uma única rede. **Baixa restrição** indica que ele transita por várias áreas diferentes (como um *Broker* da inovação). **Alta restrição** significa que ele atua dentro de "bolhas" de conhecimento.
        * **Intermediação (Betweenness):** Representada pelo tamanho da bolha no gráfico. Indica o volume de caminhos da rede que passam obrigatoriamente por este pesquisador.
        * **Como ler o gráfico:** Os pontos mais propensos à inovação interdisciplinar estão no canto **inferior direito** (Alta Diversidade, Baixa Restrição com bolhas grandes).
        """)

    df_bt = calcular_burt(dados_gerais)
    if not df_bt.empty:
        # CORRIGIDO: template alterado de "plotly_dark" para "streamlit" para
        # suportar os modos claro e escuro da interface de forma consistente
        # com o restante dos gráficos do arquivo.
        st.plotly_chart(
            px.scatter(
                df_bt,
                x="Diversidade",
                y="Restrição (Constraint)",
                size="Intermediação (Betweenness)",
                hover_name="Orientador",
                color="Restrição (Constraint)",
                color_continuous_scale="RdYlGn_r",
                template="streamlit",
                title="Mapa de Furos Estruturais"
            ),
            width='stretch'
        )
    else:
        st.warning("Dados insuficientes para calcular a métrica de restrição.")

    st.markdown("---")

    # ==========================================================
    # 3. BOXPLOT QL
    # ==========================================================
    st.markdown("#### 📦 Boxplot de Especialização (Quociente Locacional)")
    st.caption("Selecione o tipo de entidade no eixo X e até 5 elementos para comparar a distribuição de QL por nível acadêmico.")

    with st.expander("📖 O que significa o Boxplot de Quociente Locacional (QL)?"):
        st.markdown("""
        Esta visualização analisa a distribuição do grau de **Especialização** de um pesquisador ou conceito nos diferentes níveis acadêmicos (Graduação, Mestrado, Doutorado).
        * **O que é o QL?** Um Valor QL acima de `1.0` indica que o pesquisador atua/produz de forma especializada naquilo, acima da média global da universidade. Um QL abaixo de `1.0` indica uma atuação mais generalista ou pulverizada.
        * **Como ler as caixas:** A "caixa" central representa onde a grande massa dos trabalhos se concentra. A linha no meio é a mediana (o comportamento padrão). Os pontos distantes representam as exceções (outliers).
        * **Para que serve?** Descobrir se um Orientador, por exemplo, possui um foco especializado apenas em suas teses de Doutorado, enquanto orienta projetos de forma mais ampla e generalista na Graduação.
        """)

    tipo_map = {
        "Orientador": "Orientador",
        "Coorientador": "Co-orientador",
        "Palavra-chave": "Palavra-chave",
        "Macrotema": "Macrotema",
    }
    tipo_x = st.selectbox("Tipo no eixo X:", list(tipo_map.keys()), index=0)

    tipo_backend = tipo_map[tipo_x]
    entidades_unicas = set()
    for d in dados_gerais:
        if tipo_backend == "Orientador":
            if d.get("orientador"):
                entidades_unicas.add(d.get("orientador"))
        elif tipo_backend == "Co-orientador":
            for co in d.get("co_orientadores", []):
                if co:
                    entidades_unicas.add(co)
        elif tipo_backend == "Palavra-chave":
            for pk in d.get("palavras_chave", []):
                if pk:
                    entidades_unicas.add(pk)
        else:
            entidades_unicas.add(d.get("macrotema", "Multidisciplinar / Transversal"))

    entidades_sel = st.multiselect(
        "Elementos no eixo X (até 5):",
        options=sorted(list(entidades_unicas)),
        default=sorted(list(entidades_unicas))[:3],
        max_selections=5
    )

    if entidades_sel:
        df_box_ql = gerar_base_boxplot_ql(dados_gerais, tipo_backend, entidades_sel)
        if df_box_ql.empty:
            st.info("Sem dados suficientes para gerar boxplot de QL com os elementos selecionados.")
        else:
            fig_box = px.box(
                df_box_ql,
                x="Entidade",
                y="Valor QL",
                color="Entidade",
                points="all",
                hover_data=["Nível"],
                template="streamlit",
                title=f"Distribuição Estatística de Especialização (QL) - {tipo_x}"
            )
            fig_box.update_layout(
                xaxis_title=tipo_x,
                yaxis_title="Índice de Especialização (Valor QL)",
                showlegend=False,
                margin=dict(t=50, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_box, width='stretch')
    else:
        st.info("Selecione pelo menos um elemento para gerar o boxplot de QL.")

# =========================================================================
# ABAS DE ANÁLISE AVANÇADA (Abas 2 e 3)
# =========================================================================

with tab_fluxos:
    st.markdown("### Fluxos de Conhecimento e Estruturas")

    # ==========================================================
    # 2. SANKEY TEMPORAL
    # ==========================================================
    st.markdown("#### 🌊 Sankey Temporal de Palavras-chave")
    st.caption("Analise a persistência e o fluxo de tópicos de pesquisa entre três períodos de anos selecionáveis.")

    with st.expander("📖 Como interpretar o Sankey Temporal?"):
        st.markdown("""
        O Diagrama de Sankey visualiza o "fluxo" e a herança genética das ideias (Palavras-chave) ao longo do tempo.
        * **Os Blocos (Nós):** Representam os temas mais estudados em cada um dos 3 períodos. A altura do bloco reflete o volume de documentos sobre o tema.
        * **Os Caminhos (Linhas):** Mostram como o conhecimento transitou. Uma linha entre o Período 1 e o Período 2 significa que os *mesmos pesquisadores* (ou seus orientandos diretos) mantiveram e evoluíram aquele tema para a próxima camada temporal.
        * **Para que serve?** Avaliar se o programa mantém linhas de pesquisa duradouras ou se o foco de sua elite acadêmica sofre rupturas bruscas.
        """)

    df_geral_base = pd.DataFrame(dados_gerais)
    df_geral_base['Ano'] = pd.to_numeric(df_geral_base['ano'], errors='coerce')
    min_ano = int(df_geral_base['Ano'].min()) if not df_geral_base.empty else 2000
    max_ano = int(df_geral_base['Ano'].max()) if not df_geral_base.empty else 2026

    total_years = max_ano - min_ano
    part_years = max(total_years // 3, 1)

    st.markdown("##### 🔍 Defina os Períodos e Parâmetros")
    cf_p1, cf_p2, cf_p3 = st.columns(3)
    with cf_p1:
        p1_anios = st.slider("Período 1 (Anos):", min_value=min_ano, max_value=max_ano, value=(min_ano, min(min_ano + part_years, max_ano)), step=1)
        p1_range = (datetime.date(p1_anios[0], 1, 1), datetime.date(p1_anios[1], 12, 31))
    with cf_p2:
        p2_anios = st.slider("Período 2 (Anos):", min_value=min_ano, max_value=max_ano, value=(min(min_ano + part_years, max_ano), min(min_ano + 2*part_years, max_ano)), step=1)
        p2_range = (datetime.date(p2_anios[0], 1, 1), datetime.date(p2_anios[1], 12, 31))
    with cf_p3:
        p3_anios = st.slider("Período 3 (Anos):", min_value=min_ano, max_value=max_ano, value=(min(min_ano + 2*part_years, max_ano), max_ano), step=1)
        p3_range = (datetime.date(p3_anios[0], 1, 1), datetime.date(p3_anios[1], 12, 31))

    n_kw = st.slider("Qtd Palavras-chave Principais por Período:", 3, 20, 10, 1)

    st.markdown("<br>", unsafe_allow_html=True)

    if all(isinstance(r, tuple) and len(r) == 2 for r in [p1_range, p2_range, p3_range]):
        nodes_lbl, src, tgt, val = preparar_sankey_temporal(dados_gerais, n_kw, p1_range, p2_range, p3_range)

        if nodes_lbl:
            fig_sk = go.Figure(go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    label=nodes_lbl,
                    color="#F39C12"
                ),
                link=dict(source=src, target=tgt, value=val, color="rgba(243, 156, 18, 0.4)")
            ))

            fig_sk.update_layout(
                margin=dict(l=0, r=0, t=20, b=20)
            )

            html_sankey = fig_sk.to_html(include_plotlyjs="cdn", full_html=False, config={'displayModeBar': False})

            css_sankey = """
            <style>
            .sankey-node text {
                fill: white !important;
                font-family: 'Helvetica Neue', sans-serif !important;
                font-weight: bold !important;
                text-shadow: 
                    -1px -1px 0 #000,  
                     1px -1px 0 #000,
                    -1px  1px 0 #000,
                     1px  1px 0 #000,
                     2px  2px 4px rgba(0,0,0,0.9) !important;
                text-decoration: underline !important;
                text-decoration-color: black !important;
            }
            </style>
            """

            components.html(css_sankey + html_sankey, height=750)

        else:
            st.warning("Não foi possível detectar fluxos de palavras-chave comuns entre os períodos selecionados. Tente aumentar o número de anos nos períodos.")
    else:
        st.warning("Por favor, selecione um intervalo de datas para os três períodos.")

    # ==========================================================
    # RADAR DE PROSPECÇÃO (FORESIGHT ACADÊMICO)
    # ==========================================================
    st.markdown("---")
    st.markdown("#### 🔮 Radar de Prospecção (Foresight Acadêmico)")
    st.caption("Identifique Sinais Fracos e Tendências Emergentes antes que se tornem mainstream.")

    with st.expander("📖 Como interpretar o Radar de Foresight (Modelo de Sinais Fracos)?"):
        st.markdown("""
        Baseado em modelos de Inteligência Antecipatória, este gráfico cruza a **Visibilidade** (Crescimento recente em volume) com a **Novidade** (Intermediação/Betweenness na rede).
        
        * ↖️ **Sinais Fracos (Weak Signals):** Baixo crescimento de volume, mas **Alta Novidade** (atuam como pontes raras entre áreas isoladas). São as sementes das próximas inovações.
        * ↗️ **Tendências Emergentes (Bursts):** Alto crescimento e **Alta Novidade**. O sinal fraco "explodiu" e está se consolidando rapidamente na fronteira do conhecimento.
        * ↘️ **Mainstream / Consolidados:** Alto crescimento, mas **Baixa Novidade**. O tema já dominou o ecossistema e formou uma "bolha" própria, perdendo sua característica de inovação de fronteira.
        * ↙️ **Ruído / Declínio:** Baixo volume e Baixa novidade. Temas saturados ou que não ganharam tração.
        """)

    col_for1, col_for2 = st.columns(2)
    with col_for1:
        tipo_foresight = st.selectbox("Analisar Emergência de:", ["Palavra-chave", "Macrotema"])
    with col_for2:
        janela_anos = st.slider("Janela Recente (Últimos X anos para medir):", min_value=1, max_value=10, value=3)

    # CÓDIGO NOVO — com controle de bootstrap
    sna_global = st.session_state['sna_global']
    if sna_global is None:
        with st.spinner("Sincronizando topologia para o Radar..."):
            sna_global = calcular_sna_global(dados_gerais)
            st.session_state['sna_global'] = sna_global

    # Controle de robustez estatística (3ª coluna no layout)
    col_for3 = st.columns(1)[0]
    usar_bootstrap = st.checkbox(
        "🎲 Usar Betweenness Bootstrap (mais robusto, ~30s extra)",
        value=False,
        help="Reamostra a rede 100 vezes com reposição e usa a mediana. "
            "Recomendado para bases pequenas (<300 documentos) ou quando você quer "
            "eliminar ruído de termos marginais. Custa ~20–40s de processamento."
    )

    from backend import calcular_betweenness_bootstrap
    bet_bootstrap = None
    if usar_bootstrap:
        # Cache-key inclui tipo para não misturar Palavra-chave com Macrotema
        cache_key = f'bet_bootstrap_{tipo_foresight}'
        if cache_key not in st.session_state:
            with st.spinner("🎲 Executando 100 reamostragens bootstrap... (esta operação é cacheada)"):
                st.session_state[cache_key] = calcular_betweenness_bootstrap(
                    dados_gerais, tipo=tipo_foresight, n_bootstrap=100
                )
        bet_bootstrap = st.session_state[cache_key]

        # Feedback ao usuário
        n_termos_estaveis = len(bet_bootstrap) if bet_bootstrap else 0
        st.caption(f"✅ Bootstrap completo — {n_termos_estaveis} termos com betweenness estatisticamente estável.")

    df_foresight = preparar_radar_foresight(
        dados_gerais,
        sna_global,
        janela_recente=janela_anos,
        tipo=tipo_foresight,
        bet_bootstrap=bet_bootstrap  # NOVO
    )

    # CORRIGIDO: guard substituído de .sum() == 0 (frágil para floats próximos
    # de zero por imprecisão numérica) para .max() < 1e-9 (numericamente robusto).
    if df_foresight.empty or df_foresight['Novidade (Estrutural * IDF)'].max() < 1e-9:
        st.info("Não há dados recentes ou de rede suficientes para plotar o Radar de Prospecção.")
    else:
        # CÓDIGO NOVO — UI de controle da estratégia de segmentação
        st.markdown("##### ⚙️ Configurações de Segmentação do Radar")
        col_seg1, col_seg2 = st.columns([1, 2])

        with col_seg1:
            metodo_corte = st.radio(
                "Método de segmentação:",
                ["Percentil fixo", "K-Means adaptativo (4 clusters)"],
                horizontal=False,
                help="Percentil: você define onde estão as linhas de corte. "
                    "K-Means: o algoritmo encontra 4 agrupamentos naturais nos dados."
            )

        with col_seg2:
            # CÓDIGO NOVO — cálculo dual de fronteiras
            if metodo_corte == "Percentil fixo":
                percentil_corte = st.slider(
                    "Percentil de corte (quanto maior, mais rigoroso):",
                    min_value=50, max_value=85, value=65, step=5,
                    help="65% é o padrão acadêmico. 50% usa a mediana (mais inclusivo). "
                        "80%+ isola apenas os extremos (útil em bases grandes)."
                ) / 100.0

                x_mid = df_foresight['Momentum (Burst)'].quantile(percentil_corte)
                y_mid = df_foresight['Novidade (Estrutural * IDF)'].quantile(percentil_corte)
                df_foresight['Quadrante'] = df_foresight.apply(
                    lambda r: classificar_por_percentil(r, x_mid, y_mid), axis=1
                )
            else:
                percentil_corte = None  # não usado em K-Means
                st.caption(
                    "ℹ️ O K-Means encontra automaticamente 4 clusters no espaço "
                    "(Momentum × Novidade). As fronteiras entre quadrantes são definidas "
                    "pelos centroides, não por cortes fixos."
                )
                
                # K-Means 4-cluster adaptativo
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler

                X = df_foresight[['Momentum (Burst)', 'Novidade (Estrutural * IDF)']].values

                # Padronização é crítica — Momentum e Novidade têm escalas muito diferentes
                scaler = StandardScaler()
                X_std = scaler.fit_transform(X)

                # Fit com seed fixo para estabilidade visual
                kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                df_foresight['cluster_id'] = kmeans.fit_predict(X_std)

                # Mapeia os 4 clusters aos 4 quadrantes semânticos analisando os centroides
                # Centroides estão no espaço padronizado — convertemos de volta à média
                centroides_std = kmeans.cluster_centers_
                # Rankear cada cluster por momentum e novidade médios
                clusters_info = []
                for cid in range(4):
                    cluster_pts = df_foresight[df_foresight['cluster_id'] == cid]
                    clusters_info.append({
                        'id': cid,
                        'mom_medio': cluster_pts['Momentum (Burst)'].mean(),
                        'nov_media': cluster_pts['Novidade (Estrutural * IDF)'].mean(),
                    })

                # Mapeamento semântico: alto-alto = Tendência; baixo-alto = Sinal Fraco; etc.
                mom_mediano = np.median([c['mom_medio'] for c in clusters_info])
                nov_mediana = np.median([c['nov_media'] for c in clusters_info])
                mapa_cluster_quadrante = {}
                for c in clusters_info:
                    alto_mom = c['mom_medio'] >= mom_mediano
                    alta_nov = c['nov_media'] >= nov_mediana
                    if alto_mom and alta_nov:
                        mapa_cluster_quadrante[c['id']] = "↗️ Tendência"
                    elif not alto_mom and alta_nov:
                        mapa_cluster_quadrante[c['id']] = "↖️ Sinal Fraco"
                    elif alto_mom and not alta_nov:
                        mapa_cluster_quadrante[c['id']] = "↘️ Mainstream"
                    else:
                        mapa_cluster_quadrante[c['id']] = "↙️ Base/Declínio"

                df_foresight['Quadrante'] = df_foresight['cluster_id'].map(mapa_cluster_quadrante)

                # Para desenhar linhas divisórias "visuais" no gráfico, usamos as medianas
                # dos centroides como approximação das fronteiras
                x_mid = mom_mediano
                y_mid = nov_mediana

        # CORRIGIDO: hover_data agora é construído dinamicamente para evitar
        # ValueError se o backend retornar um dataframe sem essas colunas.
        colunas_hover_desejadas = ["Tração (%)", "Aparições Recentes"]
        hover_cols = [c for c in colunas_hover_desejadas if c in df_foresight.columns]

        # CORRIGIDO: template alterado de "plotly_dark" para "streamlit" para
        # consistência com o tema da aplicação (suporte a light/dark mode).
        fig_radar = px.scatter(
            df_foresight,
            x="Momentum (Burst)",
            y="Novidade (Estrutural * IDF)",
            size="Total",
            hover_name="Termo",
            hover_data=hover_cols,
            color="Quadrante",
            color_discrete_map={
                "↗️ Tendência": "#2ECC71",
                "↖️ Sinal Fraco": "#F1C40F",
                "↘️ Mainstream": "#3498DB",
                "↙️ Base/Declínio": "#E74C3C"
            },
            size_max=35,
            template="streamlit"
        )

        fig_radar.add_vline(x=x_mid, line_dash="dash", line_color="#7F8C8D", line_width=1)
        fig_radar.add_hline(y=y_mid, line_dash="dash", line_color="#7F8C8D", line_width=1)

        # CORRIGIDO: anotações migradas para coordenadas de paper (0.0–1.0)
        # em vez de coordenadas de dados (x_max, y_min, etc.). Isso garante
        # que os rótulos fiquem sempre nos cantos do gráfico, sem sobrepor
        # os pontos de dados mais extremos — que são exatamente os mais importantes.
        annotations = [
            dict(xref="paper", yref="paper", x=0.99, y=0.99,
                 text="<b>↗️ Tendências (Bursts Validados)</b>",
                 showarrow=False, font=dict(color="#2ECC71", size=14),
                 xanchor="right", yanchor="top"),
            dict(xref="paper", yref="paper", x=0.01, y=0.99,
                 text="<b>↖️ Sinais Fracos (Pivôs Raros)</b>",
                 showarrow=False, font=dict(color="#F1C40F", size=14),
                 xanchor="left", yanchor="top"),
            dict(xref="paper", yref="paper", x=0.01, y=0.01,
                 text="<b>↙️ Base / Declínio</b>",
                 showarrow=False, font=dict(color="#E74C3C", size=14),
                 xanchor="left", yanchor="bottom"),
            dict(xref="paper", yref="paper", x=0.99, y=0.01,
                 text="<b>↘️ Consolidação (Mainstream)</b>",
                 showarrow=False, font=dict(color="#3498DB", size=14),
                 xanchor="right", yanchor="bottom")
        ]

        fig_radar.update_layout(
            annotations=annotations,
            xaxis_title="Momentum Temporal (Tração Ponderada pelo Volume)",
            yaxis_title="Novidade Sistêmica (Betweenness Centrality × IDF)",
            coloraxis_showscale=False,
            height=650,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig_radar, width='stretch')

    # ==========================================================
    # 5. AUTO-ML: GRID SEARCH E VALIDAÇÃO DE ROBUSTEZ (F1-SCORE)
    # ==========================================================
    st.markdown("---")
    st.markdown("#### 🤖 Laboratório Automático: Validação do Modelo (Grid Search)")
    st.caption("O algoritmo iterará sozinho sobre múltiplos cenários no tempo para testar a robustez matemática das suas previsões.")

    with st.expander("📖 Entenda as Métricas de Validação (Matriz de Confusão)"):
        st.markdown("""
        O algoritmo viaja no tempo diversas vezes e mede seus acertos contra a realidade.
        * **Precisão:** De todos os temas que o modelo cravou que seriam inovações ("Sinais Fracos"), quantos *realmente* cresceram? (Mede a taxa de falsos alarmes).
        * **Recall (Revogação):** De todas as inovações que realmente aconteceram, quantas o modelo *conseguiu prever* antecipadamente? (Mede as oportunidades perdidas).
        * **F1-Score:** A nota final (0 a 1.0) do modelo, combinando Precisão e Recall. Valores acima de `0.6` em dados textuais de humanas/engenharia já são considerados altamente preditivos.
        """)

    if st.button("🚀 Executar Bateria de Testes (Auto-Otimização)", type="primary"):
        # CORRIGIDO: mensagens de progresso movidas para dentro de um callback
        # sequencial com st.empty() para refletir o progresso real da operação,
        # em vez de exibir todos os passos concluídos antes do processamento.
        _status_placeholder = st.empty()

        _status_placeholder.info("⏳ Passo 1/3 — Separando recortes de tempo históricos...")
        df_grid_resultado = otimizar_parametros_foresight(dados_gerais, tipo=tipo_foresight)

        if df_grid_resultado.empty:
            _status_placeholder.error("❌ Dados insuficientes para rodar o Grid Search no histórico.")
        else:
            _status_placeholder.success("✅ Bateria de Testes Concluída!")
            # CORRIGIDO: resultado e tipo utilizado persistidos em session_state
            # para sobreviver à navegação entre abas e rerenders do Streamlit.
            st.session_state['df_grid'] = df_grid_resultado
            st.session_state['tipo_foresight_grid'] = tipo_foresight

    # CORRIGIDO: renderização dos resultados movida para fora do bloco do botão.
    # Assim os resultados permanecem visíveis após qualquer rerender, enquanto o
    # usuário não acionar um novo Grid Search.
    if st.session_state.get('df_grid') is not None and not st.session_state['df_grid'].empty:
        df_grid = st.session_state['df_grid']
        tipo_grid_exibido = st.session_state.get('tipo_foresight_grid', '—')

        # Aviso de inconsistência quando o tipo atual difere do usado no último Grid Search
        if tipo_grid_exibido != tipo_foresight:
            st.warning(
                f"⚠️ Os resultados abaixo foram calculados para **{tipo_grid_exibido}**. "
                f"O tipo atual é **{tipo_foresight}**. Execute novamente para atualizar."
            )

        melhor_mcc = df_grid.iloc[0]['MCC (Robusto)']
        melhor_f1 = df_grid.iloc[0]['F1-Score']
        melhor_prec = df_grid.iloc[0]['Precisão']
        melhor_rec = df_grid.iloc[0]['Recall']

        st.markdown("##### 🏆 Desempenho Máximo Atingido")
        col_m0, col_m1, col_m2, col_m3 = st.columns(4)

        col_m0.metric("Score MCC", f"{melhor_mcc:.3f}",
                      "Sinal Preditivo Real" if melhor_mcc >= 0.15 else "Abaixo do Acaso",
                      delta_color="normal")
        col_m1.metric("F1-Score", f"{melhor_f1:.3f}")
        col_m2.metric("Precisão", f"{melhor_prec:.1%}",
                      "Baixa taxa falsos alarmes" if melhor_prec > 0.5 else "Alarmes falsos",
                      delta_color="normal" if melhor_prec > 0.5 else "inverse")
        col_m3.metric("Recall", f"{melhor_rec:.1%}",
                      "Detecta inovações" if melhor_rec > 0.5 else "Perde inovações",
                      delta_color="normal" if melhor_rec > 0.5 else "inverse")

        st.markdown("##### ⚙️ Ranking de Calibragem do Algoritmo")
        st.caption("A tabela abaixo mostra quais parâmetros de janela de tempo geram as melhores previsões para a sua base específica.")

        def highlight_first(s):
            return ['background-color: rgba(46, 204, 113, 0.2)' if s.name == 0 else '' for v in s]

        st.dataframe(df_grid.style.apply(highlight_first, axis=1), width='stretch')


with tab_memes:
    st.markdown("### A Genética das Ideias")
    st.write("Analise o ciclo de vida de teorias, modelos e ferramentas que se replicam pelo programa.")

    # --- MÓDULO DE EXTRAÇÃO ONTOLÓGICA (LLM) ---
    st.markdown("#### 🤖 Mineração de Artefatos (Google Gemini)")
    st.caption("A API lerá os resumos para extrair construtos reais (Artefatos, Teorias e Métodos), superando a limitação das palavras-chave genéricas.")

    qtd_com_ontologia = sum(1 for d in dados_gerais if 'ontologia_ia' in d and d['ontologia_ia'])
    qtd_total_validos = sum(1 for d in dados_gerais if d.get('resumo') and str(d.get('resumo')).strip() != "")

    col_ai1, col_ai2 = st.columns([2, 1])
    with col_ai1:
        st.progress(qtd_com_ontologia / max(qtd_total_validos, 1), text=f"Progresso da Base: {qtd_com_ontologia} de {qtd_total_validos} resumos lidos pela IA.")

    with col_ai2:
        tamanho_lote = st.selectbox(
            "Tamanho do Lote:",
            [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
            index=1,
            help="Pequenos lotes evitam que o limite da API gratuita estoure.",
            key="lote_ia_memetica"
        )
        btn_iniciar_ia = st.button("🚀 Processar Próximo Lote", type="primary", width='stretch')

    if btn_iniciar_ia:
        if qtd_com_ontologia >= qtd_total_validos:
            st.success("Toda a base já foi processada pela IA!")
        else:
            barra_ia = st.progress(0)
            status_ia = st.empty()

            proc, faltam = processar_lote_ontologia(dados_gerais, tamanho_lote, barra_ia, status_ia)

            st.session_state['dados_completos'] = dados_gerais

            status_ia.success(f"Lote concluído! {proc} documentos enriquecidos. Faltam {faltam} na fila.")
            time.sleep(2)
            st.rerun()

    st.markdown("---")

    # --- TABELA DE VISUALIZAÇÃO E EXPORTAÇÃO DA IA (BLINDADA) ---
    st.markdown("#### 🗂️ Catálogo de Artefatos Extraídos (Base de Conhecimento)")
    st.caption("Verifique e baixe o que a inteligência artificial conseguiu encontrar nos documentos processados até o momento.")

    dados_ontologia = []
    for d in dados_gerais:
        if 'ontologia_ia' in d and d['ontologia_ia']:
            onto = d['ontologia_ia']

            if isinstance(onto, str):
                try:
                    onto = json.loads(onto)
                except Exception:
                    continue

            if isinstance(onto, dict):
                def safe_join(item):
                    if isinstance(item, list):
                        return ", ".join([str(x) for x in item])
                    return str(item) if item else ""

                dados_ontologia.append({
                    'Ano': d.get('ano', 'N/A'),
                    'Título': d.get('titulo', 'Sem Título'),
                    'Teorias e Modelos': safe_join(onto.get('teorias_e_modelos', [])),
                    'Ferramentas e Artefatos': safe_join(onto.get('ferramentas_e_artefatos', [])),
                    'Métodos e Técnicas': safe_join(onto.get('metodos_e_tecnicas', []))
                })

    if dados_ontologia:
        df_onto = pd.DataFrame(dados_ontologia)

        st.dataframe(df_onto, width='stretch', hide_index=True, height=250)

        csv_onto = df_onto.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Exportar Catálogo Ontológico (CSV)",
            data=csv_onto,
            file_name="catalogo_ontologico_ia.csv",
            mime="text/csv",
            width='stretch',
            key="btn_export_onto"
        )
    else:
        st.info("💡 A base ainda não possui ontologia estruturada. Processe um lote na seção acima.")

    st.markdown("---")

    # --- MÉTRICAS TRADICIONAIS DE MEMÉTICA COM TOGGLE ---
    st.markdown("### A Genética das Ideias (Visão Geral de Propagação)")

    fonte_selecionada = st.radio(
        "Selecione o prisma da análise de propagação memética:",
        ["Palavras-chave e Títulos (Tradicional)", "Artefatos Extraídos pela IA (Ontologia)"],
        horizontal=True,
        key="seletor_fonte_memes"
    )

    fonte_param = "Artefatos Extraídos" if "IA" in fonte_selecionada else "Palavras-chave"

    # CORRIGIDO: dupla chamada idêntica de calcular_metricas_memeticas removida.
    # A primeira chamada (linhas 1151-1153 originais) era imediatamente sobrescrita
    # pela segunda, causando processamento duplo desnecessário.
    df_fecundidade, df_longevidade, mortos, vivos, df_mortos, df_vivos = calcular_metricas_memeticas(
        df_geral,
        fonte_memes=fonte_param
    )

    if df_fecundidade.empty:
        if fonte_param == "Artefatos Extraídos":
            st.warning("⚠️ Não há artefatos suficientes extraídos para gerar os gráficos. Gere a ontologia usando o botão de processamento acima.")
        else:
            st.warning("⚠️ Dados insuficientes para calcular métricas meméticas.")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Fecundidade vs. Mortalidade Infantil")
            st.caption("Memes que falharam em se replicar vs. Memes que sobreviveram.")

            fig_mortalidade = go.Figure(data=[go.Pie(
                labels=['Memes Mortos (1 aparição)', 'Memes Sobreviventes (>1 aparição)'],
                values=[mortos, vivos],
                hole=.6,
                marker_colors=['#E74C3C', '#2ECC71']
            )])
            fig_mortalidade.update_layout(
                template="streamlit",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_mortalidade, width='stretch', theme="streamlit")

            with st.expander("👁️ Ver Catálogo Completo", expanded=False):
                st.write("**🟢 Memes Sobreviventes (Top Fecundidade)**")
                st.dataframe(df_vivos, width='stretch', hide_index=True, height=250)

                st.write("**🔴 Memes Mortos (Cemitério de Ideias)**")
                amostra_mortos = df_mortos.sample(min(100, len(df_mortos))) if len(df_mortos) > 0 else df_mortos
                st.dataframe(amostra_mortos, width='stretch', hide_index=True, height=250)

        with col2:
            st.markdown("#### Os Super-Memes (Maior Fecundidade)")
            st.caption("Os construtos com maior espalhamento pela história do programa.")

            top_fecundos = df_vivos.head(15).rename(columns={'Memes Sobreviventes': 'meme', 'Nº de Aparições': 'fecundidade'})

            fig_fecundidade = px.bar(
                top_fecundos,
                x='fecundidade',
                y='meme',
                orientation='h',
                color='fecundidade',
                color_continuous_scale='Viridis'
            )
            fig_fecundidade.update_layout(
                template="streamlit",
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Nº de Teses/Dissertações Diferentes",
                yaxis_title="",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_fecundidade, width='stretch', theme="streamlit")

        st.markdown("---")
        st.markdown("#### Tempo de Meia-Vida do Conhecimento (Longevidade)")
        st.caption("Analisa a 'idade' de sobrevivência. Memes com vida longa indicam pilares estruturais.")

        min_aparicoes_long = st.slider("Filtrar por nº mínimo de replicações:", min_value=2, max_value=50, value=2, key="slider_longevidade_memes")

        df_long_filtrado = df_longevidade[df_longevidade['total_aparicoes'] >= min_aparicoes_long].copy()

        if not df_long_filtrado.empty:
            fig_longevidade = px.scatter(
                df_long_filtrado,
                x="ano_nascimento",
                y="tempo_vida_anos",
                size="total_aparicoes",
                color="ano_extincao",
                hover_name="meme",
                color_continuous_scale='Plasma',
                labels={
                    "ano_nascimento": "Ano de Nascimento (1ª Aparição)",
                    "tempo_vida_anos": "Longevidade (Anos de Sobrevivência)",
                    "total_aparicoes": "Total de Réplicas (Fecundidade)",
                    "ano_extincao": "Ano da Última Aparição",
                    "meme": "Meme Acadêmico"
                }
            )
            fig_longevidade.update_layout(template="streamlit", height=500)
            st.plotly_chart(fig_longevidade, width='stretch', theme="streamlit")
        else:
            st.info("Nenhum meme encontrado com essa taxa de replicação mínima.")

        # --- BLOCO UNIFICADO: ECOLOGIA SNA (MEMES E ARTEFATOS) ---
        st.markdown("---")

        if fonte_param == "Artefatos Extraídos":
            st.markdown("#### 🕸️ Ecologia dos Artefatos Ontológicos (SNA)")
            st.caption("Analise as conexões e a maturidade da rede formada exclusivamente pelos artefatos extraídos pela IA.")
        else:
            st.markdown("#### 🕸️ Ecologia Memética Tradicional (SNA)")
            st.caption("Analise as conexões e a maturidade da rede formada pelas palavras-chave e termos isolados dos títulos.")

        min_co_memes = st.slider("Filtro de Co-ocorrência Mínima (Remover ruídos visuais):", min_value=1, max_value=10, value=3, key="slider_co_ecologia")

        with st.spinner(f"Construindo rede de {fonte_param} e calculando topologia global..."):
            from backend import gerar_grafo_ecologia_memes_agraph
            nodes_ecol, edges_ecol, df_nos_ecol, m_sna, maturidade = gerar_grafo_ecologia_memes_agraph(
                dados_gerais,
                min_coocorrencia=min_co_memes,
                fonte_memes=fonte_param
            )

        if nodes_ecol:
            st.markdown("##### 🧬 Métricas de Redes Complexas")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("Densidade da Rede", f"{m_sna['densidade']:.5f}")
            col_s2.metric("Eficiência Global", f"{m_sna['eficiencia']:.4f}")
            col_s3.metric("Entropia (H)", f"{m_sna['entropia']:.2f} bits")
            col_s4.metric("Clustering Médio", f"{m_sna['clustering']:.4f}")

            with st.expander("📊 Estatísticas de Conectividade & Influência (Médias)"):
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    st.markdown("**Conectividade (Links por Nó)**")
                    c_l1, c_l2 = st.columns(2)
                    c_l1.metric("Média de Links", f"{m_sna['links_mean']:.2f}")
                    c_l2.metric("Desvio Padrão", f"{m_sna['links_std']:.2f}")
                    c_l3, c_l4 = st.columns(2)
                    c_l3.metric("Mínimo", int(m_sna['links_min']))
                    c_l4.metric("Máximo", int(m_sna['links_max']))
                with col_exp2:
                    st.markdown("**Influência Estrutural**")
                    c_i1, c_i2 = st.columns(2)
                    c_i1.metric("PageRank Médio", f"{m_sna['pr_avg']:.6f}")
                    c_i2.metric("Eigenvector Médio", f"{m_sna['ev_avg']:.6f}")
                    c_i3, c_i4 = st.columns(2)
                    c_i3.metric("Restrição (Burt)", f"{m_sna['constraint_avg']:.4f}")
                    c_i4.metric("Redundância", f"{m_sna['redundancia']:.4f}")

            st.markdown("##### 🧬 Métricas de Ecologia Profunda (SNA Avançado)")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                g_val = maturidade['gamma']
                status_g = "🟢 Saudável" if 2.0 <= g_val <= 3.0 else ("🔴 Monopolizada" if g_val < 2.0 else "🟡 Fragmentada")
                st.metric("Lei de Potência (γ)", f"{g_val:.2f}", status_g, delta_color="off")
            with col_m2:
                s_val = maturidade['spearman']
                status_s = "🔴 Hierarquia Rígida" if s_val > 0.95 else ("🟢 Inovação (Brokers)" if s_val < 0.85 else "🟡 Equilíbrio")
                st.metric("Correlação de Spearman (ρ)", f"{s_val:.2f}", status_s, delta_color="off")
            with col_m3:
                a_val = maturidade['assortatividade']
                status_a = "🟡 Endogâmica" if a_val > 0.1 else ("🟢 Expansiva" if a_val < -0.1 else "⚪ Neutra")
                st.metric("Assortatividade (r)", f"{a_val:.2f}", status_a, delta_color="off")
            with col_m4:
                rc_val = maturidade['rich_club']
                status_rc = "🟢 Elite Coesa" if rc_val > 0.3 else ("🔴 Hubs Isolados" if rc_val < 0.1 else "🟡 Moderada")
                st.metric("Coeficiente Rich-Club (Φ)", f"{rc_val:.2%}", status_rc, delta_color="off")

            st.markdown("##### 🌌 Grafo Interativo")
            from streamlit_agraph import agraph, Config
            config_art = Config(
                width="100%", height=650, directed=False, physics=True, collapsible=False,
                interaction={
                    "navigationButtons": True, "keyboard": True, "hover": True,
                    "hoverConnectedEdges": True, "selectConnectedEdges": True
                }
            )
            agraph(nodes=nodes_ecol, edges=edges_ecol, config=config_art)

            st.markdown("##### 📊 Tabela de Centralidade Global")
            st.caption("Classificação do poder de influência e intermediação de cada termo/artefato no ecossistema completo.")

            df_nos_exibicao = df_nos_ecol.copy()
            for col in ['Grau (Degree)', 'Betweenness', 'Closeness']:
                if col in df_nos_exibicao.columns:
                    df_nos_exibicao[col] = df_nos_exibicao[col].apply(lambda x: f"{x:.4f}")

            st.dataframe(df_nos_exibicao, width='stretch', hide_index=True)

        else:
            st.info("Não há conexões suficientes neste conjunto de dados.")