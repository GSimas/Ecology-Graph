import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import networkx.algorithms.community as nx_comm
from pyvis.network import Network
import pandas as pd
import json

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Ecologia do Conhecimento UFSC",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILIZAÇÃO CUSTOMIZADA (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; }
    
    /* Estilo dos Cards Estatísticos */
    .stMetric {
        background-color: #262626;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border-left: 5px solid #F39C12;
        transition: transform 0.3s;
    }
    .stMetric:hover {
        transform: translateY(-5px);
        border-left: 5px solid #2ECC71;
    }
    
    h1, h2, h3 { color: #F39C12; font-family: 'Helvetica Neue', sans-serif; }
    
    /* Botão Primário */
    button[kind="primary"] {
        background-color: #2ECC71 !important;
        color: white !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- FUNÇÕES DE CARREGAMENTO ---
@st.cache_data
def carregar_dados():
    try:
        with open('base_ppgegc.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        st.error("Erro ao carregar base_ppgegc.json.")
        return []

# --- PROCESSAMENTO DAS ESTATÍSTICAS ---
dados_completos = carregar_dados()

if dados_completos:
    # Cálculos para os Cards
    total_docs = len(dados_completos)
    teses = len([d for d in dados_completos if "Tese" in d.get('nivel_academico', '')])
    dissertacoes = len([d for d in dados_completos if "Disserta" in d.get('nivel_academico', '')])
    
    # Conjuntos únicos para contagem precisa
    autores_set = set()
    orientadores_set = set()
    coorientadores_set = set()
    keywords_set = set()
    
    for d in dados_completos:
        for a in d.get('autores', []): autores_set.add(a)
        if d.get('orientador'): orientadores_set.add(d.get('orientador'))
        for co in d.get('co_orientadores', []): coorientadores_set.add(co)
        for kw in d.get('palavras_chave', []): keywords_set.add(kw)

    # --- CABEÇALHO E CARDS ---
    st.title("🌌 Ecologia do Conhecimento: PPGEGC")
    st.markdown("### 📊 Panorama Geral da Produção Intelectual")
    
    # Primeira linha de Cards (Documentação)
    c1, c2, c3 = st.columns(3)
    c1.metric("📄 Documentos Totais", total_docs)
    c2.metric("🎓 Teses (Doutorado)", teses)
    c3.metric("📜 Dissertações (Mestrado)", dissertacoes)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Segunda linha de Cards (Atores e Saberes)
    c4, c5, c6, c7 = st.columns(4)
    c4.metric("✍️ Autores Únicos", len(autores_set))
    c5.metric("🏫 Orientadores", len(orientadores_set))
    c6.metric("🤝 Co-orientadores", len(coorientadores_set))
    c7.metric("💡 Conceitos/Keywords", len(keywords_set))

    st.markdown("---")







# === SEÇÃO 1: GRAFO INTERATIVO GERAL ===
st.header("🕸️ 1. Topologia e Grafo Interativo")
niveis_sel_grafo = st.multiselect("Nível Académico (Grafo):", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_grafo")
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
            btn_render_grafo = st.form_submit_button("Renderizar Grafo", use_container_width=True)

    if btn_render_grafo:
        with st.spinner("A construir a rede topológica visual..."):
            path, nos, arestas, legendas = gerar_html_pyvis(dados_grafo[:n_registros_grafo], metodo_cor=metodo_coloracao, metodo_tamanho=metodo_tamanho)
            st.session_state['path_grafo'] = path
            st.session_state['kpis_grafo'] = {'nos': nos, 'arestas': arestas, 'legendas': legendas}
            st.session_state['grafo_pronto'] = True

    if st.session_state['grafo_pronto']:
        kpis = st.session_state['kpis_grafo']
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nós", kpis['nos'])
        c2.metric("Arestas", kpis['arestas'])
        c3.metric("Densidade", f"{(kpis['arestas'] / kpis['nos']):.3f}" if kpis['nos'] > 0 else 0)
        c4.info("Dica: Clique num nó para ver ligações diretas.")
        
        if kpis.get('legendas'):
            st.markdown("#### 🎨 Comunidades Identificadas")
            html_legend = "<div style='background-color:#2C3E50; padding:10px; border-radius:5px; margin-bottom:15px; display:flex; flex-wrap:wrap;'>"
            lendas_ordenadas = sorted(kpis.get('legendas', []), key=lambda x: x['tamanho'], reverse=True)
            for leg in lendas_ordenadas:
                html_legend += f"<div style='margin-right:20px; margin-bottom:5px; align-items:center;'><span style='display:inline-block; width:15px; height:15px; background-color:{leg['cor']}; border-radius:50%; vertical-align:middle; margin-right:5px;'></span><b>Comunidade {leg['id']}</b> ({leg['tamanho']} nós)</div>"
            html_legend += "</div>"
            st.markdown(html_legend, unsafe_allow_html=True)

        with open(st.session_state['path_grafo'], 'r', encoding='utf-8') as f:
            components.html(f.read(), height=650, scrolling=False)
else:
    st.warning("Nenhum documento selecionado para o Grafo.")

st.markdown("---")

# === SEÇÃO 2: ANÁLISE ESTRUTURAL (RANKING SNA) ===
st.header("🏆 2. Análise Estrutural e Rankings (SNA)")
with st.form("form_tabela"):
    col_t_filt1, col_t_filt2, col_t_filt3 = st.columns(3)
    with col_t_filt1:
        niveis_sel_tabela = st.multiselect("Nível Académico:", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_tabela")
    with col_t_filt2:
        anos_sel_tabela = st.slider("Filtrar por Período (Ano):", min_ano_global, max_ano_global, (min_ano_global, max_ano_global), 1, key="ano_tab")
    with col_t_filt3:
        conceitos_contexto = st.multiselect("Filtrar Rede por Documentos que contenham os Conceitos:", options=conceitos_unicos, default=[], help="Se vazio, analisa a rede inteira. Se selecionado, calcula as métricas apenas nos documentos que possuem estas palavras-chave.")

    st.markdown("##### Configurações do Ranking")
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        n_registros_tabela = st.slider("Volume de documentos base para o cálculo:", 1, len(dados_completos), len(dados_completos), 1)
    with col_t2:
        top_x = st.number_input("Tamanho do Ranking (Top X):", min_value=1, max_value=5000, value=20, step=5)

    col_t3, col_t4, col_t5 = st.columns(3)
    categorias_disp = ["Documento", "Autor", "Orientador", "Conceito"]
    todas_metricas = ["Grau Absoluto", "Degree Centrality", "Betweenness"]
    
    with col_t3: cat_sel = st.multiselect("Categorias a exibir na tabela:", categorias_disp, default=["Orientador", "Conceito"])
    with col_t4: met_sel = st.multiselect("Métricas a exibir:", todas_metricas, default=["Grau Absoluto", "Betweenness"])
    with col_t5: met_ord = st.selectbox("Ordenar Ranking primariamente por:", met_sel if met_sel else todas_metricas)
        
    btn_render_tabela = st.form_submit_button("Processar e Atualizar Tabela", type="primary")

if btn_render_tabela:
    if met_sel and cat_sel:
        dados_tab_filtrados = []
        for d in dados_completos[:n_registros_tabela]:
            if d.get('nivel_academico', 'Outros') not in niveis_sel_tabela: continue
            ano_d = int(d.get('ano')) if d.get('ano') and str(d.get('ano')).isdigit() else None
            if not ano_d or ano_d < anos_sel_tabela[0] or ano_d > anos_sel_tabela[1]: continue
            if conceitos_contexto:
                pks_doc = set(d.get('palavras_chave', []))
                if not any(c in pks_doc for c in conceitos_contexto): continue
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
    if 'Degree Centrality' in df_exibicao.columns: df_exibicao['Degree Centrality'] = df_exibicao['Degree Centrality'].apply(lambda x: f"{x:.4f}")
    if 'Betweenness' in df_exibicao.columns: df_exibicao['Betweenness'] = df_exibicao['Betweenness'].apply(lambda x: f"{x:.4f}")
    st.dataframe(df_exibicao[colunas], use_container_width=True, hide_index=True)

st.markdown("---")

# === SEÇÃO 3: EVOLUÇÃO CRONOLÓGICA ===
st.header("📈 3. Evolução Histórica (Temporal)")
df_geral_base = preparar_dados_base_df(dados_completos)

with st.form("form_historico"):
    col_h_filt1, col_h_filt2, col_h_filt3 = st.columns(3)
    with col_h_filt1:
        niveis_sel_hist = st.multiselect("Nível Académico:", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_hist")
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
            conceitos_sel_hist = st.multiselect("Conceitos a comparar:", conceitos_unicos, default=pd.Series(lista_todos_conceitos).value_counts().head(5).index.tolist())
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
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# === SEÇÃO 4: NUVEM DE PALAVRAS ===
st.header("☁️ 4. Lexicometria e Nuvem de Palavras")
with st.form("form_nuvem"):
    col_n_filt1, col_n_filt2, col_n_filt3 = st.columns(3)
    with col_n_filt1:
        niveis_sel_nuvem = st.multiselect("Nível Académico:", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_nuvem")
    with col_n_filt2:
        orientador_sel_nuvem = st.multiselect("Orientador(es):", options=orientadores_disponiveis, default=[], help="Deixe em branco para considerar todos.")
    with col_n_filt3:
        anos_sel_nuvem = st.slider("Intervalo de Anos:", min_ano_global, max_ano_global, (min_ano_global, max_ano_global), 1, key="ano_nuvem")

    fonte_nuvem = st.radio("Base de texto:", ["Conceitos (Palavras-chave)", "Títulos dos Documentos"], horizontal=True)
    btn_render_nuvem = st.form_submit_button("Gerar Nuvem de Palavras", type="primary")

if btn_render_nuvem and not df_geral_base.empty:
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
            html_nuvem = renderizar_nuvem_interativa_html(freq_dict)
            components.html(html_nuvem, height=480, scrolling=False)

st.markdown("---")

# === SEÇÃO 5: GRAFO DE CO-OCORRÊNCIA (VOSVIEWER STYLE) ===
st.header("🔗 5. Grafo de Co-ocorrência de Palavras")
st.write("Analise como os conceitos e palavras-chave se relacionam dentro das teses e dissertações (clusters temáticos).")

with st.form("form_coocorrencia"):
    col_c_filt1, col_c_filt2, col_c_filt3 = st.columns(3)
    with col_c_filt1:
        niveis_sel_co = st.multiselect("Nível Académico:", options=niveis_disponiveis, default=niveis_disponiveis, key="niv_co")
    with col_c_filt2:
        orientador_sel_co = st.multiselect("Orientador(es):", options=orientadores_disponiveis, default=[], help="Deixe em branco para todos.")
    with col_c_filt3:
        anos_sel_co = st.slider("Intervalo de Anos:", min_ano_global, max_ano_global, (min_ano_global, max_ano_global), 1, key="ano_co")
        
    min_peso_co = st.slider("Filtro de Ruído: Mostrar apenas conexões que ocorrem juntas pelo menos X vezes:", min_value=1, max_value=20, value=2)
    
    btn_render_coocorrencia = st.form_submit_button("Gerar Grafo de Co-ocorrência", type="primary")

if btn_render_coocorrencia:
    dados_co = []
    for d in dados_completos:
        if d.get('nivel_academico', 'Outros') not in niveis_sel_co: continue
        ano_d = int(d.get('ano')) if d.get('ano') and str(d.get('ano')).isdigit() else None
        if not ano_d or ano_d < anos_sel_co[0] or ano_d > anos_sel_co[1]: continue
        if orientador_sel_co and d.get('orientador', 'Não informado') not in orientador_sel_co: continue
        dados_co.append(d)

    if not dados_co:
        st.warning("Não há documentos nos filtros selecionados para a Co-ocorrência.")
    else:
        with st.spinner("A mapear co-ocorrências..."):
            path_co, nos_co, arestas_co = gerar_html_coocorrencia(dados_co, min_coocorrencia=min_peso_co)
            st.session_state['path_co'] = path_co
            st.session_state['kpis_co'] = {'nos': nos_co, 'arestas': arestas_co}
            st.session_state['coocorrencia_pronta'] = True

if st.session_state['coocorrencia_pronta']:
    kpis_co = st.session_state['kpis_co']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conceitos Interligados", kpis_co['nos'])
    c2.metric("Conexões Formadas", kpis_co['arestas'])
    
    if kpis_co['nos'] == 0:
        st.info("O filtro de ruído está muito alto. Tente diminuir o número mínimo de co-ocorrências.")
    else:
        with open(st.session_state['path_co'], 'r', encoding='utf-8') as f:
            components.html(f.read(), height=650, scrolling=False)

st.markdown("---")

# === SEÇÃO 6: EXPORTAÇÃO DA BASE ===
st.header("📥 6. Exportação da Base de Dados Bruta")
col_b1, col_b2 = st.columns(2)
with col_b1:
    json_string = json.dumps(dados_completos, ensure_ascii=False, indent=4)
    st.download_button("📄 Baixar Base Completa (JSON Original)", file_name="base_ppgegc.json", mime="application/json", data=json_string)
with col_b2:
    csv_bytes = preparar_csv_exportacao(dados_completos)
    st.download_button("📊 Baixar Base Completa (Formato CSV)", file_name="base_ppgegc.csv", mime="text/csv", data=csv_bytes)
