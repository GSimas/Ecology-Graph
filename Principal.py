import streamlit as st
import streamlit.components.v1 as components
from streamlit_agraph import agraph, Node, Edge, Config
import networkx as nx
import networkx.algorithms.community as nx_comm
import pandas as pd
import plotly.express as px
import re
import unicodedata
from collections import Counter
from streamlit_agraph import Node, Edge
import time

from app_config import get_gemini_api_key
from backend import (
    conectar_neo4j, 
    extrair_subgrafo_neo4j,
    navegar_para,
    gerar_tabela_entidades_por_macrotema,
    calcular_sna_global,
    gerar_orbita_neo4j,
    gerar_tabela_macrotemas_perfil,
    renderizar_nuvem_interativa_html,
    gerar_descritivo_sessao,
    carregar_catalogo_capes_ufsc,
    carregar_base_consolidada,
    carregar_catalogo_programas,
    plotar_mapa_tematico,
    calcular_similares_rede,
    plotar_grafico_3d_sna
)

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Ecologia do Conhecimento UFSC", page_icon="🌌", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; }
    [data-testid="stMetricValue"] { font-size: 2rem !important; color: #F39C12 !important; }
    [data-testid="stMetricLabel"] { font-size: 1rem !important; color: #BDC3C7 !important; font-weight: bold; }
    div[data-testid="metric-container"] { background-color: #2C3E50; padding: 15px; border-radius: 12px; border-left: 5px solid #F39C12; }
    h1, h2, h3, h4, h5 { color: #F39C12; font-family: 'Helvetica Neue', sans-serif; }
    button[kind="primary"] { background-color: #2ECC71 !important; color: white !important; font-weight: bold !important; border: none !important; }
    </style>
""", unsafe_allow_html=True)

# --- INICIALIZAÇÃO DE ESTADO ---
if 'busca_tipo' not in st.session_state: 
    st.session_state.update({'busca_tipo': "Documento", 'busca_termo': None})
if 'macrotemas_computados' not in st.session_state:
    st.session_state['macrotemas_computados'] = False

# --- TELA DE SELEÇÃO INICIAL (CARREGAMENTO PREGUIÇOSO / LAZY LOADING) ---
if 'dados_completos' not in st.session_state or st.session_state.get('recarregar'):
    st.title("🔌 Seleção de Programas (PPGs)")
    st.markdown("Selecione os programas que deseja analisar para carregar a base consolidada de publicações.")
    
    catalogo_leve = carregar_catalogo_programas()
    programas_disponiveis = sorted(list(catalogo_leve.keys()))
    
    programas_selecionados = st.multiselect("Selecione um ou mais Programas de Pós-Graduação:", programas_disponiveis)
    
    if st.button("Carregar Dados e Iniciar Análise", type="primary"):
        if programas_selecionados:
            with st.spinner("Lendo a base consolidada e filtrando os PPGs selecionados. Isso levará apenas alguns segundos..."):
                try:
                    base_total = carregar_base_consolidada()
                    dados_filtrados = [d for d in base_total if d.get('programa_origem') in programas_selecionados]
                    
                    if not dados_filtrados:
                        st.warning("Nenhum documento encontrado para os PPGs selecionados.")
                        st.stop()
                        
                    st.session_state['dados_completos'] = dados_filtrados
                    st.session_state['programas_selecionados_lista'] = programas_selecionados
                    st.session_state['nome_programa'] = f"{len(programas_selecionados)} PPG(s) Selecionado(s): {', '.join(programas_selecionados)}"
                    st.session_state['macrotemas_computados'] = True
                    st.session_state['recarregar'] = False
                    st.rerun()
                    
                except FileNotFoundError:
                    st.error("O arquivo 'base_consolidada_ufsc.json.gz' não foi encontrado.")
                    st.stop()
        else:
            st.warning("Por favor, selecione pelo menos um programa para continuar.")

    # =====================================================================
    # PANORAMA GLOBAL DA UFSC (CAPES)
    # =====================================================================
    st.markdown("---")
    st.markdown("## 🏛️ Panorama Global da Pós-Graduação (UFSC)")
    st.markdown("Visão macro do ecossistema institucional com base nos dados oficiais da Plataforma Sucupira (CAPES).")
    
    with st.spinner("Carregando catálogo da CAPES..."):
        catalogo_capes = carregar_catalogo_capes_ufsc()
        
    if catalogo_capes:
        # 1. Prepara o DataFrame base
        df_capes = pd.DataFrame.from_dict(catalogo_capes, orient='index')
        
        # Filtra apenas programas que estão operacionais
        if 'Situação' in df_capes.columns:
            df_capes = df_capes[df_capes['Situação'].str.contains('FUNCIONAMENTO|ATIVO', case=False, na=True)]
            
        # 2. Extrai as opções únicas para os filtros
        op_niveis = ["Mestrado", "Doutorado"] 
        op_modalidades = sorted([str(x) for x in df_capes['Modalidade'].unique() if x and x != 'Não informado'])
        op_g_areas = sorted([str(x) for x in df_capes['Grande Área'].unique() if x and x != 'Não informado'])
        op_areas = sorted([str(x) for x in df_capes['Área de Conhecimento'].unique() if x and x != 'Não informado'])
        
        # Extração das notas únicas (ordenadas da maior para a menor)
        op_notas = sorted([str(x) for x in df_capes['Nota'].unique() if x and x != 'Não informado'], reverse=True)
        
        # 3. Desenha a barra de filtros (Ajustado para 5 colunas ou 3+2 para melhor leitura)
        st.markdown("#### 🔍 Filtros Dinâmicos")
        
        # Primeira linha de filtros
        cf1, cf2, cf3 = st.columns(3)
        with cf1:
            f_nivel = st.multiselect("Nível (Grau Acadêmico):", options=op_niveis, default=op_niveis)
        with cf2:
            f_mod = st.multiselect("Modalidade:", options=op_modalidades, default=op_modalidades)
        with cf3:
            f_nota = st.multiselect("Nota CAPES:", options=op_notas, default=op_notas)
            
        # Segunda linha de filtros
        cf4, cf5 = st.columns(2)
        with cf4:
            f_garea = st.multiselect("Grande Área:", options=op_g_areas, default=op_g_areas)
        with cf5:
            f_area = st.multiselect("Área de Conhecimento:", options=op_areas, default=op_areas)
            
        # 4. Aplica os filtros ao DataFrame
        df_filtrado = df_capes.copy()
        
        # Lógica especial para o Nível (Contains)
        if f_nivel:
            mask_nivel = pd.Series(False, index=df_filtrado.index)
            if "Mestrado" in f_nivel:
                mask_nivel = mask_nivel | df_filtrado['Grau Acadêmico'].str.contains('Mestrado', case=False, na=False)
            if "Doutorado" in f_nivel:
                mask_nivel = mask_nivel | df_filtrado['Grau Acadêmico'].str.contains('Doutorado', case=False, na=False)
            df_filtrado = df_filtrado[mask_nivel]
            
        # Filtros por correspondência exata
        if f_mod: df_filtrado = df_filtrado[df_filtrado['Modalidade'].isin(f_mod)]
        if f_nota: df_filtrado = df_filtrado[df_filtrado['Nota'].isin(f_nota)]
        if f_garea: df_filtrado = df_filtrado[df_filtrado['Grande Área'].isin(f_garea)]
        if f_area: df_filtrado = df_filtrado[df_filtrado['Área de Conhecimento'].isin(f_area)]
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if df_filtrado.empty:
            st.warning("Nenhum programa encontrado com essa combinação de filtros.")
        else:
            # 5. Calcula KPIs com o DataFrame Filtrado
            total_programas = len(df_filtrado)
            df_filtrado['Nota_Num'] = pd.to_numeric(df_filtrado['Nota'], errors='coerce').fillna(0)
            excelencia = len(df_filtrado[df_filtrado['Nota_Num'] >= 6])
            
            academicos = len(df_filtrado[df_filtrado['Modalidade'].str.contains('Acad', case=False, na=False)])
            profissionais = len(df_filtrado[df_filtrado['Modalidade'].str.contains('Prof', case=False, na=False)])
            
            # Exibe KPIs
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total de Programas (PPGs)", total_programas)
            c2.metric("Programas de Excelência (Nota 6/7)", excelencia)
            c3.metric("Modalidade Acadêmica", academicos)
            c4.metric("Modalidade Profissional", profissionais)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 6. Desenha Gráficos Interativos (Baseados no filtro)
            col_graf1, col_graf2 = st.columns([2, 1])
            
            with col_graf1:
                st.markdown("#### Distribuição por Grande Área do Conhecimento")
                df_areas = df_filtrado['Grande Área'].value_counts().reset_index()
                df_areas.columns = ['Grande Área', 'Quantidade']
                
                fig_areas = px.bar(
                    df_areas, 
                    y='Grande Área', 
                    x='Quantidade', 
                    orientation='h',
                    text='Quantidade',
                    color='Grande Área'
                )
                fig_areas.update_layout(
                    showlegend=False, 
                    yaxis={'categoryorder':'total ascending'},
                    xaxis_title="Número de Programas",
                    yaxis_title="",
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig_areas, use_container_width=True, theme="streamlit")
                
            with col_graf2:
                st.markdown("#### Conceito CAPES (Notas)")
                df_notas = df_filtrado[df_filtrado['Nota_Num'] > 0]['Nota'].astype(str).value_counts().reset_index()
                df_notas.columns = ['Nota CAPES', 'Quantidade']
                
                if not df_notas.empty:
                    fig_notas = px.pie(
                        df_notas, 
                        names='Nota CAPES', 
                        values='Quantidade',
                        hole=0.4,
                        color='Nota CAPES',
                        color_discrete_map={'7': '#2ECC71', '6': '#27AE60', '5': '#3498DB', '4': '#F1C40F', '3': '#E67E22'}
                    )
                    fig_notas.update_traces(textposition='inside', textinfo='percent+label')
                    fig_notas.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
                    st.plotly_chart(fig_notas, use_container_width=True, theme="streamlit")
                else:
                    st.info("Nenhuma nota numérica registrada para os filtros atuais.")
                
            # Tabela Geral de Consulta Rápida
            with st.expander("Tabela Completa de Programas Filtrados"):
                df_exibicao = df_filtrado[['Nome', 'Código', 'Grande Área', 'Nota', 'Modalidade', 'Grau Acadêmico']].sort_values(by='Nome')
                st.dataframe(df_exibicao, use_container_width=True, hide_index=True)

    else:
        st.info("Nenhum dado da CAPES foi carregado no momento.")

    # Trava a execução do resto do app (SNA, gráficos, etc.) até o usuário passar desta tela
    st.stop()

# --- DASHBOARD PRINCIPAL ---
dados_completos = st.session_state['dados_completos']
api_key_app = get_gemini_api_key()
st.title("🌌 Ecologia do Conhecimento")
st.subheader(f"Base: {st.session_state['nome_programa']}")

# Botão na barra lateral para voltar e escolher outros PPGs
if st.sidebar.button("🔄 Escolher outros PPGs", type="primary"):
    st.session_state['recarregar'] = True
    st.rerun()


# KPIs Básicos
autores_set = set([a for d in dados_completos for a in d.get('autores', [])])
orientadores_set = set([d.get('orientador') for d in dados_completos if d.get('orientador')])
coorientadores_set = set([co for d in dados_completos for co in d.get('co_orientadores', [])])
keywords_set = set([kw for d in dados_completos for kw in d.get('palavras_chave', [])])

c1, c2, c3 = st.columns(3)
c1.metric("📄 Documentos Totais", len(dados_completos))
c2.metric("🎓 Teses (Doutorado)", len([d for d in dados_completos if "Tese" in d.get('nivel_academico', '')]))
c3.metric("📜 Dissertações", len([d for d in dados_completos if "Disserta" in d.get('nivel_academico', '')]))

c4, c5, c6, c7 = st.columns(4)
c4.metric("✍️ Autores Únicos", len(autores_set))
c5.metric("🏫 Orientadores", len(orientadores_set))
c6.metric("🤝 Co-orientadores", len(coorientadores_set))
c7.metric("💡 Conceitos (Keywords)", len(keywords_set))


# --- APRESENTAÇÃO DO PERFIL DINÂMICO ---
if api_key_app:
    # Extrai até 25 documentos espaçados uniformemente para criar uma amostra representativa
    amostra_docs = []
    salto = max(1, len(dados_completos) // 25)
    for i in range(0, len(dados_completos), salto):
        d = dados_completos[i]
        amostra_docs.append(f"- {d.get('titulo', '')} | {', '.join(d.get('palavras_chave', []))}")
        if len(amostra_docs) >= 25: break
            
    nomes_ppgs = list(set([d.get('programa_origem', 'Programa Desconhecido') for d in dados_completos if d.get('programa_origem')]))
else:
    st.warning("🔑 Chave da API do Gemini não configurada em variáveis de ambiente ou secrets. O perfil dinâmico está desativado.")
st.markdown("<br>", unsafe_allow_html=True)



if len(st.session_state.get('programas_selecionados_lista', [])) > 1:
    st.markdown("---")
    st.subheader("📊 Comparativo entre PPGs")
    
    comparativo_data = []
    
    from collections import defaultdict
    dados_por_ppg = defaultdict(list)
    for d in dados_completos:
        ppg = d.get('programa_origem', 'Desconhecido')
        dados_por_ppg[ppg].append(d)
        
    for ppg, docs_ppg in dados_por_ppg.items():
        comparativo_data.append({
            "PPG": ppg,
            "📄 Documentos Totais": len(docs_ppg),
            "🎓 Teses (Doutorado)": len([d for d in docs_ppg if "Tese" in d.get('nivel_academico', '')]),
            "📜 Dissertações": len([d for d in docs_ppg if "Disserta" in d.get('nivel_academico', '')]),
            "✍️ Autores Únicos": len(set([a for d in docs_ppg for a in d.get('autores', [])])),
            "🏫 Orientadores": len(set([d.get('orientador') for d in docs_ppg if d.get('orientador')])),
            "🤝 Co-orientadores": len(set([co for d in docs_ppg for co in d.get('co_orientadores', [])])),
            "💡 Conceitos (Keywords)": len(set([kw for d in docs_ppg for kw in d.get('palavras_chave', [])]))
        })
        
    df_comp = pd.DataFrame(comparativo_data)
    df_melted = df_comp.melt(id_vars="PPG", var_name="Métrica", value_name="Quantidade")
    
    fig_comp = px.bar(
        df_melted, 
        x="PPG", 
        y="Quantidade", 
        color="PPG", 
        facet_col="Métrica", 
        facet_col_wrap=4, 
        template="plotly_dark", 
        text="Quantidade",
        height=650
    )
    fig_comp.update_yaxes(matches=None, showticklabels=False, title="") 
    fig_comp.update_xaxes(showticklabels=False, title="")
    fig_comp.update_traces(textposition='outside')
    fig_comp.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    
    st.plotly_chart(fig_comp, use_container_width=True)


# --- APRESENTAÇÃO DO PERFIL E DADOS OFICIAIS ---
st.markdown("#### 🏛️ Ficha Técnica e Perfil Institucional")

nomes_ppgs = list(set([d.get('programa_origem', 'Programa Desconhecido') for d in dados_completos if d.get('programa_origem')]))

# Puxa o dicionário completo da CAPES da memória (Instantâneo)
catalogo_capes = carregar_catalogo_capes_ufsc()

for nome_ppg in nomes_ppgs:
    # 1. Limpeza agressiva do nome do repositório
    nome_limpo = nome_ppg.replace("Programa de Pós-Graduação em ", "").replace("Programa de Pós-Graduação ", "").replace("PPG em ", "").strip().upper()
    nome_norm_busca = ''.join(c for c in unicodedata.normalize('NFD', nome_limpo) if unicodedata.category(c) != 'Mn')
    
    # 2. Tenta encontrar correspondência exata primeiro (o mais rápido)
    dados_capes = catalogo_capes.get(nome_norm_busca)
    
    # 3. Inteligência de Strings (Fuzzy Matching) para lidar com variações da UFSC/CAPES
    if not dados_capes:
        import difflib # Biblioteca nativa do Python para cálculo de similaridade
        
        chaves_disponiveis = list(catalogo_capes.keys())
        # Procura a chave mais parecida com pelo menos 65% de similaridade estrutural
        melhores_matches = difflib.get_close_matches(nome_norm_busca, chaves_disponiveis, n=1, cutoff=0.65)
        
        if melhores_matches:
            dados_capes = catalogo_capes[melhores_matches[0]]
        else:
            # 4. Fallback Final: Interseção de Palavras-Chave (Ignorando preposições curtas)
            palavras_busca = set([p for p in nome_norm_busca.split() if len(p) > 2])
            for key_capes, dados in catalogo_capes.items():
                palavras_capes = set([p for p in key_capes.split() if len(p) > 2])
                
                # Se houver um cruzamento muito forte das palavras principais (ex: ENGENHARIA, GESTAO, CONHECIMENTO)
                if len(palavras_busca.intersection(palavras_capes)) >= max(1, len(palavras_busca) - 1):
                    dados_capes = dados
                    break

    # 5. Desenha o Cartão Oficial da CAPES
    if dados_capes:
        st.markdown(f"**{dados_capes['Nome']} ({dados_capes['Código']}) | Nota CAPES: {dados_capes['Nota']}**")

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Grande Área:** {dados_capes['Grande Área']}")
            st.write(f"**Área de Avaliação:** {dados_capes['Área de Avaliação']}")
            st.write(f"**Área de Conhecimento:** {dados_capes['Área de Conhecimento']}")
        with col2:
            st.write(f"**Modalidade:** {dados_capes['Modalidade']} ({dados_capes['Grau Acadêmico']})")
            st.write(f"**Situação:** {dados_capes['Situação']}")
            st.write(f"**Ensino:** {dados_capes['Modalidade de Ensino']}")
    else:
        st.markdown(f"**{nome_ppg}**")
        st.caption("⚠️ Dados oficiais não localizados na base da CAPES (Possível variação de nomenclatura institucional).")
        
st.markdown("<br>", unsafe_allow_html=True)

# 4. Descritivo Dinâmico da IA (A Alma do Programa)
if api_key_app:
    amostra_docs = []
    salto = max(1, len(dados_completos) // 25)
    for i in range(0, len(dados_completos), salto):
        d = dados_completos[i]
        amostra_docs.append(f"- {d.get('titulo', '')} | {', '.join(d.get('palavras_chave', []))}")
        if len(amostra_docs) >= 25: break
            
    with st.spinner("A IA está analisando a amostra de documentos para sintetizar o perfil epistemológico..."):
        descritivo_dinamico = gerar_descritivo_sessao(tuple(nomes_ppgs), "\n".join(amostra_docs), api_key_app)
        st.info(f"**Síntese de Pesquisa do PPG:** {descritivo_dinamico}")
else:
    st.warning("🔑 Chave da API do Gemini não configurada em variáveis de ambiente ou secrets.")
st.markdown("---")


# IMPORTANTE: Movemos o cálculo SNA para cá para alimentar a nova Super-Tabela
sna_global = calcular_sna_global(dados_completos)

# =========================================================================
# 🏆 DESTAQUES DO ECOSSISTEMA (RANKINGS DE REDE)
# =========================================================================
st.markdown("#### 🏆 Destaques do Ecossistema")

# --- LÓGICA DE CÁLCULO ---
# 1. Genealogia Acadêmica (Alunos que viraram professores)
professores_ativos = orientadores_set.union(coorientadores_set)
formadores = set()
for d in dados_completos:
    for autor in d.get('autores', []):
        if autor in professores_ativos and d.get('orientador'):
            formadores.add(d.get('orientador'))

# 2. Volumes
ori_counts = Counter([d.get('orientador') for d in dados_completos if d.get('orientador')])
top_ori_vol = ori_counts.most_common(1)[0] if ori_counts else ("Nenhum", 0)

coori_counts = Counter([co for d in dados_completos for co in d.get('co_orientadores', [])])
top_coori_vol = coori_counts.most_common(1)[0] if coori_counts else ("Nenhum", 0)

# Função auxiliar para pegar o topo da rede
def get_top_sna(entidades, metrica):
    if not entidades: return "Nenhum", 0.0
    valid_nodes = {k: v for k, v in sna_global.items() if k in entidades}
    if not valid_nodes: return "Nenhum", 0.0
    top_node = max(valid_nodes.items(), key=lambda x: x[1].get(metrica, 0))
    return top_node[0], top_node[1].get(metrica, 0)

# SNA Orientadores e Coorientadores
top_ori_close = get_top_sna(orientadores_set, 'Closeness')
top_ori_bet = get_top_sna(orientadores_set, 'Betweenness')
top_coori_close = get_top_sna(coorientadores_set, 'Closeness')
top_coori_bet = get_top_sna(coorientadores_set, 'Betweenness')

# SNA Teses e Dissertações
teses_titulos = [d.get('titulo') for d in dados_completos if 'Tese' in d.get('nivel_academico', '')]
dissertacoes_titulos = [d.get('titulo') for d in dados_completos if 'Disserta' in d.get('nivel_academico', '')]

top_tese_close = get_top_sna(teses_titulos, 'Closeness')
top_tese_bet = get_top_sna(teses_titulos, 'Betweenness')
top_diss_close = get_top_sna(dissertacoes_titulos, 'Closeness')
top_diss_bet = get_top_sna(dissertacoes_titulos, 'Betweenness')

# --- RENDERIZAÇÃO EM ABAS COM BOTÕES CLICÁVEIS ---
tab_dest1, tab_dest2, tab_dest3 = st.tabs(["🎓 Volumes e Genealogia", "🌉 Intermediação (Betweenness)", "🎯 Proximidade (Closeness)"])

with tab_dest1:
    col_vol1, col_vol2 = st.columns(2)
    with col_vol1:
        st.markdown("**Orientador com maior número de orientações:**")
        if top_ori_vol[0] != "Nenhum":
            st.button(f"🏫 {top_ori_vol[0]} ({top_ori_vol[1]} orientações)", key="btn_top_ori_vol", on_click=navegar_para, args=("Orientador", top_ori_vol[0]))
        
        st.markdown("**Coorientador com maior número de coorientações:**")
        if top_coori_vol[0] != "Nenhum":
            st.button(f"🤝 {top_coori_vol[0]} ({top_coori_vol[1]} coorientações)", key="btn_top_coori_vol", on_click=navegar_para, args=("Co-orientador", top_coori_vol[0]))
            
    with col_vol2:
        st.markdown("**🌱 Formadores de Professores:**")
        st.caption("Orientadores que formaram alunos que hoje também orientam na rede.")
        if formadores:
            with st.expander(f"Ver lista de formadores ({len(formadores)})"):
                for i, formador in enumerate(sorted(list(formadores))):
                    st.button(f"🎓 {formador}", key=f"btn_formador_{i}", on_click=navegar_para, args=("Orientador", formador))
        else:
            st.info("Nenhum ciclo genealógico detectado nesta amostra.")

with tab_dest2:
    st.info("**O que é Betweenness (Intermediação)?** Mede quantas vezes um nó atua como 'ponte' no caminho mais curto entre outros nós. Ter alto Betweenness significa ser o elo que conecta bolhas de conhecimento diferentes (alta interdisciplinaridade) e controla o fluxo de informação no programa.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Orientador (Maior Betweenness):**")
        if top_ori_bet[0] != "Nenhum": 
            st.button(f"🏫 {top_ori_bet[0]} (Score: {top_ori_bet[1]:.4f})", key="btn_top_ori_bet", on_click=navegar_para, args=("Orientador", top_ori_bet[0]))
        
        st.markdown("**Coorientador (Maior Betweenness):**")
        if top_coori_bet[0] != "Nenhum": 
            st.button(f"🤝 {top_coori_bet[0]} (Score: {top_coori_bet[1]:.4f})", key="btn_top_co_bet", on_click=navegar_para, args=("Co-orientador", top_coori_bet[0]))
    with c2:
        st.markdown("**Tese Interdisciplinar (Maior Betweenness):**")
        if top_tese_bet[0] != "Nenhum": 
            st.button(f"📄 {top_tese_bet[0][:60]}... (Score: {top_tese_bet[1]:.4f})", key="btn_top_t_bet", on_click=navegar_para, args=("Documento", top_tese_bet[0]))
        
        st.markdown("**Dissertação Interdisciplinar (Maior Betweenness):**")
        if top_diss_bet[0] != "Nenhum": 
            st.button(f"📄 {top_diss_bet[0][:60]}... (Score: {top_diss_bet[1]:.4f})", key="btn_top_d_bet", on_click=navegar_para, args=("Documento", top_diss_bet[0]))

with tab_dest3:
    st.info("**O que é Closeness (Proximidade)?** Mede a distância média de um nó para todos os outros da rede. Ter alto Closeness significa estar no 'centro nervoso' do ecossistema, capaz de acessar ou disseminar o conhecimento mais rapidamente, com menos saltos e intermediários.")
    
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Orientador Mais Central (Maior Closeness):**")
        if top_ori_close[0] != "Nenhum": 
            st.button(f"🏫 {top_ori_close[0]} (Score: {top_ori_close[1]:.4f})", key="btn_top_ori_close", on_click=navegar_para, args=("Orientador", top_ori_close[0]))
        
        st.markdown("**Coorientador Mais Central (Maior Closeness):**")
        if top_coori_close[0] != "Nenhum": 
            st.button(f"🤝 {top_coori_close[0]} (Score: {top_coori_close[1]:.4f})", key="btn_top_co_close", on_click=navegar_para, args=("Co-orientador", top_coori_close[0]))
    with c4:
        st.markdown("**Tese Central (Maior Closeness):**")
        if top_tese_close[0] != "Nenhum": 
            st.button(f"📄 {top_tese_close[0][:60]}... (Score: {top_tese_close[1]:.4f})", key="btn_top_t_close", on_click=navegar_para, args=("Documento", top_tese_close[0]))
        
        st.markdown("**Dissertação Central (Maior Closeness):**")
        if top_diss_close[0] != "Nenhum": 
            st.button(f"📄 {top_diss_close[0][:60]}... (Score: {top_diss_close[1]:.4f})", key="btn_top_d_close", on_click=navegar_para, args=("Documento", top_diss_close[0]))

st.markdown("---")
# (Módulo de visualização de Macrotemas movido para a secção inferior)


# --- MOTOR DE BUSCA (EGO-GRAPH) ---
st.header("🔍 Motor de Busca e Dossiê")

opcoes_busca = ["Documento", "Autor", "Orientador", "Co-orientador", "Palavra-chave"]
if st.session_state['macrotemas_computados']:
    opcoes_busca.append("Macrotema")

tipo_busca = st.radio("Procurar por Entidade:", opcoes_busca, horizontal=True, key="busca_tipo")

if tipo_busca == "Documento": opcoes = [d['titulo'] for d in dados_completos]
elif tipo_busca == "Autor": opcoes = list(autores_set)
elif tipo_busca == "Orientador": opcoes = list(orientadores_set)
elif tipo_busca == "Co-orientador": opcoes = list(coorientadores_set)
elif tipo_busca == "Palavra-chave": opcoes = list(keywords_set)
elif tipo_busca == "Macrotema": opcoes = list(set([d.get('macrotema') for d in dados_completos if d.get('macrotema')]))

if st.session_state['busca_termo'] not in opcoes: st.session_state['busca_termo'] = None
termo_selecionado = st.selectbox("Selecione:", sorted(opcoes), index=sorted(opcoes).index(st.session_state['busca_termo']) if st.session_state['busca_termo'] in opcoes else None, placeholder="Pesquise aqui...")

if termo_selecionado != st.session_state['busca_termo']:
    st.session_state['busca_termo'] = termo_selecionado
    st.rerun()

termo_ativo = st.session_state['busca_termo']

if termo_ativo:
    col_info, col_sna = st.columns([2, 1])
    with col_info:
        st.info(f"**{termo_ativo}**")
        
        if tipo_busca == "Documento":
            doc = next((d for d in dados_completos if d['titulo'] == termo_ativo), {})
            
            st.write(f"**Ano:** {doc.get('ano', 'N/A')} | **Nível:** {doc.get('nivel_academico', 'N/A')} | **Programa:** {doc.get('programa_origem', 'N/A')}")
            
            if doc.get('url'):
                st.markdown(f"🔗 **Link Oficial na UFSC:** [{doc['url']}]({doc['url']})")
                
            if st.session_state['macrotemas_computados']:
                tema = doc.get('macrotema', 'Multidisciplinar / Transversal')
                st.write("**Macrotema Classificado:**")
                st.button(f"🏷️ {tema}", on_click=navegar_para, args=("Macrotema", tema))
                
            st.write("**Rede de Autoria e Orientação:**")
            for a in doc.get('autores', []): 
                st.button(f"👤 {a}", on_click=navegar_para, args=("Autor", a))
            if doc.get('orientador'): 
                st.button(f"🏫 {doc['orientador']}", on_click=navegar_para, args=("Orientador", doc['orientador']))
            for co in doc.get('co_orientadores', []):
                st.button(f"🤝 {co}", on_click=navegar_para, args=("Co-orientador", co))
                
            st.write("**Palavras-chave:**")
            for pk in doc.get('palavras_chave', []): 
                st.button(f"💡 {pk}", on_click=navegar_para, args=("Palavra-chave", pk))
                
            with st.expander("Ler Resumo (Abstract)"): 
                st.write(doc.get('resumo', 'Resumo não disponível.'))
                
        elif tipo_busca == "Autor":
            docs = [d for d in dados_completos if termo_ativo in d.get('autores', [])]
            
            programas = sorted(list(set([d.get('programa_origem') for d in docs if d.get('programa_origem')])))
            if programas:
                st.write(f"**🏛️ Programas (PPG):** {', '.join(programas)}")
            
            orientadores = set()
            co_orientadores = set()
            for d in docs:
                if d.get('orientador'):
                    orientadores.add(d['orientador'])
                for co in d.get('co_orientadores', []):
                    co_orientadores.add(co)
            
            if orientadores or co_orientadores:
                st.write("**👨‍🏫 Orientadores e Co-orientadores:**")
                for ori in sorted(list(orientadores)):
                    st.button(f"🏫 Orientador: {ori}", key=f"btn_ori_aut_{abs(hash(ori))}", on_click=navegar_para, args=("Orientador", ori))
                for co in sorted(list(co_orientadores)):
                    st.button(f"🤝 Co-orientador: {co}", key=f"btn_co_aut_{abs(hash(co))}", on_click=navegar_para, args=("Co-orientador", co))
                
                st.markdown("<br>", unsafe_allow_html=True)

            st.write(f"**Documentos Escritos ({len(docs)}):**")
            for i, d in enumerate(docs): st.button(f"📄 {d['titulo']}", key=f"btn_aut_{i}", on_click=navegar_para, args=("Documento", d['titulo']))
            
        elif tipo_busca in ["Orientador", "Co-orientador"]:
            if tipo_busca == "Orientador":
                docs = [d for d in dados_completos if d.get('orientador') == termo_ativo]
            else:
                docs = [d for d in dados_completos if termo_ativo in d.get('co_orientadores', [])]
                
            programas = sorted(list(set([d.get('programa_origem') for d in docs if d.get('programa_origem')])))
            if programas:
                st.write(f"**🏛️ Programas (PPG):** {', '.join(programas)}")
                
            gerar_tabela_macrotemas_perfil(docs, dados_completos)
            
            # --- 1. REDE DE PARCERIAS (CO-ORIENTAÇÕES) ---
            parcerias_dados = {}
            for d in docs:
                ori = d.get('orientador')
                cooris = d.get('co_orientadores', [])
                nivel = d.get('nivel_academico', 'Outros')
                titulo = d.get('titulo', 'Sem Título')
                
                eh_tese = 'Tese' in nivel
                eh_dissertacao = 'Disserta' in nivel
                
                parceiros_doc = []
                if tipo_busca == "Orientador":
                    parceiros_doc.extend(cooris)
                else: 
                    # Se o foco for Co-orientador, os parceiros são o Orientador e os outros Co-orientadores
                    if ori: parceiros_doc.append(ori)
                    parceiros_doc.extend([c for c in cooris if c != termo_ativo])
                    
                for p in parceiros_doc:
                    if p not in parcerias_dados:
                        parcerias_dados[p] = {'Total': 0, 'Teses': 0, 'Dissertações': 0, 'Trabalhos': []}
                    
                    parcerias_dados[p]['Total'] += 1
                    if eh_tese:
                        parcerias_dados[p]['Teses'] += 1
                    elif eh_dissertacao:
                        parcerias_dados[p]['Dissertações'] += 1
                        
                    # Guardamos o título formatado
                    prefixo = "[T]" if eh_tese else "[D]" if eh_dissertacao else "[O]"
                    parcerias_dados[p]['Trabalhos'].append(f"{prefixo} {titulo}")
                    
            if parcerias_dados:
                linhas_parcerias = []
                for parceiro, info in parcerias_dados.items():
                    linhas_parcerias.append({
                        "Parceiro(a)": parceiro,
                        "Total": info['Total'],
                        "Teses": info['Teses'],
                        "Dissertações": info['Dissertações'],
                        "Documentos em Conjunto": " | ".join(info['Trabalhos'])
                    })
                    
                df_parceiros = pd.DataFrame(linhas_parcerias).sort_values(by="Total", ascending=False)
                
                st.write("**🤝 Principais Parcerias (Coorientação):**")
                st.dataframe(
                    df_parceiros, 
                    use_container_width=True, 
                    hide_index=True
                )
                st.caption("*Legenda dos documentos: [T] = Tese, [D] = Dissertação, [O] = Outros.*")
                st.markdown("<br>", unsafe_allow_html=True)
            
            # --- 2. GENEALOGIA ACADÊMICA (ALUNOS -> PROFESSORES) ---
            alunos_orientados = set()
            for d in docs:
                for autor in d.get('autores', []):
                    alunos_orientados.add(autor)
            alunos_orientados = sorted(list(alunos_orientados))
            
            # Cruzamos os alunos desse orientador com a lista global de orientadores/co-orientadores da UFSC
            professores_globais = orientadores_set.union(coorientadores_set)
            alunos_professores = [aluno for aluno in alunos_orientados if aluno in professores_globais]
            
            if alunos_professores:
                st.success(f"🌱 **Genealogia Acadêmica:** {len(alunos_professores)} aluno(s) orientado(s) por este docente tornou-se professor(a)/orientador(a) na rede!")
                with st.expander(f"Ver linhagem ({len(alunos_professores)} acadêmicos)"):
                    for i, aluno in enumerate(alunos_professores):
                         # Roteamento inteligente: manda pro perfil de Orientador ou Co-orientador dependendo de onde ele atua
                         tipo_nav = "Orientador" if aluno in orientadores_set else "Co-orientador"
                         chave_nav = f"btn_descendente_{abs(hash(aluno))}_{i}"
                         st.button(f"🎓 {aluno} (Ver Perfil Acadêmico)", key=chave_nav, on_click=navegar_para, args=(tipo_nav, aluno))
                st.markdown("<br>", unsafe_allow_html=True)

            # --- 3. LISTA GERAL DE ALUNOS ---
            st.write(f"**🎓 Total de Alunos {'Orientados' if tipo_busca == 'Orientador' else 'Co-orientados'} ({len(alunos_orientados)}):**")
            with st.expander(f"Ver lista de todos os {len(alunos_orientados)} alunos"):
                for i, aluno in enumerate(alunos_orientados):
                    st.button(f"👤 {aluno}", key=f"btn_aluno_{tipo_busca}_{abs(hash(aluno))}_{i}", on_click=navegar_para, args=("Autor", aluno))
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- 4. LISTA DE DOCUMENTOS ---
            st.write(f"**Documentos {'Orientados' if tipo_busca == 'Orientador' else 'Co-orientados'} ({len(docs)}):**")
            
            from collections import defaultdict
            docs_por_mt = defaultdict(list)
            for d in docs:
                docs_por_mt[d.get('macrotema', 'Multidisciplinar / Transversal')].append(d)
                
            with st.expander(f"📚 Ver lista de {len(docs)} documentos"):
                for mt, docs_mt in docs_por_mt.items():
                    st.markdown(f"**🏷️ {mt}**")
                    for i, d in enumerate(docs_mt):
                        chave_unica = f"btn_{tipo_busca}_{abs(hash(d['titulo']))}_{i}"
                        st.button(f"📄 {d['titulo']}", key=chave_unica, on_click=navegar_para, args=("Documento", d['titulo']))
                        
        elif tipo_busca == "Palavra-chave":
            docs = [d for d in dados_completos if termo_ativo in d.get('palavras_chave', [])]
            gerar_tabela_macrotemas_perfil(docs, dados_completos)
            
            with st.expander(f"📚 Ver Lista Completa de Documentos Associados ({len(docs)})"):
                for i, d in enumerate(docs): 
                    chave_unica = f"btn_pk_{abs(hash(d['titulo']))}_{i}"
                    st.button(f"📄 {d['titulo']}", key=chave_unica, on_click=navegar_para, args=("Documento", d['titulo']))
            
        elif tipo_busca == "Macrotema":
            docs = [d for d in dados_completos if d.get('macrotema') == termo_ativo]
            gerar_tabela_entidades_por_macrotema(docs, dados_completos)
            
            with st.expander(f"📚 Explorar Teses e Dissertações da Categoria ({len(docs)})"):
                for i, d in enumerate(docs): 
                    chave_unica = f"btn_mt_{abs(hash(d['titulo']))}_{i}"
                    st.button(f"📄 {d['titulo']}", key=chave_unica, on_click=navegar_para, args=("Documento", d['titulo']))
                    
    with col_sna:
        metricas = sna_global.get(termo_ativo, {})
        if metricas:
            st.success(f"Cluster: {metricas.get('Comunidade')} | Rank: #{metricas.get('Ranking Global')}")
            st.metric("Grau (Conexões)", metricas.get('Grau Absoluto'))
            st.metric("Betweenness", f"{metricas.get('Betweenness', 0):.4f}")
            st.metric("Closeness", f"{metricas.get('Closeness', 0):.4f}") 
            
    st.markdown("---")
    
    # =========================================================
    # ABAS DO DOSSIÊ (HISTÓRICO, NUVEM, ÓRBITA E SIMILARES)
    # =========================================================
    tab_hist, tab_nuvem, tab_orbita, tab_similares = st.tabs([
        "📈 Evolução Histórica", 
        "☁️ Nuvem de Palavras", 
        "🌌 Órbita de Relacionamentos",
        "🔗 Itens Semelhantes"
    ])

    # Verifica se a entidade buscada possui uma lista de documentos atrelada a ela
    # (Adicionei "Autor" aqui também, pois é super útil ver o histórico de um autor com várias publicações!)
    tem_colecao_docs = tipo_busca in ["Autor", "Orientador", "Co-orientador", "Palavra-chave", "Macrotema"] and 'docs' in locals() and docs
    titulo_secao = "Orientações" if tipo_busca in ["Orientador", "Co-orientador"] else "Documentos Associados"
    
    with tab_hist:
        if tem_colecao_docs:
            st.markdown(f"**Evolução no Tempo ({titulo_secao})**")
            
            tem_multiplos_ppgs = len(st.session_state.get('programas_selecionados_lista', [])) > 1
            cols_graf = st.columns(4) if tem_multiplos_ppgs else st.columns(3)
            agrupar_niveis = cols_graf[0].radio("Visão dos Níveis:", ["Separar Teses e Dissertações", "Agrupar tudo (Total)"], horizontal=True, key="agrup_niv_perfil")
            modo_analise = cols_graf[1].radio("Modo de Análise:", ["Visão Geral (Volume)", "Análise por Macrotemas"], horizontal=True, key="modo_ana_perfil")
            tipo_grafico = cols_graf[2].radio("Tipo de Gráfico:", ["Barras", "Linhas"], horizontal=True, key="tipo_graf_perfil")
            
            separar_ppg_hist = False
            if tem_multiplos_ppgs:
                separar_ppg_hist = cols_graf[3].radio("Separar por PPG:", ["Não", "Sim"], horizontal=True, key="agrup_ppg_perfil") == "Sim"
            
            df_docs = pd.DataFrame(docs)
            if not df_docs.empty and 'ano' in df_docs.columns:
                df_docs['ano'] = pd.to_numeric(df_docs['ano'], errors='coerce')
                df_docs = df_docs.dropna(subset=['ano'])
                df_docs['ano'] = df_docs['ano'].astype(int)
                
                if not df_docs.empty:
                    if 'nivel_academico' not in df_docs.columns: df_docs['nivel_academico'] = 'Outros'
                    else: df_docs['nivel_academico'] = df_docs['nivel_academico'].fillna('Outros')
                        
                    if 'macrotema' not in df_docs.columns: df_docs['macrotema'] = 'Multidisciplinar / Transversal'
                    else: df_docs['macrotema'] = df_docs['macrotema'].fillna('Multidisciplinar / Transversal')
                    
                    graf_func = px.bar if tipo_grafico == "Barras" else px.line
                    barmode_kw = dict(barmode='stack') if tipo_grafico == "Barras" else dict()
                    marker_kw = dict() if tipo_grafico == "Barras" else dict(markers=True)
                    
                    label_y = "Orientações" if tipo_busca in ["Orientador", "Co-orientador"] else "Documentos"
                    
                    facet_kws = dict(facet_col='programa_origem', facet_col_wrap=2) if separar_ppg_hist else dict()
                    groupby_cols = ['ano']
                    if separar_ppg_hist:
                        groupby_cols.append('programa_origem')
                        df_docs['programa_origem'] = df_docs['programa_origem'].fillna('Desconhecido')
                    
                    if modo_analise == "Visão Geral (Volume)":
                        if agrupar_niveis == "Agrupar tudo (Total)":
                            df_plot = df_docs.groupby(groupby_cols).size().reset_index(name='Volume')
                            fig = graf_func(df_plot, x='ano', y='Volume', title=f"{label_y} por Ano (Total)", template="plotly_dark", **marker_kw, **facet_kws)
                        else:
                            df_plot = df_docs.groupby(groupby_cols + ['nivel_academico']).size().reset_index(name='Volume')
                            fig = graf_func(df_plot, x='ano', y='Volume', color='nivel_academico', title=f"{label_y} por Ano e Nível Acadêmico", template="plotly_dark", **barmode_kw, **marker_kw, **facet_kws)
                    else:
                        if agrupar_niveis == "Agrupar tudo (Total)":
                            df_plot = df_docs.groupby(groupby_cols + ['macrotema']).size().reset_index(name='Volume')
                            fig = graf_func(df_plot, x='ano', y='Volume', color='macrotema', title=f"{label_y} por Ano e Macrotema", template="plotly_dark", **barmode_kw, **marker_kw, **facet_kws)
                        else:
                             df_docs['Nível/Tema'] = df_docs['nivel_academico'] + " - " + df_docs['macrotema']
                             df_plot = df_docs.groupby(groupby_cols + ['Nível/Tema']).size().reset_index(name='Volume')
                             fig = graf_func(df_plot, x='ano', y='Volume', color='Nível/Tema', title=f"{label_y} por Ano, Nível e Macrotema", template="plotly_dark", **barmode_kw, **marker_kw, **facet_kws)
                    
                    fig.update_layout(xaxis_title="Ano", yaxis_title="Quantidade", xaxis=dict(tickmode='linear', dtick=1))
                    if separar_ppg_hist: fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📈 A evolução histórica é gerada automaticamente para perfis que agrupam múltiplos trabalhos (Orientadores, Autores, Conceitos, etc).")


    with tab_nuvem:
        if tem_colecao_docs:
            st.markdown(f"**Lexicometria ({titulo_secao})**")
            
            col_nuvem1, col_nuvem2 = st.columns(2)
            modo_nuvem = col_nuvem1.selectbox("Fonte de Dados para Nuvem:", ["Tudo Combinado (Título + Resumo + Palavras-Chave)", "Apenas Palavras-chave", "Apenas Título", "Apenas Resumo"])
            separar_nuvem_ppg = False
            if tem_multiplos_ppgs:
                separar_nuvem_ppg = col_nuvem2.radio("Separar Nuvem por PPG:", ["Não", "Sim"], horizontal=True, key="sep_nuvem_ppg_perfil") == "Sim"

            stopwords = set(['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'uma', 'para', 'com', 'não', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'das', 'à', 'seu', 'sua', 'ou', 'nos', 'já', 'eu', 'também', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'sem', 'mesmo', 'aos', 'nas', 'me', 'esse', 'essa', 'num', 'nem', 'numa', 'pelos', 'pelas', 'este', 'esta', 'sobre', 'estudo', 'análise', 'proposta', 'uso', 'aplicação', 'desenvolvimento', 'modelo', 'sistema', 'avaliação', 'gestão', 'conhecimento', 'engenharia', 'objetivo', 'pesquisa', 'trabalho', 'resultados', 'método', 'foi', 'foram', 'são', 'ser', 'através', 'forma', 'apresenta', 'the', 'of', 'and', 'in', 'to', 'a', 'is', 'for', 'by', 'on', 'with', 'an', 'as', 'this', 'that', 'which', 'from', 'it', 'or', 'be', 'are', 'at', 'has', 'have', 'was', 'were', 'not', 'but', 'by'])
            
            def extrair_texto_docs(lista_docs):
                texto_completo = []
                for d in lista_docs:
                    if "Resumo" in modo_nuvem or "Tudo Combinado" in modo_nuvem: texto_completo.append(d.get('resumo', ''))
                    if "Título" in modo_nuvem or "Tudo Combinado" in modo_nuvem: texto_completo.append(d.get('titulo', ''))
                    if "Palavras-chave" in modo_nuvem or "Tudo Combinado" in modo_nuvem: texto_completo.append(" ".join(d.get('palavras_chave', [])))
                texto_str = " ".join([str(t) for t in texto_completo]).lower()
                return re.sub(r'[^\w\s]', '', texto_str)

            if separar_nuvem_ppg:
                docs_por_ppg = {}
                for d in docs:
                    ppg = d.get('programa_origem', 'Desconhecido')
                    if ppg not in docs_por_ppg: docs_por_ppg[ppg] = []
                    docs_por_ppg[ppg].append(d)
                    
                abas_ppg = st.tabs(list(docs_por_ppg.keys()))
                for idx, ppg in enumerate(docs_por_ppg.keys()):
                    with abas_ppg[idx]:
                        texto_limpo = extrair_texto_docs(docs_por_ppg[ppg])
                        palavras_nuvem = [p for p in texto_limpo.split() if p not in stopwords and len(p) > 2]
                        if palavras_nuvem:
                            freq_dict = dict(Counter(palavras_nuvem).most_common(100))
                            html_nuvem = renderizar_nuvem_interativa_html(freq_dict)
                            components.html(html_nuvem, height=480, scrolling=False)
                        else:
                            st.info("Palavras insuficientes para gerar a nuvem neste PPG.")
            else:
                texto_limpo = extrair_texto_docs(docs)
                palavras_nuvem = [p for p in texto_limpo.split() if p not in stopwords and len(p) > 2]
                if palavras_nuvem:
                    freq_dict = dict(Counter(palavras_nuvem).most_common(100))
                    html_nuvem = renderizar_nuvem_interativa_html(freq_dict)
                    components.html(html_nuvem, height=480, scrolling=False)
                else:
                    st.info("Palavras insuficientes para gerar a nuvem.")
        else:
            st.info("☁️ A nuvem de palavras requer um conjunto de documentos para análise lexicográfica.")


    with tab_orbita:
        st.markdown("### ⏳ Evolução Histórica da Órbita")
        st.caption("Visualize como os relacionamentos científicos se expandiram ao longo do tempo.")

        driver = conectar_neo4j() 
        
        if not driver:
            st.error("Erro: Não foi possível conectar ao banco de dados Neo4j.")
        else:
            # ✅ CORREÇÃO: Calcula o intervalo de anos APENAS dos documentos do PPG carregado
            anos_f = [int(d['ano']) for d in dados_completos if d.get('ano') and str(d['ano']).isdigit()]
            
            if not anos_f:
                st.warning("Aviso: Intervalo temporal não localizado para a seleção atual.")
                min_ano, max_ano = 2000, 2026
            else:
                min_ano, max_ano = min(anos_f), max(anos_f)
            
            anos_lista = list(range(min_ano, max_ano + 1))

            if anos_lista:
                col_play, col_slider = st.columns([1, 5])
                
                with col_play:
                    st.write("##") 
                    btn_play = st.button("▶️ Play", use_container_width=True, key="btn_play_orbita")
                
                with col_slider:
                    if 'ano_animacao' not in st.session_state:
                        st.session_state.ano_animacao = max_ano
                    
                    ano_selecionado = st.slider(
                        "Navegação Temporal:",
                        min_value=min_ano,
                        max_value=max_ano,
                        value=st.session_state.ano_animacao,
                        key="slider_temporal_orbita"
                    )
                    st.session_state.ano_animacao = ano_selecionado

                if btn_play:
                    for ano in range(min_ano, max_ano + 1):
                        st.session_state.ano_animacao = ano
                        time.sleep(0.6) 
                        st.rerun()

                with st.spinner(f"Mapeando conexões até {st.session_state.ano_animacao}..."):
                    # ✅ CORREÇÃO: Extrai a lista VIP de títulos do PPG selecionado
                    lista_titulos_ppg = [d.get('titulo') for d in dados_completos if d.get('titulo')]

                    nodes_orb, edges_orb = gerar_orbita_neo4j(
                        driver, 
                        termo_ativo, 
                        tipo_busca, 
                        profundidade=1,
                        _sna_global=sna_global,
                        ano_limite=st.session_state.ano_animacao,
                        titulos_validos=lista_titulos_ppg
                    )

                    if nodes_orb:
                        config_orb = Config(
                            width="100%", height=600, directed=False, physics=True,
                            nodeHighlightBehavior=True, highlightColor="#F39C12",
                            interaction={"navigationButtons": True, "keyboard": True, "hover": True}
                        )
                        agraph(nodes=nodes_orb, edges=edges_orb, config=config_orb)
                        st.info(f"Exibindo {len(nodes_orb)} entidades conectadas até {st.session_state.ano_animacao}.")
                    else:
                        st.warning(f"Não foram encontradas conexões para '{termo_ativo}' até o ano {st.session_state.ano_animacao}.")

    with tab_similares:
        st.markdown(f"**Recomendação Topológica (Itens mais próximos de {termo_ativo})**")
        st.caption("A proximidade é calculada pelo **Índice de Jaccard**, que mede a sobreposição do 'DNA acadêmico' (palavras-chave, macrotemas, coautorias e bancas compartilhadas) na rede complexa.")
        
        with st.spinner("Calculando similaridade vetorial na rede..."):
            similares = calcular_similares_rede(termo_ativo, tipo_busca, dados_completos)
            
        # Função interna para gerar a tabela bonita e os botões de navegação
        def render_tabela_similares(lista_dados, titulo_coluna_item, tipo_nav):
            if not lista_dados:
                st.info(f"Nenhum item com forte correlação encontrado para esta categoria.")
                return
                
            df = pd.DataFrame(lista_dados)[['Item', 'Similaridade (%)', 'Traços em Comum']]
            df = df.rename(columns={'Item': titulo_coluna_item})
            
            # Tabela com barra de progresso embutida na similaridade
            st.dataframe(
                df, 
                hide_index=True, 
                use_container_width=True,
                column_config={
                    "Similaridade (%)": st.column_config.ProgressColumn("Similaridade (%)", min_value=0, max_value=100, format="%.1f%%")
                }
            )
            
            # Botões clicáveis para pular direto para o perfil do item semelhante
            with st.expander(f"Navegar para os perfis ({titulo_coluna_item})"):
                for idx, row in df.iterrows():
                    item_nome = row[titulo_coluna_item]
                    st.button(f"Ir para: {item_nome}", key=f"btn_sim_{tipo_nav}_{hash(item_nome)}_{idx}", on_click=navegar_para, args=(tipo_nav, item_nome))

        if not similares or all(len(v) == 0 for v in similares.values()):
            st.warning("Este item possui conexões muito isoladas do resto do programa para que vizinhos próximos sejam calculados com precisão.")
        else:
            if tipo_busca == 'Documento':
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("##### 🎓 Teses Semelhantes")
                    render_tabela_similares(similares.get('Teses', []), "Tese", "Documento")
                with c2:
                    st.markdown("##### 📜 Dissertações Semelhantes")
                    render_tabela_similares(similares.get('Dissertações', []), "Dissertação", "Documento")
            elif tipo_busca == 'Autor':
                st.markdown("##### ✍️ Autores com Perfil Semelhante")
                render_tabela_similares(similares.get('Autores', []), "Autor", "Autor")
            elif tipo_busca in ['Orientador', 'Co-orientador']:
                st.markdown("##### 🏫 Orientadores/Coorientadores Semelhantes")
                # Se for professor, joga a navegação para Orientador como padrão de leitura
                render_tabela_similares(similares.get('Professores', []), "Professor(a)", "Orientador") 
            elif tipo_busca == 'Palavra-chave':
                st.markdown("##### 💡 Conceitos Semelhantes")
                render_tabela_similares(similares.get('Palavras-chave', []), "Palavra-chave", "Palavra-chave")
            elif tipo_busca == 'Macrotema':
                st.markdown("##### 🧠 Macrotemas Próximos")
                render_tabela_similares(similares.get('Macrotemas', []), "Macrotema", "Macrotema")

    
st.markdown("---")

# --- MÓDULO DE MACROTEMAS ---
st.header("🧠 Análise Temática Estrutural")

# --- NOVA TABELA ROBUSTA DE MACROTEMAS ---
O_total = len(dados_completos)
contagem_ori = Counter([d.get('orientador') for d in dados_completos if d.get('orientador')])
contagem_coori = Counter([co for d in dados_completos for co in d.get('co_orientadores', [])])

linhas_tabela = []
macrotemas_unicos = set([d.get('macrotema', 'Multidisciplinar / Transversal') for d in dados_completos])

for mt in macrotemas_unicos:
    docs_mt = [d for d in dados_completos if d.get('macrotema', 'Multidisciplinar / Transversal') == mt]
    O_k = len(docs_mt)
    
    teses = sum(1 for d in docs_mt if 'Tese' in d.get('nivel_academico', ''))
    dissertacoes = sum(1 for d in docs_mt if 'Disserta' in d.get('nivel_academico', ''))
    
    anos = [int(d['ano']) for d in docs_mt if d.get('ano') and str(d['ano']).isdigit()]
    ano_antigo = min(anos) if anos else "-"
    ano_recente = max(anos) if anos else "-"
    ano_modal = Counter(anos).most_common(1)[0][0] if anos else "-"
    
    def top_ql(entidades_na_mt, contagem_global):
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
                    if O_ik > contagem_local.get(top_ent, 0): top_ent = ent
        return top_ent, max_ql

    oris_mt = [d.get('orientador') for d in docs_mt if d.get('orientador')]
    top_ori, ql_ori = top_ql(oris_mt, contagem_ori)
    
    cooris_mt = [co for d in docs_mt for co in d.get('co_orientadores', [])]
    top_coori, ql_coori = top_ql(cooris_mt, contagem_coori)
    
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
    st.dataframe(df_temas, use_container_width=True, hide_index=True)

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
        st.plotly_chart(fig_mapa_macro, use_container_width=True)

with tab_tm3:
    st.markdown("Análise granulada dos micro-conceitos mais frequentes.")
    
    # Prepara o DataFrame dinâmico para as Top 50 Palavras-chave
    kw_data = []
    # Usamos o contagem_pk que já pegamos para a tabela, ou reconstruímos
    todas_pks = [pk for d in dados_completos for pk in d.get('palavras_chave', [])]
    kw_counter = Counter(todas_pks)
    
    # Limitar aos 40 principais para o mapa não virar uma "nuvem de poluição visual"
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
        st.plotly_chart(fig_mapa_kw, use_container_width=True)

with tab_tm4:
    st.markdown("### 🧊 Distribuição Espacial do Ecossistema")
    st.caption("Esta visualização tridimensional permite identificar a arquitetura da rede. Itens no topo do eixo **Betweenness** são os grandes intermediadores, enquanto o eixo **Closeness** revela os centros nervosos de disseminação.")
    
    # Botão de seleção de categoria (Radio horizontal para parecer um menu de botões)
    categoria_3d = st.radio(
        "Selecione a Dimensão para Visualizar:", 
        ["Documento", "Autor", "Orientador", "Palavra-chave", "Macrotema"],
        horizontal=True,
        key="selector_3d_global"
    )
    
    with st.spinner(f"Processando geometria 3D para {categoria_3d}s..."):
        # Chamamos a função sem o 'termo_destaque' para mostrar a visão global
        fig_3d_global = plotar_grafico_3d_sna(sna_global, categoria_3d)
        
        if fig_3d_global:
            st.plotly_chart(fig_3d_global, use_container_width=True, height=800)
        else:
            st.warning(f"Não há dados suficientes para gerar o gráfico 3D de {categoria_3d}s.")

st.markdown("---")
# A partir daqui, o código de "# --- MOTOR DE BUSCA (EGO-GRAPH) ---" continua exatamente igual.

st.header("🗄️ Base de Dados Completa com Métricas SNA")

base_expandida = []
for d in dados_completos:
    row = d.copy()
    titulo = row.get('titulo')
    
    if isinstance(row.get('autores'), list): row['autores'] = ", ".join(row['autores'])
    if isinstance(row.get('co_orientadores'), list): row['co_orientadores'] = ", ".join(row['co_orientadores'])
    if isinstance(row.get('palavras_chave'), list): row['palavras_chave'] = ", ".join(row['palavras_chave'])
    
    metricas_doc = sna_global.get(titulo, {})
    row['Grau (SNA)'] = metricas_doc.get('Grau Absoluto', 0)
    row['Betweenness (SNA)'] = round(metricas_doc.get('Betweenness', 0.0), 4)
    row['Closeness (SNA)'] = round(metricas_doc.get('Closeness', 0.0), 4)
    row['Comunidade (SNA)'] = metricas_doc.get('Comunidade', 'N/A')
    row['Ranking Global (SNA)'] = metricas_doc.get('Ranking Global', 'N/A')
    
    base_expandida.append(row)

df_base_completa = pd.DataFrame(base_expandida)
colunas_principais = ['titulo', 'ano', 'nivel_academico', 'autores', 'orientador', 'co_orientadores', 'palavras_chave', 'macrotema', 'Grau (SNA)', 'Betweenness (SNA)', 'Closeness (SNA)', 'Comunidade (SNA)', 'Ranking Global (SNA)', 'resumo', 'url']
colunas_finais = [c for c in colunas_principais if c in df_base_completa.columns] + [c for c in df_base_completa.columns if c not in colunas_principais]

max_g = int(df_base_completa['Grau (SNA)'].max()) if not df_base_completa.empty else 100
max_b = float(df_base_completa['Betweenness (SNA)'].max()) if not df_base_completa.empty else 1.0
max_c = float(df_base_completa['Closeness (SNA)'].max()) if not df_base_completa.empty else 1.0

df_base_completa['nivel_academico'] = df_base_completa['nivel_academico'].astype('category')

if 'macrotema' in df_base_completa.columns:
    df_base_completa['macrotema'] = df_base_completa['macrotema'].astype('category')
if 'Comunidade (SNA)' in df_base_completa.columns:
    df_base_completa['Comunidade (SNA)'] = df_base_completa['Comunidade (SNA)'].astype('category')

st.dataframe(
    df_base_completa[colunas_finais],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Grau (SNA)": st.column_config.ProgressColumn("Grau (SNA)", min_value=0, max_value=max_g, format="%d"),
        "Betweenness (SNA)": st.column_config.ProgressColumn("Betweenness (SNA)", min_value=0, max_value=max_b, format="%.4f"),
        "Closeness (SNA)": st.column_config.ProgressColumn("Closeness (SNA)", min_value=0, max_value=max_c, format="%.4f")
    }
)
