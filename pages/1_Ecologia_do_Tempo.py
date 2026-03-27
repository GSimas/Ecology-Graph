import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import datetime

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Ecologia do Tempo | PPGEGC",
    page_icon="⏳",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# Estilização customizada (CSS) mantendo a identidade visual
st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; }
    h1, h2, h3, h4, h5 { color: #F39C12; font-family: 'Helvetica Neue', sans-serif; }
    .stMetric { background-color: #2C3E50; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.5); }
    
    button[kind="primary"] {
        background-color: #2ECC71 !important;
        color: white !important;
        border-color: #27AE60 !important;
        font-weight: bold !important;
    }
    button[kind="primary"]:hover {
        background-color: #27AE60 !important;
        border-color: #2ECC71 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- FUNÇÕES DE BACK-END ---
@st.cache_data
def carregar_dados():
    try:
        with open('base_ppgegc.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Ficheiro base_ppgegc.json não encontrado. Certifique-se de que está na pasta raiz do projeto.")
        return []

@st.cache_data
def preparar_dados_temporais(dados):
    df = pd.DataFrame(dados)
    df['Ano'] = pd.to_numeric(df.get('ano'), errors='coerce')
    df = df.dropna(subset=['Ano'])
    df['Ano'] = df['Ano'].astype(int)
    df['nivel_academico'] = df.get('nivel_academico', 'Outros / Não Especificado').fillna('Outros / Não Especificado')
    
    # Explodir as palavras-chave para ter uma linha por conceito/ano
    df_exp = df.explode('palavras_chave')
    df_exp = df_exp.dropna(subset=['palavras_chave'])
    df_exp['palavras_chave'] = df_exp['palavras_chave'].str.strip()
    return df_exp

@st.cache_data
def detetar_explosoes(df_exp, min_ocorrencias_totais, sensibilidade_z):
    """
    Algoritmo de Deteção de Picos (Burst Detection) adaptado para séries anuais.
    Identifica anos em que a frequência de um conceito superou significativamente a sua média histórica.
    """
    # 1. Agrupar por Ano e Conceito
    df_freq = df_exp.groupby(['Ano', 'palavras_chave']).size().reset_index(name='Frequencia')
    
    # 2. Filtrar conceitos que têm relevância estatística mínima
    contagem_total = df_freq.groupby('palavras_chave')['Frequencia'].sum()
    conceitos_validos = contagem_total[contagem_total >= min_ocorrencias_totais].index
    df_freq = df_freq[df_freq['palavras_chave'].isin(conceitos_validos)]
    
    if df_freq.empty:
        return pd.DataFrame()

    # 3. Criar uma grelha completa (Ano vs Conceito) para preencher com zeros os anos sem menções
    anos_unicos = range(df_freq['Ano'].min(), df_freq['Ano'].max() + 1)
    grelha = pd.MultiIndex.from_product([anos_unicos, conceitos_validos], names=['Ano', 'palavras_chave']).to_frame(index=False)
    df_completo = pd.merge(grelha, df_freq, on=['Ano', 'palavras_chave'], how='left').fillna(0)
    
    # 4. Cálculo do Z-Score e Médias Móveis (A Matemática da Explosão)
    # Ordenar chronologicamente
    df_completo = df_completo.sort_values(['palavras_chave', 'Ano'])
    
    # Calcular média e desvio padrão histórico expansivo para cada palavra
    df_completo['Media_Historica'] = df_completo.groupby('palavras_chave')['Frequencia'].transform(lambda x: x.expanding().mean().shift(1).fillna(0))
    df_completo['Desvio_Historico'] = df_completo.groupby('palavras_chave')['Frequencia'].transform(lambda x: x.expanding().std().shift(1).fillna(0))
    
    # Uma explosão ocorre quando a frequência atual é maior que a média + (Z * desvio padrão)
    # e a frequência é pelo menos 2 (para evitar falsos positivos de 0 para 1)
    df_completo['Limiar_Explosao'] = df_completo['Media_Historica'] + (sensibilidade_z * df_completo['Desvio_Historico'])
    df_completo['Limiar_Explosao'] = df_completo['Limiar_Explosao'].replace(0, 1) # Proteção matemática
    
    df_completo['Em_Explosao'] = (df_completo['Frequencia'] > df_completo['Limiar_Explosao']) & (df_completo['Frequencia'] >= 2)
    
    return df_completo

# --- INÍCIO DA INTERFACE ---
st.title("⏳ Ecologia do Tempo: Deteção de Emergências")
st.markdown("""
> **Burst Detection (Deteção de Explosões):** Esta análise vai além da simples contagem de palavras. Utilizando modelos estatísticos expansivos, a ferramenta identifica o ano exato em que um conceito deixou de ser periférico e sofreu uma "explosão" de interesse, tornando-se um paradigma emergente no PPGEGC.
""")
st.markdown("---")

dados_brutos = carregar_dados()
if not dados_brutos:
    st.stop()

df_base = preparar_dados_temporais(dados_brutos)
niveis_disponiveis = sorted(list(df_base['nivel_academico'].unique()))
min_ano_global = int(df_base['Ano'].min())
max_ano_global = int(df_base['Ano'].max())

# --- FILTROS DE ANÁLISE ---
with st.form("form_burst"):
    st.subheader("⚙️ Calibração do Algoritmo Estatístico")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        niveis_sel = st.multiselect("Nível Académico:", niveis_disponiveis, default=niveis_disponiveis)
    
    with col2:
        min_freq = st.number_input("Ocorrências Totais Mínimas (Filtro de Ruído):", min_value=2, max_value=50, value=5, 
                                   help="Ignora palavras-chave que apareceram muito poucas vezes na história do programa.")
    
    with col3:
        z_score = st.slider("Sensibilidade da Explosão (Z-Score):", min_value=1.0, max_value=3.0, value=1.5, step=0.1,
                            help="Valores mais baixos detetam mais picos menores. Valores próximos de 3.0 revelam apenas explosões sísmicas massivas.")

    btn_analisar = st.form_submit_button("Executar Algoritmo de Ecologia do Tempo", type="primary")

if btn_analisar:
    # 1. Aplicar filtros básicos
    df_filtrado = df_base[df_base['nivel_academico'].isin(niveis_sel)].copy()
    
    if df_filtrado.empty:
        st.warning("Não há documentos suficientes com estes filtros.")
    else:
        with st.spinner("A calcular matrizes expansivas e a procurar quebras de paradigma..."):
            # 2. Processar Algoritmo
            df_analise = detetar_explosoes(df_filtrado, min_freq, z_score)
            
            if df_analise.empty or not df_analise['Em_Explosao'].any():
                st.info("O algoritmo não detetou nenhuma explosão estatística com esta configuração. Tente diminuir a Sensibilidade (Z-Score) ou o Filtro de Ruído.")
            else:
                # 3. Extrair os Eventos de Explosão
                eventos_explosao = df_analise[df_analise['Em_Explosao']].copy()
                
                # Criar blocos para o Gráfico de Gantt
                gantt_data = []
                for conceito in eventos_explosao['palavras_chave'].unique():
                    anos_explosao = eventos_explosao[eventos_explosao['palavras_chave'] == conceito]['Ano'].tolist()
                    
                    # Agrupar anos consecutivos num único bloco de explosão
                    bloco_inicio = anos_explosao[0]
                    bloco_fim = anos_explosao[0]
                    
                    for ano in anos_explosao[1:]:
                        if ano == bloco_fim + 1:
                            bloco_fim = ano
                        else:
                            gantt_data.append({'Conceito': conceito, 'Início': f"{bloco_inicio}-01-01", 'Fim': f"{bloco_fim}-12-31", 'Ano_Pico': bloco_fim})
                            bloco_inicio = ano
                            bloco_fim = ano
                    gantt_data.append({'Conceito': conceito, 'Início': f"{bloco_inicio}-01-01", 'Fim': f"{bloco_fim}-12-31", 'Ano_Pico': bloco_fim})
                
                df_gantt = pd.DataFrame(gantt_data).sort_values(by='Início')

                # --- VISUALIZAÇÃO 1: CICLO DE VIDA (GANTT) ---
                st.markdown("### 📊 Ciclo de Vida do Conhecimento (Paradigma e Duração)")
                st.write("Cada barra representa a 'janela temporal' em que o conceito viveu o seu pico de atenção no programa.")
                
                fig_gantt = px.timeline(df_gantt, x_start="Início", x_end="Fim", y="Conceito", color="Conceito",
                                        title="Cronologia das Explosões Epistémicas")
                fig_gantt.update_yaxes(autorange="reversed") # Ordem cronológica de cima para baixo
                fig_gantt.update_layout(template="plotly_dark", showlegend=False, xaxis_title="Eixo do Tempo", yaxis_title="")
                st.plotly_chart(fig_gantt, use_container_width=True)

                st.markdown("---")

                # --- VISUALIZAÇÃO 2: A ANATOMIA DA EXPLOSÃO (LINE CHART) ---
                st.markdown("### 📈 A Anatomia das Explosões")
                st.write("Análise detalhada do volume. Os pontos vermelhos marcam o ano exato em que a matemática validou a 'explosão'.")
                
                # Para não poluir, selecionamos os 5 conceitos com os picos mais altos de frequência durante a explosão
                top_conceitos_explosao = eventos_explosao.sort_values(by='Frequencia', ascending=False)['palavras_chave'].unique()[:8]
                df_linhas = df_analise[df_analise['palavras_chave'].isin(top_conceitos_explosao)]
                
                fig_lines = px.line(df_linhas, x='Ano', y='Frequencia', color='palavras_chave', 
                                    title="Trajetória de Frequência e Pontos de Rutura")
                
                # Adicionar marcadores apenas onde houve explosão
                explosoes_filtradas = df_linhas[df_linhas['Em_Explosao']]
                fig_lines.add_trace(go.Scatter(
                    x=explosoes_filtradas['Ano'],
                    y=explosoes_filtradas['Frequencia'],
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='star', line=dict(width=2, color='white')),
                    name='Momento de Explosão (Burst)'
                ))
                
                fig_lines.update_layout(template="plotly_dark", xaxis_title="Ano", yaxis_title="Documentos Publicados", hovermode="x unified", xaxis=dict(tickmode='linear', dtick=1))
                st.plotly_chart(fig_lines, use_container_width=True)

                # --- VISUALIZAÇÃO 3: TABELA DE REGISTOS ---
                st.markdown("### 📋 Diário de Ruturas Paradigmáticas")
                
                # Formatar a tabela final
                df_tabela_final = eventos_explosao[['Ano', 'palavras_chave', 'Frequencia']].copy()
                df_tabela_final.columns = ['Ano da Explosão', 'Conceito Emergente', 'Volume de Publicações no Ano']
                df_tabela_final = df_tabela_final.sort_values(by=['Ano da Explosão', 'Volume de Publicações no Ano'], ascending=[False, False])
                df_tabela_final.insert(0, 'Posição (Histórica)', range(1, len(df_tabela_final) + 1))
                
                st.dataframe(df_tabela_final, use_container_width=True, hide_index=True)
                
                # Exportação
                csv = df_tabela_final.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Descarregar Tabela de Explosões (CSV)", data=csv, file_name='explosoes_historicas.csv', mime='text/csv')
