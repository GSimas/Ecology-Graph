import streamlit as st
import pandas as pd
import networkx as nx
import time

from app_config import get_gemini_api_key
from gemini_utils import content_from_text, generate_content

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Consultor Acadêmico IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização Adaptativa Nativa
st.markdown("""
    <style>
    /* Box do chat usando as variáveis fluídas nativas do Streamlit */
    /* Isso garante contraste perfeito independente se for forçado o light ou dark theme no app */
    .stChatMessage { 
        background-color: var(--secondary-background-color); 
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px; 
        margin-bottom: 10px; 
        padding: 5px 15px;
    }
    
    /* Cor vibrante dos títulos que combina com fundos claros e escuros */
    h1, h2, h3 { 
        color: #F39C12; 
        font-family: 'Helvetica Neue', sans-serif; 
    }
    
    /* Suaviza e melhora o contraste geral para as caixas de aviso */
    .stAlert {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- PROTEÇÃO E LEITURA DA MEMÓRIA VIVA (SESSION STATE) ---
if 'dados_completos' not in st.session_state or not st.session_state['dados_completos']:
    st.warning("⚠️ Nenhuma base de dados carregada na memória.")
    st.info("Por favor, vá à página inicial e carregue a base de um Programa de Pós-Graduação antes de iniciar o chat.")
    st.stop()

base_dados = st.session_state['dados_completos']
nome_programa = st.session_state.get('nome_programa', 'Programa Selecionado')

# --- CONFIGURAÇÃO AUTOMÁTICA DA API KEY ---
api_key = get_gemini_api_key()
if not api_key:
    st.error("⚠️ Erro: GEMINI_API_KEY não encontrada em variáveis de ambiente ou secrets.")
    st.stop()

# --- MOTOR DE CONTEXTO ABSOLUTO (SNA + CATÁLOGO) ---
def preparar_contexto_academico(dados, nome_prog):
    """Constrói o cérebro da IA com a rede matemática e o catálogo completo de pesquisas."""
    cache_key = f"contexto_ia_{nome_prog}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
        
    total_docs = len(dados)
    # Estimativa aproximada de tempo: ~1 segundo a cada 150 documentos
    tempo_estimado = max(3, int(total_docs / 150))
    
    time_container = st.empty()
    time_container.info(f"⏱️ **Tempo estimado de processamento:** ~{tempo_estimado} segundos.")
    bar = st.progress(0)
    status_text = st.empty()
    
    status_text.markdown("⏳ **Passo 1/4:** Construindo a Topologia da Rede Acadêmica...")
    
    # 1. Processamento Topológico (SNA)
    G = nx.Graph()
    passo_progresso = max(1, total_docs // 40)
    for i, d in enumerate(dados):
        if i % passo_progresso == 0:
            bar.progress(min(40, int((i / total_docs) * 40)))
            
        titulo = d.get('titulo')
        if not titulo: continue
        G.add_node(titulo, tipo='Documento')
        ori = d.get('orientador')
        if ori: G.add_node(ori, tipo='Orientador'); G.add_edge(titulo, ori)
        for co in d.get('co_orientadores', []):
            if co: G.add_node(co, tipo='Co-orientador'); G.add_edge(titulo, co)
        for pk in d.get('palavras_chave', []):
            if pk: G.add_node(pk, tipo='Conceito'); G.add_edge(titulo, pk)

    bar.progress(40)
    status_text.markdown("⏳ **Passo 2/4:** Calculando Centralidade de Grau (Degree Centrality)...")
    deg_cent = nx.degree_centrality(G)
    
    bar.progress(60)
    status_text.markdown("⏳ **Passo 3/4:** Mapeando Pontes Interdisciplinares (Betweenness Centrality)...")
    num_nodos = len(G.nodes)
    if num_nodos > 1000:
        bet_cent = nx.betweenness_centrality(G, k=100) # Aproximação rápida
    else:
        bet_cent = nx.betweenness_centrality(G)
        
    bar.progress(85)
    status_text.markdown("⏳ **Passo 4/4:** Redigindo Dossiê Comportamental para a IA...")

    def extrair_top(tipo_filtro, metrica_dict, top_n=10):
        nós = [n for n, attr in G.nodes(data=True) if attr.get('tipo') == tipo_filtro]
        sub = {n: met_val for n, met_val in metrica_dict.items() if n in nós}
        return sorted(sub.items(), key=lambda x: x[1], reverse=True)[:top_n]

    top_oris_vol = extrair_top('Orientador', deg_cent)
    top_oris_ponte = extrair_top('Orientador', bet_cent)
    top_conceitos = extrair_top('Conceito', deg_cent, top_n=20)

    # 2. Mapeamento de Perfil dos Orientadores
    perfil_ori = {}
    for d in dados:
        ori = d.get('orientador')
        if not ori: continue
        if ori not in perfil_ori:
            perfil_ori[ori] = {'temas': set(), 'total': 0}
        perfil_ori[ori]['total'] += 1
        if d.get('macrotema'): 
            perfil_ori[ori]['temas'].add(d.get('macrotema'))

    # 3. Construção do Dossiê de Texto para a IA
    ctx = f"=== DOSSIÊ INSTITUCIONAL: {nome_prog.upper()} ===\n"
    ctx += f"Total de Trabalhos Publicados: {len(dados)}\n"
    ctx += f"Corpo Docente (Orientadores Ativos): {len(perfil_ori)}\n\n"
    
    ctx += "--- MÉTRICAS DE REDE (SNA) ---\n"
    ctx += "Líderes em Volume de Orientação (Degree Centrality): " + ", ".join([f"{n}" for n, v in top_oris_vol]) + "\n"
    ctx += "Pontes Interdisciplinares (Betweenness Centrality - Conectam diferentes áreas): " + ", ".join([f"{n}" for n, v in top_oris_ponte]) + "\n"
    ctx += "Principais Conceitos Pesquisados: " + ", ".join([f"{n}" for n, v in top_conceitos]) + "\n\n"

    ctx += "--- PERFIL DE ORIENTAÇÃO (MAPA DE ESPECIALISTAS) ---\n"
    for ori, info in sorted(perfil_ori.items(), key=lambda x: x[1]['total'], reverse=True):
        temas_str = ", ".join(list(info['temas']))
        ctx += f"- {ori} | Orientou: {info['total']} trabalhos | Especialidades: [{temas_str}]\n"

    ctx += "\n--- CATÁLOGO DE TESES E DISSERTAÇÕES (BASE PARA RECOMENDAÇÃO) ---\n"
    for d in dados[:1500]:
        pks = ", ".join(d.get('palavras_chave', [])[:4])
        ctx += f"TÍTULO: {d.get('titulo')} | AUTOR: {', '.join(d.get('autores', []))} | ORIENTADOR: {d.get('orientador')} | TEMA: {d.get('macrotema')} | CONCEITOS: {pks}\n"

    bar.progress(100)
    status_text.markdown("✅ **Genoma acadêmico compilado com sucesso!**")
    time.sleep(1) # Dá um tempinho pro usuário ver o sucesso
    
    # Limpa a tela
    time_container.empty()
    bar.empty()
    status_text.empty()
    
    st.session_state[cache_key] = ctx
    return ctx

contexto_absoluto = preparar_contexto_academico(base_dados, nome_programa)

# --- INSTRUÇÕES DO SISTEMA (SYSTEM PROMPT) ---
system_prompt = f"""
Você é o Consultor Acadêmico e Analista de Inteligência de Redes especializado no(s) programa(s): {nome_programa} da Universidade Federal de Santa Catarina (UFSC).

Seu cérebro foi carregado com a taxonomia completa, estatísticas de rede e o catálogo de produções deste ecossistema.

SUA MISSÃO:
1. Auxiliar futuros mestrandos e doutorandos a refinarem suas propostas de pesquisa.
2. Recomendar o melhor Orientador(a) ou Co-orientador(a) com base na ideia do candidato, cruzando a ideia dele com as 'Especialidades' dos professores listados.
3. Sugerir teses/dissertações anteriores exatas (cite o TÍTULO e o AUTOR presentes no catálogo) para o aluno ler e se inspirar, se a ideia for semelhante.
4. Explicar a dinâmica da rede do programa (quem são os líderes de pesquisa, quem atua como ponte interdisciplinar).

REGRAS DE CONDUTA:
- Seja acolhedor, altamente profissional e acadêmico.
- Baseie suas recomendações EXCLUSIVAMENTE nos dados fornecidos no Dossiê abaixo.
- Se a ideia de projeto do candidato fugir completamente do escopo do programa, seja honesto e diga que o programa pode não ser o melhor encaixe, ou sugira uma adaptação para os 'Principais Conceitos Pesquisados'.

DOSSIÊ DE CONHECIMENTO (BASE DE DADOS):
{contexto_absoluto}
"""

st.title("🤖 Consultoria Acadêmica de IA")
st.caption(f"Powered by Gemini Flash | Especialista em: {nome_programa}")
st.markdown("Bem-vindo! Descreva sua ideia de projeto, pergunte sobre o perfil dos professores, ou peça indicações de teses alinhadas com seu interesse de pesquisa.")

# Histórico de mensagens UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada do Chat
if prompt := st.chat_input("Ex: Quero pesquisar sobre governança de dados na saúde. Quem seria o melhor orientador?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Converte o histórico do Streamlit para o formato do Gemini
    gemini_history = []
    for m in st.session_state.messages[:-1]: # Exclui a última (que acabamos de adicionar)
        role = "model" if m["role"] == "assistant" else "user"
        gemini_history.append(content_from_text(role, m["content"]))
        
    # Adiciona a mensagem atual
    gemini_history.append(content_from_text("user", prompt))

    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        full_res = ""
        
        try:
            response = generate_content(
                contents=gemini_history,
                api_key=api_key,
                model_candidates=("gemini-2.5-flash", "gemini-2.0-flash"),
                system_instruction=system_prompt,
                temperature=0.3,
                stream=True,
            )

            for chunk in response:
                if getattr(chunk, "text", None):
                    full_res += chunk.text
                    msg_placeholder.markdown(full_res + "▌")
            
            msg_placeholder.markdown(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})
            
        except Exception as e:
            st.error(f"Erro na comunicação com a IA do Google: {e}")
