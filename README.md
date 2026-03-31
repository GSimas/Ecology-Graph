# 🌌 Ecologia do Conhecimento: Plataforma Cientométrica e Topológica

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)
![NetworkX](https://img.shields.io/badge/NetworkX-000000?style=for-the-badge&logo=python&logoColor=white)

Plataforma avançada de **Cienciometria, Análise de Redes Sociais (SNA) e Mineração de Textos** focada no mapeamento de Programas de Pós-Graduação (PPGs). O sistema analisa a produção acadêmica (teses e dissertações) para revelar a estrutura latente do conhecimento, redes de colaboração, genealogia acadêmica e o papel topológico de pesquisadores e conceitos.

---

## 🎯 Visão Geral

Este projeto transcende a contagem tradicional de publicações. Utilizando a teoria de **Sistemas Complexos** e **Grafos**, a aplicação mapeia a "Ecologia do Conhecimento" de ecossistemas acadêmicos. Através de uma arquitetura híbrida (memória + banco de grafos em nuvem) e IA Generativa, o sistema permite:

- Avaliar a maturidade e resiliência de linhas de pesquisa.
- Identificar *Brokers* (corretores de conhecimento) e *Hubs* interdisciplinares.
- Rastrear a genealogia acadêmica (alunos que se tornam formadores).
- Recomendar conexões através de similaridade vetorial/topológica.

---

## 🚀 Funcionalidades Principais

### 1. 🧊 Espaço Topológico Multidimensional
Renderização de um espaço de fase 3D utilizando as três principais leis de força da rede: **Grau (Volume), Betweenness (Intermediação) e Closeness (Proximidade)**. Permite a identificação visual rápida de anomalias, clusters isolados e líderes do ecossistema.

### 2. 🧭 Mapeamento Temático (Bibliometrix Style)
Implementação inspirada no modelo de Callon para quadrantes de maturidade. Classifica Macrotemas e Palavras-chave em:
* **Temas Motores:** Alta centralidade e alta densidade.
* **Temas de Nicho:** Alta densidade, baixa centralidade.
* **Temas Emergentes/Declínio:** Baixa centralidade e densidade.
* **Temas Básicos (Transversais):** Alta centralidade, baixa densidade.

### 3. 🧬 Métricas de Ecologia Profunda (SNA Avançado)
Cálculo em tempo real da "física" da rede acadêmica:
* **Assortatividade ($r$):** Mede a endogenia (panelinhas) vs. expansão interdisciplinar.
* **Rich-Club Coefficient ($\Phi$):** Avalia se a "elite" do programa (Top 20% hubs) colabora entre si ou atua em silos.
* **Expoente Gamma ($\gamma$) da Lei de Potência:** Avalia a dependência da rede em relação aos grandes líderes (validação da estrutura *Scale-Free*).
* **Correlação de Spearman ($\rho$):** Mede a presença de Inovadores/Brokers na rede cruzando Grau e Betweenness.

### 4. 🔗 Recomendação Topológica (Índice de Jaccard)
Sistema de recomendação que sugere conexões latentes (Autores parecidos, Teses correlatas, Conceitos vizinhos) calculando a sobreposição do "DNA acadêmico" (similaridade de vizinhança) na rede complexa.

### 5. 🌌 Ego-Graphs e Órbitas de Relacionamento
Consultas na linguagem **Cypher** disparadas diretamente para o **Neo4j AuraDB**, renderizando subgrafos interativos que mostram a área de influência de um pesquisador ou conceito em até 3 graus de profundidade.

### 6. 🤖 Síntese Dinâmica com IA Generativa
Integração com a API do **Google Gemini** para ler amostras do repositório em tempo real e gerar descritivos fenomenológicos/epistemológicos do foco de pesquisa de um PPG, eliminando a necessidade de leitura manual massiva.

---

## 🏗️ Arquitetura e Stack Tecnológico

A plataforma utiliza um modelo de **Arquitetura Híbrida**:
* **Front-end & UI:** Streamlit (Python)
* **Back-end Matemático:** Pandas, SciPy, NetworkX (Cálculo de métricas em RAM para alta velocidade em filtros tabulares).
* **Back-end de Grafos:** Neo4j AuraDB (Cloud) para armazenamento persistente da ontologia e consultas estruturais complexas via Cypher.
* **Visualização de Dados:** Plotly (Gráficos 3D, Linhas, Barras, Scatter), Streamlit-agraph (vis.js para redes).
* **NLP & IA:** Google Generative AI (Gemini 2.5/2.0 Flash).

---

## 🛠️ Instalação e Configuração

### Pré-requisitos
- Python 3.9 ou superior.
- Uma conta gratuita no [Neo4j Aura](https://console.neo4j.io/).
- Uma chave de API do [Google AI Studio](https://aistudio.google.com/).

### Passo a Passo

1. **Clone o Repositório:**
```bash
git clone [https://github.com/SEU_USUARIO/ecologia-conhecimento.git](https://github.com/SEU_USUARIO/ecologia-conhecimento.git)
cd ecologia-conhecimento