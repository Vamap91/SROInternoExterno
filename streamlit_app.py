import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
import json
from datetime import datetime
import re

st.set_page_config(
    page_title="Análise de Risco de Externalização",
    page_icon="⚠️",
    layout="wide"
)

# Configurar OpenAI API usando secrets do Streamlit
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error("⚠️ Erro ao configurar OpenAI API. Verifique se a chave está configurada em Settings > Secrets do Streamlit.")
    st.stop()

def classify_channel_risk(channel_value):
    """
    Classifica o peso de risco baseado no canal (coluna CANAL_DE_ENTRADA_MANIFESTACAO)
    
    Pesos conforme solicitado:
    - Ext. Ouvidoria: 100 pontos (mais crítico)
    - Externo / Web - Reclame Aqui: 75 pontos
    - Externo - Focais: 50 pontos
    - Interno: 0 pontos (para análise de externalização)
    """
    if pd.isna(channel_value):
        return 0, "Não classificado"
    
    channel_str = str(channel_value).strip().lower()
    
    # Ext. Ouvidoria - 100 pontos (mais crítico)
    if "ouvidoria" in channel_str:
        return 100, "Ext. Ouvidoria"
    
    # Web - Reclame Aqui / Externo (sem focais) - 75 pontos
    elif "reclame aqui" in channel_str or "reclameaqui" in channel_str:
        return 75, "Web - Reclame Aqui"
    elif "externo" in channel_str and "focais" not in channel_str:
        return 75, "Externo"
    
    # Externo - Focais - 50 pontos
    elif "focais" in channel_str or "ext. focais" in channel_str or "externo - focais" in channel_str:
        return 50, "Externo - Focais"
    
    # Interno - 0 pontos
    elif "interno" in channel_str or "interna" in channel_str:
        return 0, "Interno"
    
    else:
        return 0, "Não classificado"

def analyze_internal_to_external_risk(client, text_content, channel_type):
    """
    Análise 1: Risco de reclamações INTERNAS virarem EXTERNAS (0-100)
    """
    
    prompt = f"""Você é um analista de risco especializado em prever a probabilidade de reclamações internas se tornarem externas.

CONTEXTO:
Esta é uma reclamação atualmente classificada como: {channel_type}

TEXTO DA RECLAMAÇÃO:
{text_content}

TAREFA:
Analise o texto e calcule o risco (0-100) de esta reclamação se tornar externa (ReclameAqui, Procon, Ouvidoria).

FATORES A CONSIDERAR (pontuação 0-100):

1. INDICADORES DE EXTERNALIZAÇÃO (peso alto):
   - Menções a "ReclameAqui", "Procon", "advogado", "processar": +20 pts cada
   - Menções a "ouvidoria", "órgão de defesa": +15 pts cada
   - Ameaças diretas ("vou publicar", "vou denunciar"): +25 pts cada
   - Menção a corretor/corretora: +10 pts
   - Múltiplos canais de contato: +10 pts

2. ESTADO EMOCIONAL (peso médio):
   - Palavras críticas ("absurdo", "inaceitável", "revoltado", "indignado"): +5 pts cada
   - Frustração com processo ("ninguém resolve", "já tentei X vezes"): +10 pts
   - Ultimatos ("última vez", "prazo de X dias"): +15 pts

3. GRAVIDADE DO PROBLEMA (peso médio):
   - Problemas técnicos graves (defeito, dano, prejuízo, mal feito, torto, amassado): +10 pts
   - Múltiplas tentativas sem resolução: +15 pts
   - Tempo de espera excessivo (muitos dias): +10 pts
   - Falta de cuidado/qualidade: +10 pts

4. ATENUANTES (reduzem risco):
   - Negativa técnica sem insatisfação explícita: -20 pts
   - Procedimentos administrativos padrão: -10 pts
   - Palavras positivas ou neutras: -5 pts cada

FORMATO DE SAÍDA (JSON):
{{
    "risk_score": <número de 0 a 100>,
    "risk_level": "<Baixo/Médio/Alto/Crítico>",
    "key_factors": ["fator1", "fator2", "fator3"],
    "detected_threats": ["ameaça1", "ameaça2"],
    "emotional_tone": "<descrição breve do tom emocional>",
    "recommendation": "<recomendação de ação>"
}}

CLASSIFICAÇÃO:
- Baixo: 0-30
- Médio: 31-60
- Alto: 61-85
- Crítico: 86-100

Retorne APENAS o JSON, sem texto adicional."""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Você é um analista de risco especializado em prever externalizações de reclamações."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Extrair JSON da resposta
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            return {
                "risk_score": 0,
                "risk_level": "Erro",
                "key_factors": ["Erro ao processar resposta"],
                "detected_threats": [],
                "emotional_tone": "N/A",
                "recommendation": "Revisar manualmente"
            }
    except Exception as e:
        st.error(f"Erro na análise: {str(e)}")
        return {
            "risk_score": 0,
            "risk_level": "Erro",
            "key_factors": [str(e)],
            "detected_threats": [],
            "emotional_tone": "N/A",
            "recommendation": "Erro na análise"
        }

def analyze_external_repeat_risk(client, text_content, channel_type, channel_risk_score):
    """
    Análise 2: Risco de reclamações EXTERNAS serem REPETIDAS/ESCALADAS (0-100)
    """
    
    prompt = f"""Você é um analista de risco especializado em prever reincidência e escalação de reclamações externas.

CONTEXTO:
Esta é uma reclamação EXTERNA classificada como: {channel_type}
Peso base do canal: {channel_risk_score} pontos

TEXTO DA RECLAMAÇÃO:
{text_content}

TAREFA:
Analise o texto e calcule o risco (0-100) de o cliente RECLAMAR NOVAMENTE ou ESCALAR para outros canais.

FATORES A CONSIDERAR:

1. INSATISFAÇÃO COM RESOLUÇÃO ANTERIOR (peso crítico):
   - "Não resolveram", "continua o problema": +25 pts
   - "Mesma situação de antes", "voltou a acontecer": +20 pts
   - "Já reclamei antes": +15 pts
   - Menção a múltiplas reclamações anteriores: +30 pts

2. ESCALAÇÃO PROGRESSIVA (peso alto):
   - Menção a canais adicionais ("agora vou ao Procon", "vou processar"): +25 pts cada
   - Ameaças jurídicas após reclamação externa: +35 pts
   - Menção a advogado após ReclameAqui: +40 pts
   - "Última tentativa antes de processar": +45 pts

3. GRAVIDADE DO CANAL ATUAL (peso base):
   - Ext. Ouvidoria (100 pts): já é crítico, risco de ação jurídica
   - Web - Reclame Aqui (75 pts): risco de Procon/jurídico
   - Externo (75 pts): risco de canais formais
   - Externo - Focais (50 pts): risco de ReclameAqui/Procon

4. ESTADO EMOCIONAL ATUAL (peso médio):
   - Frustração extrema ("cansado", "desistindo"): +20 pts
   - Raiva/indignação crescente: +15 pts
   - Menção a prejuízo financeiro/tempo: +10 pts

5. PADRÃO DE COMPORTAMENTO (peso médio):
   - Cliente persistente (múltiplos contatos): +15 pts
   - Cliente documenta tudo: +10 pts
   - Cliente menciona prazos legais: +20 pts

FORMATO DE SAÍDA (JSON):
{{
    "risk_score": <número de 0 a 100>,
    "risk_level": "<Baixo/Médio/Alto/Crítico>",
    "repeat_probability": "<Baixa/Média/Alta/Muito Alta>",
    "escalation_channels": ["canal1", "canal2"],
    "previous_complaints_detected": <true/false>,
    "key_indicators": ["indicador1", "indicador2"],
    "urgency_level": "<Baixa/Média/Alta/Urgente>",
    "recommendation": "<recomendação de ação>"
}}

CLASSIFICAÇÃO:
- Baixo: 0-30 (improvável repetir)
- Médio: 31-60 (pode reclamar novamente)
- Alto: 61-85 (provável escalação)
- Crítico: 86-100 (escalação iminente)

Retorne APENAS o JSON, sem texto adicional."""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Você é um analista de risco especializado em prever reincidência de reclamações."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Extrair JSON da resposta
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            # Ajustar score baseado no peso do canal
            base_score = result.get("risk_score", 0)
            adjusted_score = min(100, int(base_score * 0.6 + channel_risk_score * 0.4))
            result["risk_score"] = adjusted_score
            result["channel_base_score"] = channel_risk_score
            return result
        else:
            return {
                "risk_score": channel_risk_score,
                "risk_level": "Erro",
                "repeat_probability": "N/A",
                "escalation_channels": [],
                "previous_complaints_detected": False,
                "key_indicators": ["Erro ao processar resposta"],
                "urgency_level": "N/A",
                "recommendation": "Revisar manualmente",
                "channel_base_score": channel_risk_score
            }
    except Exception as e:
        st.error(f"Erro na análise: {str(e)}")
        return {
            "risk_score": channel_risk_score,
            "risk_level": "Erro",
            "repeat_probability": "N/A",
            "escalation_channels": [],
            "previous_complaints_detected": False,
            "key_indicators": [str(e)],
            "urgency_level": "N/A",
            "recommendation": "Erro na análise",
            "channel_base_score": channel_risk_score
        }

def process_excel_file(uploaded_file, client):
    """
    Processa o arquivo Excel da planilha "Base Manifestações" e analisa cada linha
    """
    try:
        # Ler planilha "Base Manifestações"
        df = pd.read_excel(uploaded_file, sheet_name='Base Manifestações')
        
        st.info(f"📊 Planilha 'Base Manifestações' carregada: {len(df)} linhas, {len(df.columns)} colunas")
        
        # Identificar colunas importantes
        col_names = df.columns.tolist()
        
        # Coluna de canal (índice 30 = CANAL_DE_ENTRADA_MANIFESTACAO)
        channel_col = col_names[30] if len(col_names) > 30 else None
        
        # Coluna de histórico/texto (geralmente contém "HISTORICO" ou similar)
        text_col = None
        for col in col_names:
            if 'HISTORICO' in str(col).upper() or 'MANIFESTACAO' in str(col).upper() or 'DESCRICAO' in str(col).upper():
                text_col = col
                break
        
        # Se não encontrou, buscar coluna com textos longos
        if text_col is None:
            for col in df.columns:
                if df[col].dtype == 'object':
                    avg_length = df[col].astype(str).str.len().mean()
                    if avg_length > 100:  # Coluna com textos longos
                        text_col = col
                        break
        
        st.write(f"**Coluna de Canal:** `{channel_col}`")
        st.write(f"**Coluna de Texto:** `{text_col}`")
        
        # Mostrar preview
        st.subheader("📋 Preview dos Dados")
        preview_cols = [col for col in ['NR_OCORRENCIA', channel_col, 'TIPO_MANIFESTACAO', 'SITUACAO', text_col] if col in df.columns]
        st.dataframe(df[preview_cols].head(10) if preview_cols else df.head(10))
        
        # Processar cada linha
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in df.iterrows():
            status_text.text(f"Processando linha {idx + 1} de {len(df)}...")
            progress_bar.progress((idx + 1) / len(df))
            
            # Obter canal e texto
            channel_value = row[channel_col] if channel_col else None
            text_value = row[text_col] if text_col else ""
            
            # Classificar risco do canal
            channel_risk, channel_type = classify_channel_risk(channel_value)
            
            # Concatenar informações relevantes para análise
            nr_ocorrencia = row.get('NR_OCORRENCIA', 'N/A')
            tipo_manifestacao = row.get('TIPO_MANIFESTACAO', '')
            situacao = row.get('SITUACAO', '')
            
            full_text = f"Número da Ocorrência: {nr_ocorrencia}\n"
            full_text += f"Tipo: {tipo_manifestacao}\n"
            full_text += f"Situação: {situacao}\n"
            full_text += f"Canal: {channel_value}\n\n"
            full_text += f"Histórico: {text_value}"
            
            if not full_text or len(full_text.strip()) < 20:
                full_text = "Sem informações textuais disponíveis"
            
            # Análise 1: Risco de interna virar externa
            analysis1 = analyze_internal_to_external_risk(client, full_text, channel_type)
            
            # Análise 2: Risco de externa repetir (só para externas)
            if channel_risk > 0:  # É externa
                analysis2 = analyze_external_repeat_risk(client, full_text, channel_type, channel_risk)
            else:  # É interna
                analysis2 = {
                    "risk_score": 0,
                    "risk_level": "N/A (Interna)",
                    "repeat_probability": "N/A",
                    "escalation_channels": [],
                    "previous_complaints_detected": False,
                    "key_indicators": ["Reclamação interna"],
                    "urgency_level": "N/A",
                    "recommendation": "Monitorar para evitar externalização",
                    "channel_base_score": 0
                }
            
            results.append({
                "Linha": idx + 1,
                "NR_OCORRENCIA": nr_ocorrencia,
                "Canal Original": channel_value,
                "Canal Classificado": channel_type,
                "Peso do Canal": channel_risk,
                "Tipo Manifestação": tipo_manifestacao,
                "Situação": situacao,
                
                # Análise 1: Interno → Externo
                "Risco Interno→Externo (0-100)": analysis1.get("risk_score", 0),
                "Nível Risco Int→Ext": analysis1.get("risk_level", "N/A"),
                "Fatores Críticos Int→Ext": ", ".join(analysis1.get("key_factors", [])),
                "Ameaças Detectadas": ", ".join(analysis1.get("detected_threats", [])),
                "Tom Emocional": analysis1.get("emotional_tone", "N/A"),
                
                # Análise 2: Externo → Repetição
                "Risco Repetição Externa (0-100)": analysis2.get("risk_score", 0),
                "Nível Risco Repetição": analysis2.get("risk_level", "N/A"),
                "Probabilidade Repetir": analysis2.get("repeat_probability", "N/A"),
                "Canais de Escalação": ", ".join(analysis2.get("escalation_channels", [])),
                "Reclamações Anteriores": "Sim" if analysis2.get("previous_complaints_detected", False) else "Não",
                "Indicadores Chave": ", ".join(analysis2.get("key_indicators", [])),
                "Urgência": analysis2.get("urgency_level", "N/A"),
                
                # Recomendações
                "Recomendação Int→Ext": analysis1.get("recommendation", "N/A"),
                "Recomendação Repetição": analysis2.get("recommendation", "N/A")
            })
        
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(results)
        
    except Exception as e:
        st.error(f"❌ Erro ao processar arquivo: {str(e)}")
        import traceback
        st.error(f"Detalhes: {traceback.format_exc()}")
        return None

# Interface principal
st.title("⚠️ Análise de Risco de Externalização de Reclamações")
st.markdown("---")

st.markdown("""
### 📊 Como funciona:

Esta ferramenta analisa a planilha **"Base Manifestações"** e gera **duas análises de risco** (0-100):

1. **Risco de Internalização → Externalização**: Probabilidade de reclamações internas virarem externas (ReclameAqui, Procon, Ouvidoria)

2. **Risco de Repetição/Escalação Externa**: Para reclamações já externas, qual o risco de o cliente reclamar novamente ou escalar para outros canais

#### Pesos dos Canais (Coluna CANAL_DE_ENTRADA_MANIFESTACAO):
- **Ext. Ouvidoria**: 100 pontos (🔴 mais crítico)
- **Externo / Web - Reclame Aqui**: 75 pontos (🟠 alto)
- **Externo - Focais**: 50 pontos (🟡 médio)
- **Interno**: 0 pontos (🟢 base para análise)
""")

st.markdown("---")

# Upload de arquivo
uploaded_file = st.file_uploader(
    "📁 Faça upload do Excel do dia (com planilha 'Base Manifestações')",
    type=['xlsx', 'xls'],
    help="Arquivo Excel contendo a planilha 'Base Manifestações' com as reclamações"
)

if uploaded_file is not None:
    st.success("✅ Arquivo carregado com sucesso!")
    
    if st.button("🚀 Iniciar Análise", type="primary"):
        with st.spinner("🔍 Analisando reclamações da planilha 'Base Manifestações'... Isso pode levar alguns minutos."):
            results_df = process_excel_file(uploaded_file, client)
        
        if results_df is not None:
            st.success("✅ Análise concluída!")
            
            # Estatísticas gerais
            st.subheader("📈 Estatísticas Gerais")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_internal_risk = results_df["Risco Interno→Externo (0-100)"].mean()
                st.metric("Risco Médio Int→Ext", f"{avg_internal_risk:.1f}/100")
            
            with col2:
                avg_external_risk = results_df["Risco Repetição Externa (0-100)"].mean()
                st.metric("Risco Médio Repetição", f"{avg_external_risk:.1f}/100")
            
            with col3:
                critical_internal = len(results_df[results_df["Risco Interno→Externo (0-100)"] >= 86])
                st.metric("Casos Críticos Int→Ext", critical_internal)
            
            with col4:
                critical_external = len(results_df[results_df["Risco Repetição Externa (0-100)"] >= 86])
                st.metric("Casos Críticos Repetição", critical_external)
            
            # Distribuição por canal
            st.subheader("📊 Distribuição por Canal")
            col_a, col_b = st.columns(2)
            
            with col_a:
                channel_dist = results_df["Canal Classificado"].value_counts()
                st.bar_chart(channel_dist)
            
            with col_b:
                st.write("**Contagem por Canal:**")
                st.dataframe(channel_dist.reset_index().rename(columns={'index': 'Canal', 'Canal Classificado': 'Quantidade'}))
            
            # Tabela de resultados
            st.subheader("📋 Resultados Detalhados")
            
            # Colorir células baseado no risco
            def color_risk(val):
                if isinstance(val, (int, float)):
                    if val >= 86:
                        return 'background-color: #ff4444; color: white'
                    elif val >= 61:
                        return 'background-color: #ff9944; color: white'
                    elif val >= 31:
                        return 'background-color: #ffdd44; color: black'
                    else:
                        return 'background-color: #44ff44; color: black'
                return ''
            
            styled_df = results_df.style.applymap(
                color_risk,
                subset=["Risco Interno→Externo (0-100)", "Risco Repetição Externa (0-100)"]
            )
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Download dos resultados
            st.subheader("💾 Download dos Resultados")
            
            # Gerar Excel
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"analise_risco_externalizacao_{timestamp}.xlsx"
            
            # Salvar em buffer
            from io import BytesIO
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                results_df.to_excel(writer, index=False, sheet_name='Análise de Risco')
            
            st.download_button(
                label="📥 Baixar Resultados (Excel)",
                data=buffer.getvalue(),
                file_name=output_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # Casos prioritários
            st.subheader("🚨 Casos Prioritários")
            
            priority_cases = results_df[
                (results_df["Risco Interno→Externo (0-100)"] >= 61) | 
                (results_df["Risco Repetição Externa (0-100)"] >= 61)
            ].sort_values(
                by=["Risco Repetição Externa (0-100)", "Risco Interno→Externo (0-100)"],
                ascending=False
            )
            
            if len(priority_cases) > 0:
                st.warning(f"⚠️ {len(priority_cases)} casos requerem atenção prioritária!")
                st.dataframe(
                    priority_cases[["Linha", "NR_OCORRENCIA", "Canal Classificado", 
                                   "Risco Interno→Externo (0-100)", "Risco Repetição Externa (0-100)", 
                                   "Recomendação Int→Ext", "Recomendação Repetição"]],
                    use_container_width=True
                )
            else:
                st.success("✅ Nenhum caso prioritário identificado!")

else:
    st.info("👆 Faça upload de um arquivo Excel para começar a análise")

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>Análise de Risco de Externalização | Powered by OpenAI GPT-4.1-mini</p>
    <p>📊 Planilha analisada: <strong>Base Manifestações</strong></p>
    <p>⚙️ Configure a chave da OpenAI em: Settings > Secrets > OPENAI_API_KEY</p>
</div>
""", unsafe_allow_html=True)
