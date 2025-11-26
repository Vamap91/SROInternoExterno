import streamlit as st
import pandas as pd
from openai import OpenAI
import json
from datetime import datetime
import re
from io import BytesIO
import time

st.set_page_config(
    page_title="Análise de Risco de Externalização - Base Manifestações",
    page_icon="⚠️",
    layout="wide"
)

# Configurar OpenAI API usando secrets do Streamlit
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error("⚠️ Erro ao configurar OpenAI API. Verifique se a chave está configurada em Settings > Secrets do Streamlit.")
    st.stop()

def classify_internal_risk(score):
    """Classifica risco interno (0-100) de forma granular"""
    if score >= 75:
        return "🔴 RISCO ALTO DE EXTERNALIZAR"
    else:
        lower = (score // 5) * 5
        upper = lower + 5
        if upper > 74:
            upper = 74
        return f"{lower}-{upper} pts"

def classify_external_risk(score):
    """Classifica risco externo (100-1000)"""
    if score >= 851:
        return "🔴 Vai Reclamar Novamente"
    elif score >= 701:
        return "🟠 Muito Alto"
    elif score >= 501:
        return "🟡 Alto"
    elif score >= 301:
        return "🟢 Médio"
    else:
        return "⚪ Baixo"

def classify_channel_type(channel_value):
    """Classifica o canal como Interno ou Externo"""
    if pd.isna(channel_value):
        return "Interno", 0
    
    channel_str = str(channel_value).strip().lower()
    
    # Externos
    if "ouvidoria" in channel_str:
        return "Externo", 100
    elif "reclame aqui" in channel_str or "reclameaqui" in channel_str:
        return "Externo", 75
    elif "focais" in channel_str or "externo - focais" in channel_str:
        return "Externo", 50
    elif "externo" in channel_str:
        return "Externo", 75
    
    # Internos
    else:
        return "Interno", 0

def analyze_internal_risk(client, text, nr_ocorrencia="N/A"):
    """EIXO 1: Análise de risco de reclamações INTERNAS virarem EXTERNAS (0-100 pontos)"""
    
    prompt = f"""Você é um analista preditivo especializado em prever o risco de reclamações internas se tornarem externas.

CONTEXTO:
Esta é uma reclamação INTERNA (NR_OCORRENCIA: {nr_ocorrencia})

TEXTO DA RECLAMAÇÃO:
{text}

TAREFA:
Analise o texto e calcule o risco (0-100 pontos) de esta reclamação INTERNA se tornar EXTERNA (ReclameAqui, Procon, Ouvidoria).

METODOLOGIA DE ANÁLISE (EIXO 1):

Fatores Preditivos e Pesos:

1. FREQUÊNCIA DE CONTATOS – Peso 4 (máximo 40 pontos)
   - 1 contato: 0 pts
   - 2 contatos: 5 pts
   - 3+ contatos: 10 pts

2. TEMPO DE ESPERA / ATRASOS – Peso 3 (máximo 30 pontos)
   - Menção a atrasos: +10 pts
   - Menção a "dias", "semanas" de espera: +10 pts
   - Menção a prazos não cumpridos: +10 pts

3. FALHAS OPERACIONAIS – Peso 2 (máximo 20 pontos)
   - Indícios técnicos graves: 10 pts cada
   - Falhas de processo: 5 pts cada

4. ESTADO EMOCIONAL – Peso 1 (máximo 10 pontos)
   - Termos negativos moderados: 1 pt cada
   - Termos de risco jurídico: 3 pts cada
   - Termos positivos: -1 pt cada

REGRA ESPECIAL: Negativas técnicas sem insatisfação = máximo 30 pontos

FORMATO DE SAÍDA (JSON):
{{
    "risk_score": <número de 0 a 100>,
    "frequency_score": <0-40>,
    "delay_score": <0-30>,
    "operational_score": <0-20>,
    "emotional_score": <0-10>,
    "key_factors": ["fator1", "fator2"],
    "detected_threats": ["ameaça1", "ameaça2"],
    "emotional_tone": "<descrição>",
    "is_technical_negative": <true/false>,
    "recommendation": "<recomendação>"
}}

Retorne APENAS o JSON."""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Você é um analista preditivo especializado."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        
        result_text = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        
        if json_match:
            return json.loads(json_match.group())
        else:
            return create_error_result("Erro ao processar resposta")
            
    except Exception as e:
        return create_error_result(str(e))

def analyze_external_risk(client, text, nr_ocorrencia="N/A", channel_base_score=50):
    """EIXO 2: Análise de risco de reclamações EXTERNAS serem ESCALADAS/REPETIDAS (100-1000 pontos)"""
    
    prompt = f"""Você é um analista preditivo especializado em prever escalação de reclamações externas.

CONTEXTO:
Esta é uma reclamação EXTERNA (NR_OCORRENCIA: {nr_ocorrencia})
Peso base do canal: {channel_base_score} pontos

TEXTO DA RECLAMAÇÃO:
{text}

TAREFA:
Analise o texto e calcule o risco (100-1000 pontos) de o cliente ESCALAR ou RECLAMAR NOVAMENTE.

METODOLOGIA DE ANÁLISE (EIXO 2):

1. INDICADORES TEXTUAIS – Peso 5 (máximo 500 pontos)
   - Menções a canais externos: 100 pts cada
   - Palavras emocionais críticas: 30 pts cada
   - Ameaças diretas: 150 pts cada
   - Padrões comportamentais: até 150 pts

2. INSATISFAÇÃO ANTERIOR – Peso 3 (máximo 300 pontos)
   - "Não resolveram": +250 pts
   - "Voltou a acontecer": +200 pts
   - "Já reclamei antes": +150 pts

3. GRAVIDADE DO CANAL – Peso 2 (máximo 200 pontos)
   - Baseado no peso base do canal

FORMATO DE SAÍDA (JSON):
{{
    "risk_score": <número de 100 a 1000>,
    "external_indicators_score": <0-500>,
    "previous_dissatisfaction_score": <0-300>,
    "channel_gravity_score": <0-200>,
    "channel_base_score": {channel_base_score},
    "repeat_probability": "<Baixa/Média/Alta/Muito Alta/Certeza>",
    "escalation_channels": ["canal1", "canal2"],
    "previous_complaints_detected": <true/false>,
    "behavioral_patterns": ["padrão1", "padrão2"],
    "key_indicators": ["indicador1", "indicador2"],
    "urgency_level": "<Baixa/Média/Alta/Urgente>",
    "recommendation": "<recomendação>"
}}

Retorne APENAS o JSON."""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Você é um analista preditivo especializado."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        result_text = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        
        if json_match:
            result = json.loads(json_match.group())
            score = result.get("risk_score", 100)
            if score < 100:
                score = 100 + score
            result["risk_score"] = min(1000, score)
            return result
        else:
            return create_error_result_external(channel_base_score)
            
    except Exception as e:
        return create_error_result_external(channel_base_score, str(e))

def create_error_result(error_msg):
    """Resultado de erro para análise interna"""
    return {
        "risk_score": 0,
        "frequency_score": 0,
        "delay_score": 0,
        "operational_score": 0,
        "emotional_score": 0,
        "key_factors": [error_msg],
        "detected_threats": [],
        "emotional_tone": "N/A",
        "is_technical_negative": False,
        "recommendation": "Revisar manualmente"
    }

def create_error_result_external(channel_base, error_msg="Erro na análise"):
    """Resultado de erro para análise externa"""
    return {
        "risk_score": 100 + channel_base,
        "external_indicators_score": 0,
        "previous_dissatisfaction_score": 0,
        "channel_gravity_score": 0,
        "channel_base_score": channel_base,
        "repeat_probability": "N/A",
        "escalation_channels": [],
        "previous_complaints_detected": False,
        "behavioral_patterns": [],
        "key_indicators": [error_msg],
        "urgency_level": "N/A",
        "recommendation": "Revisar manualmente"
    }

def process_internals_only(uploaded_file, client):
    """Processa APENAS reclamações INTERNAS"""
    try:
        df = pd.read_excel(uploaded_file, sheet_name='Base Manifestações')
        
        col_names = df.columns.tolist()
        channel_col = col_names[30] if len(col_names) > 30 else None
        
        text_col = None
        for col in col_names:
            if 'HISTORICO' in str(col).upper() or 'MANIFESTACAO' in str(col).upper():
                text_col = col
                break
        
        if text_col is None:
            for col in df.columns:
                if df[col].dtype == 'object':
                    avg_length = df[col].astype(str).str.len().mean()
                    if avg_length > 100:
                        text_col = col
                        break
        
        # Filtrar apenas INTERNOS
        df_filtered = df.copy()
        df_filtered['_channel_type'] = df_filtered[channel_col].apply(lambda x: classify_channel_type(x)[0])
        df_internos = df_filtered[df_filtered['_channel_type'] == 'Interno'].copy()
        
        st.info(f"📊 Total de linhas: {len(df)} | **Internos: {len(df_internos)}** | Externos: {len(df) - len(df_internos)}")
        
        if len(df_internos) == 0:
            st.warning("⚠️ Nenhuma reclamação interna encontrada!")
            return None
        
        # Processar
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        times_per_row = []
        
        for idx, (orig_idx, row) in enumerate(df_internos.iterrows()):
            try:
                row_start = time.time()
                
                # Calcular tempo previsto
                if idx > 0:
                    avg_time_per_row = sum(times_per_row) / len(times_per_row)
                    remaining_rows = len(df_internos) - (idx + 1)
                    estimated_seconds = remaining_rows * avg_time_per_row
                    estimated_minutes = int(estimated_seconds / 60)
                    
                    if estimated_minutes > 0:
                        status_text.text(f"Processando INTERNO {idx + 1} de {len(df_internos)}... (tempo previsto: {estimated_minutes} minutos)")
                    else:
                        estimated_secs = int(estimated_seconds)
                        status_text.text(f"Processando INTERNO {idx + 1} de {len(df_internos)}... (tempo previsto: {estimated_secs} segundos)")
                else:
                    status_text.text(f"Processando INTERNO {idx + 1} de {len(df_internos)}... (calculando tempo previsto...)")
                
                progress_bar.progress((idx + 1) / len(df_internos))
                
                channel_value = row[channel_col] if channel_col else None
                text_value = row[text_col] if text_col else ""
                
                nr_ocorrencia = row.get('NR_OCORRENCIA', 'N/A')
                tipo_manifestacao = row.get('TIPO_MANIFESTACAO', '')
                situacao = row.get('SITUACAO', '')
                
                full_text = f"Número: {nr_ocorrencia}\nTipo: {tipo_manifestacao}\nSituação: {situacao}\nCanal: {channel_value}\n\nHistórico: {text_value}"
                
                # Análise INTERNA
                analysis = analyze_internal_risk(client, full_text, nr_ocorrencia)
                score = analysis.get("risk_score", 0)
                classification = classify_internal_risk(score)
                
                results.append({
                    "Linha Original": orig_idx + 1,
                    "NR_OCORRENCIA": nr_ocorrencia,
                    "Canal": channel_value,
                    "Tipo Manifestação": tipo_manifestacao,
                    "Situação": situacao,
                    "Pontuação": score,
                    "Classificação": classification,
                    "Score Frequência": analysis.get("frequency_score", 0),
                    "Score Atraso": analysis.get("delay_score", 0),
                    "Score Operacional": analysis.get("operational_score", 0),
                    "Score Emocional": analysis.get("emotional_score", 0),
                    "Fatores Críticos": ", ".join(analysis.get("key_factors", [])),
                    "Ameaças Detectadas": ", ".join(analysis.get("detected_threats", [])),
                    "Tom Emocional": analysis.get("emotional_tone", "N/A"),
                    "Negativa Técnica?": "Sim" if analysis.get("is_technical_negative", False) else "Não",
                    "Recomendação": analysis.get("recommendation", "N/A")
                })
                
                row_end = time.time()
                times_per_row.append(row_end - row_start)
                
            except Exception as e:
                st.warning(f"⚠️ Erro na linha {orig_idx + 1}: {str(e)}")
                continue
        
        total_time = time.time() - start_time
        total_minutes = int(total_time / 60)
        total_seconds = int(total_time % 60)
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Análise de INTERNOS concluída em {total_minutes}min {total_seconds}s")
        
        return pd.DataFrame(results)
        
    except Exception as e:
        st.error(f"❌ Erro: {str(e)}")
        import traceback
        st.error(f"Detalhes: {traceback.format_exc()}")
        return None

def process_externals_only(uploaded_file, client):
    """Processa APENAS reclamações EXTERNAS"""
    try:
        df = pd.read_excel(uploaded_file, sheet_name='Base Manifestações')
        
        col_names = df.columns.tolist()
        channel_col = col_names[30] if len(col_names) > 30 else None
        
        text_col = None
        for col in col_names:
            if 'HISTORICO' in str(col).upper() or 'MANIFESTACAO' in str(col).upper():
                text_col = col
                break
        
        if text_col is None:
            for col in df.columns:
                if df[col].dtype == 'object':
                    avg_length = df[col].astype(str).str.len().mean()
                    if avg_length > 100:
                        text_col = col
                        break
        
        # Filtrar apenas EXTERNOS
        df_filtered = df.copy()
        df_filtered['_channel_info'] = df_filtered[channel_col].apply(classify_channel_type)
        df_filtered['_channel_type'] = df_filtered['_channel_info'].apply(lambda x: x[0])
        df_filtered['_channel_base'] = df_filtered['_channel_info'].apply(lambda x: x[1])
        df_externos = df_filtered[df_filtered['_channel_type'] == 'Externo'].copy()
        
        st.info(f"📊 Total de linhas: {len(df)} | Internos: {len(df) - len(df_externos)} | **Externos: {len(df_externos)}**")
        
        if len(df_externos) == 0:
            st.warning("⚠️ Nenhuma reclamação externa encontrada!")
            return None
        
        # Processar
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        times_per_row = []
        
        for idx, (orig_idx, row) in enumerate(df_externos.iterrows()):
            try:
                row_start = time.time()
                
                # Calcular tempo previsto
                if idx > 0:
                    avg_time_per_row = sum(times_per_row) / len(times_per_row)
                    remaining_rows = len(df_externos) - (idx + 1)
                    estimated_seconds = remaining_rows * avg_time_per_row
                    estimated_minutes = int(estimated_seconds / 60)
                    
                    if estimated_minutes > 0:
                        status_text.text(f"Processando EXTERNO {idx + 1} de {len(df_externos)}... (tempo previsto: {estimated_minutes} minutos)")
                    else:
                        estimated_secs = int(estimated_seconds)
                        status_text.text(f"Processando EXTERNO {idx + 1} de {len(df_externos)}... (tempo previsto: {estimated_secs} segundos)")
                else:
                    status_text.text(f"Processando EXTERNO {idx + 1} de {len(df_externos)}... (calculando tempo previsto...)")
                
                progress_bar.progress((idx + 1) / len(df_externos))
                
                channel_value = row[channel_col] if channel_col else None
                text_value = row[text_col] if text_col else ""
                channel_base = row['_channel_base']
                
                nr_ocorrencia = row.get('NR_OCORRENCIA', 'N/A')
                tipo_manifestacao = row.get('TIPO_MANIFESTACAO', '')
                situacao = row.get('SITUACAO', '')
                
                full_text = f"Número: {nr_ocorrencia}\nTipo: {tipo_manifestacao}\nSituação: {situacao}\nCanal: {channel_value}\n\nHistórico: {text_value}"
                
                # Análise EXTERNA
                analysis = analyze_external_risk(client, full_text, nr_ocorrencia, channel_base)
                score = analysis.get("risk_score", 100)
                classification = classify_external_risk(score)
                
                results.append({
                    "Linha Original": orig_idx + 1,
                    "NR_OCORRENCIA": nr_ocorrencia,
                    "Canal": channel_value,
                    "Tipo Manifestação": tipo_manifestacao,
                    "Situação": situacao,
                    "Pontuação": score,
                    "Classificação": classification,
                    "Score Indicadores Externos": analysis.get("external_indicators_score", 0),
                    "Score Insatisfação Anterior": analysis.get("previous_dissatisfaction_score", 0),
                    "Score Gravidade Canal": analysis.get("channel_gravity_score", 0),
                    "Peso Base Canal": analysis.get("channel_base_score", channel_base),
                    "Probabilidade Repetir": analysis.get("repeat_probability", "N/A"),
                    "Padrões Comportamentais": ", ".join(analysis.get("behavioral_patterns", [])),
                    "Canais de Escalação": ", ".join(analysis.get("escalation_channels", [])),
                    "Reclamações Anteriores": "Sim" if analysis.get("previous_complaints_detected", False) else "Não",
                    "Indicadores Chave": ", ".join(analysis.get("key_indicators", [])),
                    "Urgência": analysis.get("urgency_level", "N/A"),
                    "Recomendação": analysis.get("recommendation", "N/A")
                })
                
                row_end = time.time()
                times_per_row.append(row_end - row_start)
                
            except Exception as e:
                st.warning(f"⚠️ Erro na linha {orig_idx + 1}: {str(e)}")
                continue
        
        total_time = time.time() - start_time
        total_minutes = int(total_time / 60)
        total_seconds = int(total_time % 60)
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Análise de EXTERNOS concluída em {total_minutes}min {total_seconds}s")
        
        return pd.DataFrame(results)
        
    except Exception as e:
        st.error(f"❌ Erro: {str(e)}")
        import traceback
        st.error(f"Detalhes: {traceback.format_exc()}")
        return None

# Interface principal
st.title("⚠️ Análise de Risco de Externalização - Base Manifestações")
st.markdown("**Sistema com Metodologia SRO Dual Avançada - Análises Separadas**")
st.markdown("---")

st.markdown("""
### 📊 Metodologia de Análise:

#### 🟢 **INTERNOS: 0-100 pontos** (Risco de virar externo)
- **0-74 pontos**: Classificação granular (0-5, 5-10... 70-74)
- **75-100 pontos**: 🔴 **RISCO ALTO DE EXTERNALIZAR**

#### 🔴 **EXTERNOS: 100-1000 pontos** (Risco de escalação/repetição)
- **100-300**: ⚪ Baixo
- **301-500**: 🟢 Médio
- **501-700**: 🟡 Alto
- **701-850**: 🟠 Muito Alto
- **851-1000**: 🔴 **Vai Reclamar Novamente**

### 💡 **Estratégia de Processamento:**
Para evitar timeout, as análises foram separadas em dois botões:
1. **Analisar INTERNOS** → Gera Excel com internos
2. **Analisar EXTERNOS** → Gera Excel com externos

Você pode processar um de cada vez e depois juntar os resultados!
""")

st.markdown("---")

# Upload
uploaded_file = st.file_uploader(
    "📁 Faça upload do Excel do dia (com planilha 'Base Manifestações')",
    type=['xlsx', 'xls'],
    help="Arquivo Excel contendo a planilha 'Base Manifestações'"
)

if uploaded_file is not None:
    st.success("✅ Arquivo carregado!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🟢 Analisar INTERNOS (0-100)", type="primary", use_container_width=True):
            with st.spinner("🔍 Analisando reclamações INTERNAS..."):
                results_df = process_internals_only(uploaded_file, client)
            
            if results_df is not None:
                st.success("✅ Análise de INTERNOS concluída!")
                
                # Estatísticas
                st.subheader("📈 Estatísticas - INTERNOS")
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Total Internos", len(results_df))
                
                with col_b:
                    avg_score = results_df["Pontuação"].mean()
                    st.metric("Pontuação Média", f"{avg_score:.1f}/100")
                
                with col_c:
                    criticos = len(results_df[results_df["Pontuação"] >= 75])
                    st.metric("Casos Críticos (≥75)", criticos)
                
                # Resultados
                st.subheader("📋 Resultados - INTERNOS")
                st.dataframe(results_df, use_container_width=True, height=400)
                
                # Download
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"analise_internos_{timestamp}.xlsx"
                
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    results_df.to_excel(writer, index=False, sheet_name='Internos')
                
                st.download_button(
                    label="📥 Baixar Resultados INTERNOS (Excel)",
                    data=buffer.getvalue(),
                    file_name=output_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    with col2:
        if st.button("🔴 Analisar EXTERNOS (100-1000)", type="secondary", use_container_width=True):
            with st.spinner("🔍 Analisando reclamações EXTERNAS..."):
                results_df = process_externals_only(uploaded_file, client)
            
            if results_df is not None:
                st.success("✅ Análise de EXTERNOS concluída!")
                
                # Estatísticas
                st.subheader("📈 Estatísticas - EXTERNOS")
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Total Externos", len(results_df))
                
                with col_b:
                    avg_score = results_df["Pontuação"].mean()
                    st.metric("Pontuação Média", f"{avg_score:.0f}/1000")
                
                with col_c:
                    criticos = len(results_df[results_df["Pontuação"] >= 851])
                    st.metric("Vai Reclamar (≥851)", criticos)
                
                # Resultados
                st.subheader("📋 Resultados - EXTERNOS")
                st.dataframe(results_df, use_container_width=True, height=400)
                
                # Download
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"analise_externos_{timestamp}.xlsx"
                
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    results_df.to_excel(writer, index=False, sheet_name='Externos')
                
                st.download_button(
                    label="📥 Baixar Resultados EXTERNOS (Excel)",
                    data=buffer.getvalue(),
                    file_name=output_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

else:
    st.info("👆 Faça upload de um arquivo Excel para começar a análise")

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p><strong>Análise de Risco SRO Dual Avançada</strong> | Powered by OpenAI GPT-4.1-mini</p>
    <p>📊 Metodologia: INTERNOS (0-100 granular) | EXTERNOS (100-1000 com 5 níveis)</p>
    <p>⚙️ Configure OPENAI_API_KEY em Settings > Secrets</p>
    <p>💡 Análises separadas para evitar timeout do Streamlit Cloud</p>
</div>
""", unsafe_allow_html=True)
