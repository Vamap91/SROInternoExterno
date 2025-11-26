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

def classify_channel_type(channel_value):
    """
    Classifica o canal como Interno ou Externo
    
    Pesos dos canais externos:
    - Ext. Ouvidoria: 100 pontos
    - Externo / Web - Reclame Aqui: 75 pontos
    - Externo - Focais: 50 pontos
    """
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
    """
    EIXO 1: Análise de risco de reclamações INTERNAS virarem EXTERNAS (0-100 pontos)
    
    Usa a metodologia completa do código original SRO
    """
    
    prompt = f"""Você é um analista preditivo especializado em prever o risco de reclamações internas se tornarem externas.

CONTEXTO:
Esta é uma reclamação INTERNA (NR_OCORRENCIA: {nr_ocorrencia})

TEXTO DA RECLAMAÇÃO:
{text}

TAREFA:
Analise o texto e calcule o risco (0-100 pontos) de esta reclamação INTERNA se tornar EXTERNA (ReclameAqui, Procon, Ouvidoria).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METODOLOGIA DE ANÁLISE (EIXO 1 - INTERNALIZAÇÃO → EXTERNALIZAÇÃO)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fatores Preditivos e Pesos:

1. FREQUÊNCIA DE CONTATOS – Peso 4 (máximo 40 pontos)
   - 1 contato: 0 pts (risco baixo)
   - 2 contatos: 5 pts (risco médio)
   - 3+ contatos: 10 pts (risco elevado)
   
   Atenuação: Se múltiplos contatos contêm palavras neutras de acompanhamento, reduzir pontos.
   
   Palavras neutras: fila, data, equipe, atualização, agenda, recontato, inserido, tabela, negociado, complemento, evento, telefone, inicial, observação, pergunta, item, escala, criação, responsável, cancelado, negativa, técnica, cobertura, atendimento

2. TEMPO DE ESPERA / ATRASOS – Peso 3 (máximo 30 pontos)
   - Menção a atrasos: +10 pts
   - Menção a "dias", "semanas" de espera: +10 pts
   - Menção a prazos não cumpridos: +10 pts

3. FALHAS OPERACIONAIS – Peso 2 (máximo 20 pontos)
   
   A. Indícios técnicos graves (10 pts cada):
      - defeito, conserto, danos, sinistro, vazamento, barulho, quebra
      - arranhado, sujo, manchado, escorrida, descolado, solto
      - acendendo, parou, sumiu, faltando, faltou, errado, errada
      - incompleto, danificado, estragado, pior, voltou
      - torto, amassado, mal feito, falta de cuidado
   
   B. Falhas de processo (5 pts cada):
      - cadastro incorreto, solicitações não atendidas
      - falhas de comunicação, problemas técnicos pós-serviço
      - cada atendente dá informação diferente

4. ESTADO EMOCIONAL – Peso 1 (máximo 10 pontos)
   
   Termos negativos moderados (1 pt cada):
   - terrível, péssimo, horrível, decepcionado, frustrado
   - reclamar, problema, erro, falha, demora, demorado
   - insatisfeito, revoltado, indignado, absurdo, inaceitável
   
   Termos de risco jurídico (3 pts cada):
   - processar, advogado, jurídico, procon, denúncia
   - órgão, fiscalização, consumidor, direito, prejuízo
   
   Termos positivos (reduzem -1 pt cada):
   - excelente, ótimo, perfeito, maravilhoso, fantástico
   - agradecer, obrigado, parabéns, satisfeito, contente
   - recomendo, eficiente, rápido, atencioso, prestativo

REGRA ESPECIAL - Negativas Técnicas:
Se o texto contém apenas negativa técnica/cancelamento SEM insatisfação explícita do cliente:
→ Score máximo = 30 pontos (Baixo)

Para elevar acima de 30, deve haver:
- Manifestação direta de descontentamento
- Termos emocionais negativos do cliente
- Questionamento da decisão técnica
- Ameaças ou menções a órgãos externos
- Múltiplos contatos com tom de cobrança

CÁLCULO FINAL:
1. Atribua score (0-10) para cada fator
2. Multiplique pelo peso do fator
3. Some os valores ponderados (máximo 100)
4. Aplique regra especial se for negativa técnica
5. Classifique:
   - Baixo: 0-30 pontos
   - Médio: 31-60 pontos
   - Alto: 61-85 pontos
   - Crítico: 86-100 pontos

FORMATO DE SAÍDA (JSON):
{{
    "risk_score": <número de 0 a 100>,
    "risk_level": "<Baixo/Médio/Alto/Crítico>",
    "frequency_score": <0-40>,
    "delay_score": <0-30>,
    "operational_score": <0-20>,
    "emotional_score": <0-10>,
    "key_factors": ["fator1", "fator2", "fator3"],
    "detected_threats": ["ameaça1", "ameaça2"],
    "emotional_tone": "<descrição do tom emocional>",
    "is_technical_negative": <true/false>,
    "recommendation": "<recomendação de ação>"
}}

Retorne APENAS o JSON, sem texto adicional."""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Você é um analista preditivo especializado em prever externalizações de reclamações usando metodologia ponderada."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        result_text = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        
        if json_match:
            return json.loads(json_match.group())
        else:
            return create_error_result("Erro ao processar resposta da IA")
            
    except Exception as e:
        return create_error_result(str(e))

def analyze_external_risk(client, text, nr_ocorrencia="N/A", channel_base_score=50):
    """
    EIXO 2: Análise de risco de reclamações EXTERNAS serem ESCALADAS/REPETIDAS (100-1000 pontos)
    
    Usa a metodologia completa do código original SRO para externalização
    Base: 100-1000 pontos (10x a escala original para dar mais granularidade)
    """
    
    prompt = f"""Você é um analista preditivo especializado em prever escalação e reincidência de reclamações externas.

CONTEXTO:
Esta é uma reclamação EXTERNA (NR_OCORRENCIA: {nr_ocorrencia})
Peso base do canal: {channel_base_score} pontos

TEXTO DA RECLAMAÇÃO:
{text}

TAREFA:
Analise o texto e calcule o risco (0-1000 pontos) de o cliente ESCALAR ou RECLAMAR NOVAMENTE.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METODOLOGIA DE ANÁLISE (EIXO 2 - EXTERNALIZAÇÃO E ESCALAÇÃO)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fatores de Externalização e Escalação:

1. INDICADORES TEXTUAIS DE EXTERNALIZAÇÃO – Peso 5 (máximo 500 pontos)

   A. Menções Explícitas a Canais Externos (100 pts cada):
      - "reclame aqui", "reclameaqui" → +100 pts
      - "procon" → +100 pts
      - "advogado", "jurídico", "processar" → +100 pts cada
      - "ouvidoria" (da seguradora) → +80 pts
      - "google", "avaliar", "avaliação" → +50 pts cada
   
   B. Palavras Emocionais Críticas (30 pts cada):
      - "absurdo", "inaceitável", "prejuízo"
      - "indignado", "revoltado", "insatisfeito", "furioso"
   
   C. Escalação Progressiva:
      - 2+ palavras-chave de externalização → +100 pts bônus
      - 3+ palavras-chave → +200 pts bônus
   
   D. Frases de Ameaça Direta (150 pts cada):
      - "vou publicar", "vou denunciar", "vou processar", "vou ao procon"
   
   E. PADRÕES COMPORTAMENTAIS DE ESCALAÇÃO (150 pts máximo):
      
      Menção a corretor/corretora:
      - "vou falar com meu corretor", "meu corretor vai saber" → +80 pts
      
      Ameaça de acionar seguradora:
      - "vou ligar na seguradora", "vou acionar o SAC" → +100 pts
      - "vou falar com a [Porto/Bradesco/Azul/etc]" → +100 pts
      
      Múltiplos canais de contato:
      - 2 canais (telefone + email) → +50 pts
      - 3+ canais (telefone + email + WhatsApp) → +100 pts
      
      Redes sociais:
      - "vou expor nas redes sociais" → +80 pts
      - "vou postar no Facebook/Instagram/Twitter" → +70 pts
      - "vou fazer um vídeo" → +100 pts
      
      Ultimatos:
      - "é a última vez que ligo", "última oportunidade" → +80 pts
      - "se não resolver até [data]", "prazo de X dias" → +80 pts
      - "já tentei X vezes" → +50 pts
      
      Frustração com processo:
      - "já falei com X atendentes diferentes" → +70 pts
      - "cada um me dá uma informação diferente" → +60 pts
      - "ninguém resolve nada", "não consigo solução" → +80 pts
      - "estou há X dias tentando resolver" → +60 pts

2. INSATISFAÇÃO COM RESOLUÇÃO ANTERIOR – Peso 3 (máximo 300 pontos)
   - "Não resolveram", "continua o problema" → +250 pts
   - "Mesma situação de antes", "voltou a acontecer" → +200 pts
   - "Já reclamei antes" → +150 pts
   - Múltiplas reclamações anteriores → +300 pts

3. GRAVIDADE DO CANAL ATUAL – Peso 2 (máximo 200 pontos)
   - Ext. Ouvidoria (100 pts base): já crítico, risco jurídico → +200 pts
   - Web - Reclame Aqui (75 pts base): risco Procon/jurídico → +150 pts
   - Externo (75 pts base): risco canais formais → +150 pts
   - Externo - Focais (50 pts base): risco ReclameAqui/Procon → +100 pts

CÁLCULO FINAL:
1. Some todos os pontos dos fatores acima
2. Adicione o peso base do canal
3. Resultado: 0-1000 pontos (escala ampliada para melhor granularidade)
4. Classifique:
   - Baixo: 100-300 pontos
   - Médio: 301-500 pontos
   - Alto: 501-750 pontos
   - Crítico: 751-1000 pontos

FORMATO DE SAÍDA (JSON):
{{
    "risk_score": <número de 0 a 1000>,
    "risk_level": "<Baixo/Médio/Alto/Crítico>",
    "external_indicators_score": <0-500>,
    "previous_dissatisfaction_score": <0-300>,
    "channel_gravity_score": <0-200>,
    "channel_base_score": {channel_base_score},
    "repeat_probability": "<Baixa/Média/Alta/Muito Alta>",
    "escalation_channels": ["canal1", "canal2"],
    "previous_complaints_detected": <true/false>,
    "behavioral_patterns": ["padrão1", "padrão2"],
    "key_indicators": ["indicador1", "indicador2"],
    "urgency_level": "<Baixa/Média/Alta/Urgente>",
    "recommendation": "<recomendação de ação>"
}}

Retorne APENAS o JSON, sem texto adicional."""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Você é um analista preditivo especializado em prever escalação de reclamações externas usando metodologia ponderada avançada."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1200
        )
        
        result_text = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        
        if json_match:
            result = json.loads(json_match.group())
            # Garantir que está na escala 100-1000
            score = result.get("risk_score", 100)
            if score < 100:
                score = 100 + score  # Ajustar para mínimo de 100
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
        "risk_level": "Erro",
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
        "risk_level": "Erro",
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

def process_excel_file(uploaded_file, client):
    """
    Processa o arquivo Excel da planilha "Base Manifestações"
    """
    try:
        df = pd.read_excel(uploaded_file, sheet_name='Base Manifestações')
        
        st.info(f"📊 Planilha 'Base Manifestações' carregada: {len(df)} linhas, {len(df.columns)} colunas")
        
        col_names = df.columns.tolist()
        
        # Identificar colunas
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
        
        st.write(f"**Coluna de Canal:** `{channel_col}`")
        st.write(f"**Coluna de Texto:** `{text_col}`")
        
        # Preview
        st.subheader("📋 Preview dos Dados")
        preview_cols = [col for col in ['NR_OCORRENCIA', channel_col, 'TIPO_MANIFESTACAO', 'SITUACAO'] if col in df.columns]
        st.dataframe(df[preview_cols].head(10) if preview_cols else df.head(10))
        
        # Processar
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        times_per_row = []
        
        for idx, row in df.iterrows():
            row_start = time.time()
            
            # Calcular tempo previsto
            if idx > 0:
                avg_time_per_row = sum(times_per_row) / len(times_per_row)
                remaining_rows = len(df) - (idx + 1)
                estimated_seconds = remaining_rows * avg_time_per_row
                estimated_minutes = int(estimated_seconds / 60)
                
                if estimated_minutes > 0:
                    status_text.text(f"Processando linha {idx + 1} de {len(df)}... (tempo previsto: {estimated_minutes} minutos)")
                else:
                    estimated_secs = int(estimated_seconds)
                    status_text.text(f"Processando linha {idx + 1} de {len(df)}... (tempo previsto: {estimated_secs} segundos)")
            else:
                status_text.text(f"Processando linha {idx + 1} de {len(df)}... (calculando tempo previsto...)")
            
            progress_bar.progress((idx + 1) / len(df))
            
            channel_value = row[channel_col] if channel_col else None
            text_value = row[text_col] if text_col else ""
            
            channel_type, channel_base = classify_channel_type(channel_value)
            
            nr_ocorrencia = row.get('NR_OCORRENCIA', 'N/A')
            tipo_manifestacao = row.get('TIPO_MANIFESTACAO', '')
            situacao = row.get('SITUACAO', '')
            
            full_text = f"Número: {nr_ocorrencia}\nTipo: {tipo_manifestacao}\nSituação: {situacao}\nCanal: {channel_value}\n\nHistórico: {text_value}"
            
            if channel_type == "Interno":
                # Análise INTERNA: 0-100 pontos
                analysis = analyze_internal_risk(client, full_text, nr_ocorrencia)
                
                results.append({
                    "Linha": idx + 1,
                    "NR_OCORRENCIA": nr_ocorrencia,
                    "Canal Original": channel_value,
                    "Tipo": channel_type,
                    "Tipo Manifestação": tipo_manifestacao,
                    "Situação": situacao,
                    
                    # Análise Interna (0-100)
                    "Risco (0-100 ou 100-1000)": analysis.get("risk_score", 0),
                    "Nível de Risco": analysis.get("risk_level", "N/A"),
                    "Score Frequência": analysis.get("frequency_score", 0),
                    "Score Atraso": analysis.get("delay_score", 0),
                    "Score Operacional": analysis.get("operational_score", 0),
                    "Score Emocional": analysis.get("emotional_score", 0),
                    "Fatores Críticos": ", ".join(analysis.get("key_factors", [])),
                    "Ameaças Detectadas": ", ".join(analysis.get("detected_threats", [])),
                    "Tom Emocional": analysis.get("emotional_tone", "N/A"),
                    "Negativa Técnica?": "Sim" if analysis.get("is_technical_negative", False) else "Não",
                    "Recomendação": analysis.get("recommendation", "N/A"),
                    
                    # Campos vazios para externos
                    "Padrões Comportamentais": "N/A (Interno)",
                    "Canais de Escalação": "N/A (Interno)",
                    "Reclamações Anteriores": "N/A (Interno)",
                    "Urgência": "N/A (Interno)"
                })
                
            else:  # Externo
                # Análise EXTERNA: 100-1000 pontos
                analysis = analyze_external_risk(client, full_text, nr_ocorrencia, channel_base)
                
                results.append({
                    "Linha": idx + 1,
                    "NR_OCORRENCIA": nr_ocorrencia,
                    "Canal Original": channel_value,
                    "Tipo": channel_type,
                    "Tipo Manifestação": tipo_manifestacao,
                    "Situação": situacao,
                    
                    # Análise Externa (100-1000)
                    "Risco (0-100 ou 100-1000)": analysis.get("risk_score", 100),
                    "Nível de Risco": analysis.get("risk_level", "N/A"),
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
                    "Recomendação": analysis.get("recommendation", "N/A"),
                    
                    # Campos vazios para internos
                    "Score Frequência": "N/A (Externo)",
                    "Score Atraso": "N/A (Externo)",
                    "Score Operacional": "N/A (Externo)",
                    "Score Emocional": "N/A (Externo)",
                    "Fatores Críticos": ", ".join(analysis.get("key_indicators", [])),
                    "Ameaças Detectadas": ", ".join(analysis.get("escalation_channels", [])),
                    "Tom Emocional": "N/A (Externo)",
                    "Negativa Técnica?": "N/A (Externo)"
                })
            
            # Registrar tempo da linha
            row_end = time.time()
            times_per_row.append(row_end - row_start)
        
        # Tempo total
        total_time = time.time() - start_time
        total_minutes = int(total_time / 60)
        total_seconds = int(total_time % 60)
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Processamento concluído em {total_minutes}min {total_seconds}s")
        
        return pd.DataFrame(results)
        
    except Exception as e:
        st.error(f"❌ Erro ao processar arquivo: {str(e)}")
        import traceback
        st.error(f"Detalhes: {traceback.format_exc()}")
        return None

# Interface principal
st.title("⚠️ Análise de Risco de Externalização - Base Manifestações")
st.markdown("**Sistema com Metodologia SRO Dual Avançada**")
st.markdown("---")

st.markdown("""
### 📊 Metodologia de Análise:

Esta ferramenta usa a **metodologia SRO dual avançada** para analisar a planilha "Base Manifestações":

#### 🟢 **INTERNOS: 0-100 pontos** (Risco de virar externo)

**Fatores Ponderados:**
1. **Frequência de Contatos** (Peso 4) - até 40 pts
2. **Tempo de Espera/Atrasos** (Peso 3) - até 30 pts
3. **Falhas Operacionais** (Peso 2) - até 20 pts
4. **Estado Emocional** (Peso 1) - até 10 pts

**Classificação:**
- 0-30: 🟢 Baixo
- 31-60: 🟡 Médio
- 61-85: 🟠 Alto
- 86-100: 🔴 Crítico

#### 🔴 **EXTERNOS: 100-1000 pontos** (Risco de escalação/repetição)

**Fatores Ponderados:**
1. **Indicadores Textuais de Externalização** (Peso 5) - até 500 pts
2. **Insatisfação com Resolução Anterior** (Peso 3) - até 300 pts
3. **Gravidade do Canal Atual** (Peso 2) - até 200 pts

**Padrões Comportamentais Detectados:**
- Menção a corretor/seguradora
- Múltiplos canais de contato
- Ameaças a redes sociais
- Ultimatos e prazos
- Frustração com processo interno

**Classificação:**
- 100-300: 🟢 Baixo
- 301-500: 🟡 Médio
- 501-750: 🟠 Alto
- 751-1000: 🔴 Crítico

#### Pesos dos Canais Externos:
- **Ext. Ouvidoria**: 100 pontos base
- **Externo / Web - Reclame Aqui**: 75 pontos base
- **Externo - Focais**: 50 pontos base
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
    
    if st.button("🚀 Iniciar Análise Dual", type="primary"):
        with st.spinner("🔍 Analisando com metodologia SRO dual... Isso pode levar alguns minutos."):
            results_df = process_excel_file(uploaded_file, client)
        
        if results_df is not None:
            st.success("✅ Análise concluída!")
            
            # Estatísticas
            st.subheader("📈 Estatísticas Gerais")
            
            internos = results_df[results_df["Tipo"] == "Interno"]
            externos = results_df[results_df["Tipo"] == "Externo"]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total de Casos", len(results_df))
            
            with col2:
                st.metric("Casos Internos", len(internos))
                if len(internos) > 0:
                    avg_int = internos["Risco (0-100 ou 100-1000)"].mean()
                    st.caption(f"Risco médio: {avg_int:.1f}/100")
            
            with col3:
                st.metric("Casos Externos", len(externos))
                if len(externos) > 0:
                    avg_ext = externos["Risco (0-100 ou 100-1000)"].mean()
                    st.caption(f"Risco médio: {avg_ext:.0f}/1000")
            
            with col4:
                criticos_int = len(internos[internos["Risco (0-100 ou 100-1000)"] >= 86])
                criticos_ext = len(externos[externos["Risco (0-100 ou 100-1000)"] >= 751])
                st.metric("Casos Críticos", criticos_int + criticos_ext)
                st.caption(f"Int: {criticos_int} | Ext: {criticos_ext}")
            
            # Distribuição
            st.subheader("📊 Distribuição por Tipo")
            col_a, col_b = st.columns(2)
            
            with col_a:
                type_dist = results_df["Tipo"].value_counts()
                st.bar_chart(type_dist)
            
            with col_b:
                st.write("**Contagem:**")
                st.dataframe(type_dist.reset_index().rename(columns={'index': 'Tipo', 'Tipo': 'Quantidade'}))
            
            # Resultados
            st.subheader("📋 Resultados Detalhados")
            
            def color_risk(val):
                if isinstance(val, (int, float)):
                    if val >= 751 or (val < 100 and val >= 86):  # Crítico
                        return 'background-color: #ff4444; color: white'
                    elif val >= 501 or (val < 100 and val >= 61):  # Alto
                        return 'background-color: #ff9944; color: white'
                    elif val >= 301 or (val < 100 and val >= 31):  # Médio
                        return 'background-color: #ffdd44; color: black'
                    else:  # Baixo
                        return 'background-color: #44ff44; color: black'
                return ''
            
            styled_df = results_df.style.applymap(
                color_risk,
                subset=["Risco (0-100 ou 100-1000)"]
            )
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Download
            st.subheader("💾 Download dos Resultados")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"analise_risco_sro_dual_{timestamp}.xlsx"
            
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                results_df.to_excel(writer, index=False, sheet_name='Análise de Risco SRO')
            
            st.download_button(
                label="📥 Baixar Resultados (Excel)",
                data=buffer.getvalue(),
                file_name=output_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # Casos prioritários
            st.subheader("🚨 Casos Prioritários")
            
            priority_int = internos[internos["Risco (0-100 ou 100-1000)"] >= 61]
            priority_ext = externos[externos["Risco (0-100 ou 100-1000)"] >= 501]
            priority_cases = pd.concat([priority_int, priority_ext]).sort_values(
                by="Risco (0-100 ou 100-1000)", ascending=False
            )
            
            if len(priority_cases) > 0:
                st.warning(f"⚠️ {len(priority_cases)} casos requerem atenção prioritária!")
                st.dataframe(
                    priority_cases[["Linha", "NR_OCORRENCIA", "Tipo", "Canal Original",
                                   "Risco (0-100 ou 100-1000)", "Nível de Risco", "Recomendação"]],
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
    <p><strong>Análise de Risco SRO Dual Avançada</strong> | Powered by OpenAI GPT-4.1-mini</p>
    <p>📊 Metodologia: INTERNOS (0-100) | EXTERNOS (100-1000)</p>
    <p>⚙️ Configure OPENAI_API_KEY em Settings > Secrets</p>
</div>
""", unsafe_allow_html=True)
