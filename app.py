"""
PATO QUANT TERMINAL PRO - Versión Refactorizada
Arquitectura modular con separación de responsabilidades
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import pytz
import sys
import os

# No necesitamos agregar paths, usamos imports relativos

from core.state_manager import StateManager, DataProcessor
from core.risk_manager import RiskManager
from ui.chart_builder import ChartBuilder
from market_data import MarketDataFetcher
from technical_analysis import TechnicalAnalyzer
from notifications import NotificationManager
from groq import Groq
from ml_model import TradingMLModel, train_ml_model_for_ticker, get_ml_prediction, format_ml_output

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

st.set_page_config(
    page_title="🦆 Pato Quant Terminal Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar configuración
try:
    if "API_CONFIG" in st.secrets:
        API_CONFIG = st.secrets["API_CONFIG"]
        PORTFOLIO_CONFIG = st.secrets["PORTFOLIO_CONFIG"]
        TECHNICAL_INDICATORS = st.secrets["TECHNICAL_INDICATORS"]
        NOTIFICATIONS = st.secrets.get("NOTIFICATIONS", {})
    else:
        raise Exception("Sin secretos")
except:
    try:
        from config import API_CONFIG, PORTFOLIO_CONFIG, TECHNICAL_INDICATORS, NOTIFICATIONS
    except:
        st.error("❌ Fallo de configuración")
        st.stop()

# Inicializar managers
if 'state_manager' not in st.session_state:
    st.session_state.state_manager = StateManager(cache_ttl_seconds=300)
    st.session_state.risk_manager = RiskManager()
    st.session_state.chart_builder = ChartBuilder()
    st.session_state.fetcher = MarketDataFetcher(API_CONFIG)
    st.session_state.analyzer = TechnicalAnalyzer(TECHNICAL_INDICATORS)
    st.session_state.notifier = NotificationManager({'NOTIFICATIONS': NOTIFICATIONS})
    # 🤖 ESTO DEBE QUEDAR AFUERA (Línea ~60 aprox)
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = {}

state_mgr = st.session_state.state_manager
risk_mgr = st.session_state.risk_manager
chart_builder = st.session_state.chart_builder
fetcher = st.session_state.fetcher
analyzer = st.session_state.analyzer
notifier = st.session_state.notifier

"""
PROMPT MEJORADO DE GROQ - VERSION PRO
Copiar esta función completa y reemplazar la actual en tu app.py
"""

def consultar_ia_groq(ticker, analysis, signals, market_regime, data_processed):
    """
    Versión mejorada con contexto histórico completo y análisis estructurado
    
    Args:
        ticker: Símbolo del activo
        analysis: Análisis técnico completo
        signals: Señales actuales
        market_regime: Contexto macro
        data_processed: DataFrame con todos los indicadores
    """
    try:
        from groq import Groq
        client = Groq(api_key=API_CONFIG['groq_api_key'])
        
        # ====================================================================
        # CALCULAR MÉTRICAS ADICIONALES DEL HISTORIAL
        # ====================================================================
        
        # Performance reciente (últimos 20 días)
        ultimo_mes = data_processed.tail(20)
        retorno_20d = ((ultimo_mes['Close'].iloc[-1] / ultimo_mes['Close'].iloc[0]) - 1) * 100
        
        # Volatilidad anualizada
        volatilidad_20d = ultimo_mes['Returns'].std() * (252 ** 0.5) * 100
        
        # Posición vs SMAs
        precio_vs_sma20 = ((signals['price'] / ultimo_mes['SMA20'].iloc[-1]) - 1) * 100
        precio_vs_sma50 = ((signals['price'] / ultimo_mes['SMA50'].iloc[-1]) - 1) * 100
        
        # Rango de 52 semanas
        datos_52w = data_processed.tail(min(252, len(data_processed)))
        max_52w = datos_52w['High'].max()
        min_52w = datos_52w['Low'].min()
        rango_52w = max_52w - min_52w
        
        # Niveles de Fibonacci
        fib_236 = min_52w + (rango_52w * 0.236)
        fib_382 = min_52w + (rango_52w * 0.382)
        fib_500 = min_52w + (rango_52w * 0.500)
        fib_618 = min_52w + (rango_52w * 0.618)
        fib_786 = min_52w + (rango_52w * 0.786)
        
        # Posición en el rango de 52 semanas
        posicion_en_rango = ((signals['price'] - min_52w) / rango_52w) * 100
        
        # Comparar volatilidad actual vs promedio
        atr_promedio = data_processed['ATR'].tail(50).mean()
        atr_actual = analysis['indicators']['atr']
        volatilidad_ratio = (atr_actual / atr_promedio) if atr_promedio > 0 else 1
        
        # Momentum de 5 días
        retorno_5d = data_processed['Close'].pct_change(5).iloc[-1] * 100
        
        # ====================================================================
        # CONSTRUIR PROMPT ULTRA-DETALLADO
        # ====================================================================
        
        ind = analysis['indicators']
        
        prompt = f"""Actúa como un Senior Quantitative Analyst de un hedge fund institucional.
Analiza {ticker} con datos técnicos profundos en tiempo real:

═══════════════════════════════════════════════════════════
📊 CONTEXTO MACRO Y DE MERCADO
═══════════════════════════════════════════════════════════
• Régimen de Mercado: {market_regime['regime']}
• VIX (Índice de Miedo): {market_regime['vix']:.2f}
• SPY Trend: {market_regime.get('spy_trend', 'N/A')}
• Descripción: {market_regime.get('description', 'N/A')}

═══════════════════════════════════════════════════════════
💰 PRECIO Y PERFORMANCE
═══════════════════════════════════════════════════════════
• Precio Actual: ${signals['price']:.2f}
• Cambio Intraday: {signals['price_change_pct']:+.2f}%
• Performance 20D: {retorno_20d:+.2f}%
• Performance 5D: {retorno_5d:+.2f}%

═══════════════════════════════════════════════════════════
📈 ANÁLISIS DE TENDENCIA
═══════════════════════════════════════════════════════════
• Tendencia Actual: {signals['trend']} ({signals['trend_strength']})
• vs SMA20: {precio_vs_sma20:+.2f}%
• vs SMA50: {precio_vs_sma50:+.2f}%
• ADX (Fuerza): {ind['adx']:.1f}
• Interpretación ADX: {"Tendencia FUERTE" if ind['adx'] > 25 else "Mercado LATERAL"}

═══════════════════════════════════════════════════════════
⚡ MOMENTUM E INDICADORES
═══════════════════════════════════════════════════════════
• RSI(14): {ind['rsi']:.1f} - {"SOBRECOMPRA" if ind['rsi'] > 70 else "SOBREVENTA" if ind['rsi'] < 30 else "NEUTRAL"}
• Stochastic RSI: {ind['stoch_rsi']:.2f} - {"Alto" if ind['stoch_rsi'] > 0.8 else "Bajo" if ind['stoch_rsi'] < 0.2 else "Medio"}
• MACD Histogram: {ind['macd_hist']:.4f} - {"ALCISTA +" if ind['macd_hist'] > 0 else "BAJISTA -"}
• MACD Línea: {ind['macd']:.4f}

═══════════════════════════════════════════════════════════
🌊 VOLATILIDAD Y VOLUMEN
═══════════════════════════════════════════════════════════
• ATR Actual: ${atr_actual:.2f}
• ATR Promedio (50D): ${atr_promedio:.2f}
• Volatilidad Ratio: {volatilidad_ratio:.2f}x {"(ALTA)" if volatilidad_ratio > 1.5 else "(NORMAL)" if volatilidad_ratio > 0.7 else "(BAJA)"}
• Volatilidad Anualizada 20D: {volatilidad_20d:.1f}%
• RVOL (Volumen Relativo): {ind['rvol']:.2f}x - {"Alto" if ind['rvol'] > 1.5 else "Normal" if ind['rvol'] > 0.8 else "Bajo"}
• Posición Bollinger: {signals['bb_position']}

═══════════════════════════════════════════════════════════
🎯 NIVELES TÉCNICOS CLAVE (Fibonacci 52W)
═══════════════════════════════════════════════════════════
• Máximo 52W: ${max_52w:.2f} ({((max_52w - signals['price'])/signals['price']*100):+.1f}% desde actual)
• Fib 78.6%: ${fib_786:.2f} {"← RESISTENCIA" if signals['price'] < fib_786 else "← Superado"}
• Fib 61.8%: ${fib_618:.2f} {"← RESISTENCIA" if signals['price'] < fib_618 else "← Superado"}
• Fib 50.0%: ${fib_500:.2f} {"← MEDIO" if abs(signals['price'] - fib_500) < rango_52w * 0.05 else ""}
• Fib 38.2%: ${fib_382:.2f} {"← SOPORTE" if signals['price'] > fib_382 else "← Roto"}
• Fib 23.6%: ${fib_236:.2f} {"← SOPORTE" if signals['price'] > fib_236 else "← Roto"}
• Mínimo 52W: ${min_52w:.2f} ({((signals['price'] - min_52w)/min_52w*100):+.1f}% desde actual)
• Posición en Rango: {posicion_en_rango:.1f}% {"(Zona ALTA)" if posicion_en_rango > 70 else "(Zona MEDIA)" if posicion_en_rango > 30 else "(Zona BAJA)"}

═══════════════════════════════════════════════════════════
🎲 SISTEMA DE SCORING MULTIFACTORIAL
═══════════════════════════════════════════════════════════
• Score Total: {analysis['signals']['score']}/100
• Recomendación: {analysis['signals']['recommendation']}
• Nivel de Confianza: {analysis['signals']['confidence']}
• Señales de Compra: {len(analysis['signals']['buy_signals'])} activas
• Señales de Venta: {len(analysis['signals']['sell_signals'])} activas

SEÑALES DETECTADAS:
"""
        
        # Agregar señales de compra
        if analysis['signals']['buy_signals']:
            prompt += "\n🟢 COMPRA:\n"
            for signal in analysis['signals']['buy_signals'][:3]:  # Top 3
                prompt += f"  • {signal}\n"
        
        # Agregar señales de venta
        if analysis['signals']['sell_signals']:
            prompt += "\n🔴 VENTA:\n"
            for signal in analysis['signals']['sell_signals'][:3]:  # Top 3
                prompt += f"  • {signal}\n"
        
        # Agregar observaciones neutrales
        if analysis['signals']['neutral_signals']:
            prompt += "\n⚪ OBSERVACIONES:\n"
            for signal in analysis['signals']['neutral_signals'][:2]:  # Top 2
                prompt += f"  • {signal}\n"
        
        prompt += f"""
═══════════════════════════════════════════════════════════
📋 TU ANÁLISIS REQUERIDO (Formato estructurado)
═══════════════════════════════════════════════════════════

### Análisis Técnico de Convergencia
(2-3 líneas) Evalúa si momentum, tendencia y volumen están alineados. Identifica la confluencia o divergencia más importante entre indicadores.

### Validación del Volumen
(2 líneas) ¿El RVOL de {ind['rvol']:.2f}x confirma el movimiento del precio? ¿Hay convicción institucional o es movimiento retail?

### Posicionamiento en Rango
(2 líneas) Con el activo en {posicion_en_rango:.1f}% del rango 52W, evalúa si está cerca de soportes/resistencias clave. Considera los niveles Fibonacci.

### Gestión de Riesgo
(2-3 líneas) Evalúa la volatilidad actual (ATR {volatilidad_ratio:.2f}x vs promedio). Recomienda:
- Stop loss sugerido: Entrada - (ATR × 2) = Entrada - ${atr_actual * 2:.2f}
- Target 1: Entrada + (ATR × 3) = Entrada + ${atr_actual * 3:.2f}
- Tamaño de posición recomendado: ¿Reducir por volatilidad?

### Veredicto Final
(2-3 líneas máximo)
- ACCIÓN: [COMPRA AGRESIVA / COMPRA MODERADA / ESPERAR / VENTA / SIN OPERACIÓN]
- TIMEFRAME: [Intraday / Swing 3-5D / Posición 1-4W]
- CATALYST: ¿Qué evento o nivel técnico validaría/invalidaría la tesis?

IMPORTANTE: 
- Responde en formato markdown con ### para headers
- Usa bullets (•) para listas
- Máximo 400 palabras
- Tono técnico y directo, sin fluff
- Menciona números específicos (niveles de precio)
"""
        
        # ====================================================================
        # LLAMADA A GROQ
        # ====================================================================
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un analista cuantitativo senior con 15 años de experiencia en trading institucional. Respondes de forma estructurada, técnica y accionable. Usas números específicos y niveles de precio concretos."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,  # Más bajo para mayor precisión
            max_tokens=800,   # Más tokens para análisis completo
            top_p=0.9
        )
        
        analisis = completion.choices[0].message.content
        
        # ====================================================================
        # FORMATEAR OUTPUT
        # ====================================================================
        
        output = f"""
🧠 **Análisis Pro de Groq (Llama 3.3)**

---

{analisis}

---

📊 **Contexto de Datos:**
- Performance 20D: {retorno_20d:+.2f}% | Volatilidad: {volatilidad_20d:.1f}%
- Posición en rango 52W: {posicion_en_rango:.1f}%
- ATR: ${atr_actual:.2f} ({volatilidad_ratio:.2f}x vs promedio)
- Score técnico: {analysis['signals']['score']}/100
"""
        
        return output
        
    except Exception as e:
        return f"⚠️ Error en análisis Groq: {str(e)}"

        
# Watchlist management
import json
FILE_PATH = "data/watchlist.json"

def cargar_watchlist():
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r") as f:
            return json.load(f)
    return {"stocks": PORTFOLIO_CONFIG['stocks'], "crypto": PORTFOLIO_CONFIG['crypto']}

def guardar_watchlist(data_dict):
    with open(FILE_PATH, "w") as f:
        json.dump(data_dict, f)

if 'mis_activos' not in st.session_state:
    st.session_state.mis_activos = cargar_watchlist()

# ============================================================================
# SIDEBAR - GESTIÓN DE WATCHLIST
# ============================================================================

st.sidebar.title("🦆 Pato Quant Terminal")
st.sidebar.markdown("---")

st.sidebar.header("🕹️ Gestión de Watchlist")

# Agregar ticker
nuevo = st.sidebar.text_input("Añadir Ticker:").upper()
if st.sidebar.button("➕ Agregar"):
    if nuevo:
        if nuevo not in st.session_state.mis_activos['stocks']:
            st.session_state.mis_activos['stocks'].append(nuevo)
            guardar_watchlist(st.session_state.mis_activos)
            state_mgr.invalidate_cache()  # Limpiar caché
            st.rerun()

# Selector de activo
lista_completa = st.session_state.mis_activos['stocks'] + st.session_state.mis_activos['crypto']
ticker = st.sidebar.selectbox("📊 Activo Seleccionado:", lista_completa)

# Eliminar ticker
if st.sidebar.button("🗑️ Eliminar"):
    for c in ['stocks', 'crypto']:
        if ticker in st.session_state.mis_activos[c]:
            st.session_state.mis_activos[c].remove(ticker)
    guardar_watchlist(st.session_state.mis_activos)
    state_mgr.invalidate_cache(ticker)
    st.rerun()

# ============================================================================
# 🤖 PASO C: MACHINE LEARNING (PÉGALO AQUÍ)
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.header("🤖 Machine Learning")

if st.sidebar.button("🎓 Entrenar Modelo ML"):
    with st.spinner(f"Entrenando modelo para {ticker}..."):
        # Entrenar modelo usando los datos procesados actuales
        model = train_ml_model_for_ticker(ticker, data_processed, prediction_days=5)
        
        if model:
            # Guardar en el baúl que creamos en el paso B
            st.session_state.ml_models[ticker] = model
            st.sidebar.success(f"✅ Modelo entrenado para {ticker}")
            st.sidebar.caption(f"Accuracy: {model.model_metrics['accuracy']*100:.1f}%")
        else:
            st.sidebar.error("❌ Error entrenando modelo")

st.sidebar.markdown("---")
# Configuración de riesgo
st.sidebar.header("⚙️ Configuración de Riesgo")
account_size = st.sidebar.number_input(
    "Capital Total ($)",
    min_value=1000,
    value=10000,
    step=1000
)
risk_pct = st.sidebar.slider(
    "Riesgo por Trade (%)",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5
)

st.sidebar.markdown("---")

# Stats del caché
if st.sidebar.button("🔄 Limpiar Caché"):
    state_mgr.invalidate_cache()
    st.sidebar.success("Caché limpiado")

cache_stats = state_mgr.get_cache_stats()
st.sidebar.caption(f"📊 Caché: {cache_stats['valid_items']}/{cache_stats['total_items']} items válidos")

# ============================================================================
# MAIN AREA - CARGA DE DATOS CON CACHÉ
# ============================================================================

st.title(f"🦆 Análisis de {ticker}")

# Intentar recuperar datos del caché
cached_data = state_mgr.get_cached_data(ticker, 'market_data', period='1y')

if cached_data is not None:
    data = cached_data
    st.caption("✅ Datos recuperados del caché")
else:
    with st.spinner(f"Cargando datos de {ticker}..."):
        data = fetcher.get_portfolio_data([ticker], period='1y')[ticker]
        if not data.empty:
            state_mgr.set_cached_data(ticker, 'market_data', data, period='1y')

if data.empty:
    st.error(f"No se pudieron cargar datos para {ticker}")
    st.stop()

# Pre-procesar datos con TODOS los indicadores
data_processed = DataProcessor.prepare_full_analysis(data, analyzer)

# Análisis técnico completo
analysis = analyzer.analyze_asset(data_processed, ticker)

# Extraer señales actuales
signals = DataProcessor.get_latest_signals(data_processed)

with st.spinner("Analizando contexto macro..."):
    market_regime = fetcher.get_market_regime()

# ============================================================================
# TABS PRINCIPALES
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard Principal",
    "📈 Análisis Técnico Avanzado",
    "💰 Risk Management",
    "🧪 Backtesting Pro",
    "🔍 Scanner Multi-Activo",
    "🤖 Machine Learning"  # ← NUEVO
])

# ============================================================================
# TAB 1: DASHBOARD PRINCIPAL
# ============================================================================

with tab1:
    # Métricas principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Precio",
            f"${signals['price']:.2f}",
            f"{signals['price_change_pct']:.2f}%"
        )
    
    with col2:
        rsi_delta = "Sobrecompra" if signals['rsi'] > 70 else "Sobreventa" if signals['rsi'] < 30 else "Neutral"
        st.metric("RSI", f"{signals['rsi']:.1f}", rsi_delta)
    
    with col3:
        st.metric("ADX", f"{signals['adx']:.1f}", signals['trend_strength'])
    
    with col4:
        st.metric("RVOL", f"{signals['rvol']:.2f}x")
    
    with col5:
        rec = analysis['signals']['recommendation']
        rec_color = "🟢" if "COMPRA" in rec else "🔴" if "VENTA" in rec else "🟡"
        st.metric("Señal", rec, rec_color)
    
    st.markdown("---")
    
    # Gráfico principal
    st.subheader("📊 Análisis Técnico Completo")
    
    fig = chart_builder.create_multi_indicator_chart(
        data_processed,
        ticker,
        show_signals=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    if st.button("🔮 Consultar al Oráculo (Análisis Profundo)"):
        with st.spinner("Analizando contexto histórico..."):
            # Obtener régimen de mercado
            market_regime = fetcher.get_market_regime()
            
            # Llamar función mejorada (ahora con data_processed)
            analisis_ia = consultar_ia_groq(
                ticker=ticker,
                analysis=analysis, 
                signals=signals, 
                market_regime=market_regime,
                data_processed=data_processed  # ← NUEVO PARÁMETRO
            )
            
            st.markdown(analisis_ia)  # Cambiado de st.info a st.markdown
            
    
    # Resumen de señales
    st.markdown("---")
    st.subheader("🎯 Resumen de Señales")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("### ✅ Señales de Compra")
        buy_signals = analysis['signals'].get('buy_signals', [])
        if buy_signals:
            for signal in buy_signals:
                st.markdown(f"- {signal}")
        else:
            st.info("Sin señales de compra")
    
    with col_b:
        st.markdown("### ❌ Señales de Venta")
        sell_signals = analysis['signals'].get('sell_signals', [])
        if sell_signals:
            for signal in sell_signals:
                st.markdown(f"- {signal}")
        else:
            st.info("Sin señales de venta")
    
    with col_c:
        st.markdown("### ↔️ Observaciones")
        neutral_signals = analysis['signals'].get('neutral_signals', [])
        if neutral_signals:
            for signal in neutral_signals:
                st.markdown(f"- {signal}")
        else:
            st.info("Sin observaciones adicionales")
    
    # Score total
    st.markdown("---")
    score = analysis['signals']['score']
    score_color = "green" if score > 0 else "red" if score < 0 else "gray"
    
    st.markdown(f"### 🎯 Score Total: <span style='color:{score_color}; font-size:2em;'>{score}</span>", 
                unsafe_allow_html=True)
    st.caption(f"Confianza: {analysis['signals']['confidence']}")

    # ============================================================================
    # 🤖 PASO D: PREDICCIÓN MACHINE LEARNING (PÉGALO AQUÍ)
    # ============================================================================
    st.markdown("---")
    st.subheader("🤖 Predicción Machine Learning (Próximos 5 días)")
    
    # Verificar si ya entrenaste el modelo para este ticker en el sidebar
    if ticker in st.session_state.ml_models:
        model = st.session_state.ml_models[ticker]
        
        # Obtener la predicción basada en los datos actuales
        ml_prediction = get_ml_prediction(model, data_processed)
        
        if ml_prediction:
            # Diseño de métricas tipo Terminal Profesional
            col_ml1, col_ml2, col_ml3, col_ml4 = st.columns(4)
            
            with col_ml1:
                st.metric(
                    "Prob. Alcista", 
                    f"{ml_prediction['probability_up']*100:.1f}%",
                    delta=f"{(ml_prediction['probability_up'] - 0.5)*100:+.1f}% vs neutro"
                )
            
            with col_ml2:
                st.metric("Prob. Bajista", f"{ml_prediction['probability_down']*100:.1f}%")
            
            with col_ml3:
                # Color dinámico para la confianza
                conf_icon = "🟢" if ml_prediction['confidence'] > 0.7 else "🟡" if ml_prediction['confidence'] > 0.6 else "🔴"
                st.metric("Confianza ML", f"{conf_icon} {ml_prediction['confidence_level']}")
            
            with col_ml4:
                st.metric("Veredicto ML", ml_prediction['recommendation'])
            
            # Análisis profundo y explicabilidad
            with st.expander("📊 Ver Razonamiento del Modelo ML"):
                st.markdown(format_ml_output(ml_prediction, ticker))
                
                # Mostrar qué indicadores pesaron más en esta decisión
                if st.checkbox("Mostrar importancia de indicadores (Features)"):
                    feat_imp = model.get_feature_importance().head(10)
                    st.dataframe(feat_imp, use_container_width=True)
        else:
            st.warning("⚠️ El modelo no pudo generar una predicción con los datos actuales.")
    else:
        # Mensaje amigable si el usuario olvidó entrenar el modelo
        st.info(f"💡 Para ver la predicción de IA, primero haz clic en '🎓 Entrenar Modelo ML' en la barra lateral.")

# ============================================================================
# TAB 2: ANÁLISIS TÉCNICO AVANZADO
# ============================================================================

with tab2:
    st.header("📈 Análisis Técnico Detallado")
    
    regime = market_regime['regime']
    regime_color = "#27ae60" if "ON" in regime else "#e74c3c"
    
    st.markdown(f"""
    <div style='padding: 15px; background-color: {regime_color}20; border-left: 5px solid {regime_color}; margin-bottom: 20px;'>
        <h3 style='margin: 0;'>🌍 Contexto de Mercado: {regime}</h3>
        <p><strong>VIX:</strong> {market_regime['vix']:.2f} | <strong>Tendencia SPY:</strong> {market_regime['spy_trend']}</p>
        <p>{market_regime['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabla de indicadores
    st.subheader("📊 Tabla de Indicadores")
    
    indicators_data = {
        'Indicador': ['RSI', 'Stoch RSI', 'MACD Hist', 'ADX', 'ATR', 'RVOL', 'BB Width'],
        'Valor': [
            f"{signals['rsi']:.2f}",
            f"{signals['stoch_rsi']:.2f}",
            f"{signals['macd_hist']:.4f}",
            f"{signals['adx']:.2f}",
            f"{signals['atr']:.2f}",
            f"{signals['rvol']:.2f}x",
            f"{data_processed['BB_Width'].iloc[-1]*100:.2f}%"
        ],
        'Interpretación': [
            "Sobrecompra" if signals['rsi'] > 70 else "Sobreventa" if signals['rsi'] < 30 else "Neutral",
            "Alto" if signals['stoch_rsi'] > 0.8 else "Bajo" if signals['stoch_rsi'] < 0.2 else "Medio",
            "Alcista" if signals['macd_hist'] > 0 else "Bajista",
            "Tendencia Fuerte" if signals['adx'] > 25 else "Lateral",
            f"${signals['atr']:.2f} por día",
            "Alto" if signals['rvol'] > 1.5 else "Normal",
            "Comprimido" if data_processed['BB_Width'].iloc[-1] < 0.05 else "Normal"
        ]
    }
    
    df_indicators = pd.DataFrame(indicators_data)
    st.dataframe(df_indicators, use_container_width=True, hide_index=True)
    
    # Análisis de volatilidad
    st.markdown("---")
    st.subheader("📉 Análisis de Volatilidad (ATR)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ATR histórico
        fig_atr = go.Figure()
        fig_atr.add_trace(go.Scatter(
            x=data_processed.index,
            y=data_processed['ATR'],
            fill='tozeroy',
            line=dict(color='orange'),
            name='ATR'
        ))
        fig_atr.update_layout(
            title="ATR Histórico",
            template="plotly_dark",
            height=300
        )
        st.plotly_chart(fig_atr, use_container_width=True)
    
    with col2:
        # Distribución de volatilidad
        atr_current = data_processed['ATR'].iloc[-1]
        atr_avg = data_processed['ATR'].mean()
        atr_std = data_processed['ATR'].std()
        
        st.metric("ATR Actual", f"${atr_current:.2f}")
        st.metric("ATR Promedio", f"${atr_avg:.2f}")
        st.metric("Volatilidad vs Promedio", 
                 f"{((atr_current - atr_avg) / atr_avg * 100):.1f}%")

# ============================================================================
# TAB 3: RISK MANAGEMENT
# ============================================================================

with tab3:
    st.header("💰 Gestión de Riesgo Profesional")
    
    # Calcular stops y targets basados en ATR
    current_price = signals['price']
    
    risk_calc = risk_mgr.calculate_atr_stops(
        data_processed,
        entry_price=current_price,
        atr_multiplier_stop=2.0,
        atr_multiplier_target=3.0
    )
    
    # Mostrar niveles
    st.subheader("🎯 Niveles de Entrada/Salida (Basados en ATR)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Entrada", f"${risk_calc['entry_price']:.2f}")
    
    with col2:
        st.metric(
            "Stop Loss",
            f"${risk_calc['stop_loss']:.2f}",
            f"{risk_calc['stop_loss_pct']:.2f}%"
        )
    
    with col3:
        st.metric(
            "Target 1 (2R)",
            f"${risk_calc['take_profit_1']:.2f}",
            f"+{risk_calc['take_profit_1_pct']:.2f}%"
        )
    
    with col4:
        st.metric(
            "Target 2 (4R)",
            f"${risk_calc['take_profit_2']:.2f}",
            f"+{risk_calc['take_profit_2_pct']:.2f}%"
        )
    
    st.markdown("---")
    
    # Risk/Reward
    rr = risk_calc['risk_reward_ratio']
    rr_color = "green" if rr >= 2 else "orange" if rr >= 1.5 else "red"
    
    st.markdown(f"### Risk/Reward Ratio: <span style='color:{rr_color}; font-size:1.5em;'>{rr:.2f}:1</span>",
                unsafe_allow_html=True)
    
    if rr >= 2:
        st.success("✅ Excelente relación riesgo/recompensa")
    elif rr >= 1.5:
        st.warning("⚠️ Relación riesgo/recompensa aceptable")
    else:
        st.error("❌ Relación riesgo/recompensa deficiente - No recomendado")
    
    st.markdown("---")
    
    # Cálculo de posición
    st.subheader("💵 Cálculo de Posición")
    
    position_calc = risk_mgr.calculate_position_size(
        account_size=account_size,
        entry_price=current_price,
        stop_loss=risk_calc['stop_loss'],
        risk_pct=risk_pct
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Acciones a Comprar", f"{position_calc['shares']}")
        st.metric("Valor de Posición", f"${position_calc['position_value']:.2f}")
    
    with col2:
        st.metric("% del Portfolio", f"{position_calc['position_size_pct']:.2f}%")
        st.metric("Riesgo por Acción", f"${position_calc['risk_per_share']:.2f}")
    
    with col3:
        st.metric("Pérdida Máxima", f"${position_calc['max_loss']:.2f}")
        st.metric("% de Pérdida Máx", f"{position_calc['max_loss_pct']:.2f}%")
    
    if position_calc['is_within_limits']:
        st.success("✅ Posición dentro de límites de riesgo")
    else:
        st.error("❌ Posición excede límites - Reducir tamaño")
    
    # Visualización de niveles en gráfico
    st.markdown("---")
    st.subheader("📊 Visualización de Niveles")
    
    fig_levels = go.Figure()
    
    # Precio histórico
    fig_levels.add_trace(go.Scatter(
        x=data_processed.index[-60:],  # Últimos 60 días
        y=data_processed['Close'][-60:],
        name='Precio',
        line=dict(color='white', width=2)
    ))
    
    # Niveles de stop/target
    fig_levels.add_hline(
        y=risk_calc['stop_loss'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Stop Loss: ${risk_calc['stop_loss']:.2f}",
        annotation_position="right"
    )
    
    fig_levels.add_hline(
        y=risk_calc['take_profit_1'],
        line_dash="dash",
        line_color="green",
        annotation_text=f"Target 1: ${risk_calc['take_profit_1']:.2f}",
        annotation_position="right"
    )
    
    fig_levels.add_hline(
        y=risk_calc['take_profit_2'],
        line_dash="dash",
        line_color="lightgreen",
        annotation_text=f"Target 2: ${risk_calc['take_profit_2']:.2f}",
        annotation_position="right"
    )
    
    fig_levels.update_layout(
        title="Niveles de Riesgo/Recompensa",
        template="plotly_dark",
        height=500
    )
    
    st.plotly_chart(fig_levels, use_container_width=True)

# ============================================================================
# TAB 4: BACKTESTING
# ============================================================================

with tab4:
    st.header(f"🧪 Backtesting Profesional: {ticker}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        backtest_capital = st.number_input("Capital Inicial ($)", min_value=1000, value=10000, step=1000)
    with col2:
        take_profit = st.slider("Take Profit (%)", min_value=1.0, max_value=20.0, value=5.0, step=0.5) / 100
    with col3:
        stop_loss = st.slider("Stop Loss (%)", min_value=1.0, max_value=10.0, value=2.0, step=0.5) / 100
    
    if st.button("▶️ Ejecutar Backtest"):
        with st.spinner("Ejecutando simulación con estrategia clásica..."):
            # Variables de simulación
            capital = backtest_capital
            posicion = 0
            precio_compra = 0
            historial_capital = []
            trades = []
            
            # --- ESTRATEGIA RESTAURADA (MODELO ANTERIOR) ---
            for i in range(1, len(data_processed)):
                precio = data_processed['Close'].iloc[i]
                rsi = data_processed['RSI'].iloc[i]
                # Usamos MACD_Hist que es el nombre de columna en tu StateManager
                macd_h = data_processed['MACD_Hist'].iloc[i] 
                
                # 1. Señal de COMPRA (Fórmula original: solo RSI bajo)
                if posicion == 0 and rsi < 35:
                    posicion = capital / precio
                    precio_compra = precio
                    capital = 0
                    
                    trades.append({
                        "Fecha": data_processed.index[i].date(),
                        "Tipo": "🟢 COMPRA",
                        "Precio": round(precio, 2),
                        "Motivo": "RSI Sobrevendido"
                    })
                
                # 2. Señal de VENTA (Fórmula original: TP, SL o MACD debil)
                elif posicion > 0:
                    rendimiento = (precio - precio_compra) / precio_compra
                    
                    # Condiciones de salida exactas del modelo anterior
                    if rendimiento >= take_profit:
                        motivo = f"💰 Take Profit ({rendimiento*100:.1f}%)"
                        vender = True
                    elif rendimiento <= -stop_loss:
                        motivo = f"🛡️ Stop Loss ({rendimiento*100:.1f}%)"
                        vender = True
                    elif macd_h < 0 and rsi > 50:
                        motivo = "📉 Debilidad MACD + RSI"
                        vender = True
                    else:
                        vender = False

                    if vender:
                        capital = posicion * precio
                        trades.append({
                            "Fecha": data_processed.index[i].date(),
                            "Tipo": "🔴 VENTA",
                            "Precio": round(precio, 2),
                            "Motivo": motivo,
                            "P/L %": f"{rendimiento*100:.2f}%"
                        })
                        posicion = 0
                
                # Registrar valor actual del portafolio
                valor_actual = capital if posicion == 0 else posicion * precio
                historial_capital.append(valor_actual)
            
            # --- CÁLCULO DE RESULTADOS FINALES ---
            valor_final = capital if posicion == 0 else posicion * data_processed['Close'].iloc[-1]
            rendimiento_total = ((valor_final - backtest_capital) / backtest_capital) * 100
            
            st.markdown("---")
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Capital Inicial", f"${backtest_capital:,.0f}")
            col_b.metric("Valor Final", f"${valor_final:,.2f}")
            col_c.metric("Rendimiento", f"{rendimiento_total:.2f}%", delta=f"{rendimiento_total:.2f}%")
            col_d.metric("Trades Totales", len(trades))
            
            # Gráfico de evolución
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=data_processed.index[1:], y=historial_capital, fill='tozeroy', line=dict(color='cyan')))
            fig_bt.update_layout(title="Evolución de tu Capital ($)", template="plotly_dark", height=400)
            st.plotly_chart(fig_bt, use_container_width=True)
            
            if trades:
                st.write("### 📜 Bitácora de Operaciones")
                st.dataframe(pd.DataFrame(trades).sort_values(by="Fecha", ascending=False), use_container_width=True, hide_index=True)

# ============================================================================
# TAB 5: SCANNER MULTI-ACTIVO
# ============================================================================

with tab5:
    st.header("🔍 Scanner Maestro de 13 Indicadores")
    
    if st.button("🚀 Iniciar Escaneo de Alta Precisión"):
        resultados = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(lista_completa):
            status_text.text(f"Analizando {symbol}...")
            
            try:
                # Obtener y procesar datos con el StateManager
                symbol_data = fetcher.get_portfolio_data([symbol], period='6mo')[symbol]
                
                if not symbol_data.empty:
                    symbol_processed = DataProcessor.prepare_full_analysis(symbol_data, analyzer)
                    symbol_analysis = analyzer.analyze_asset(symbol_processed, symbol)
                    
                    if symbol_analysis:
                        ind = symbol_analysis['indicators']
                        
                        # Recolección de los 13 indicadores
                        resultados.append({
                            'Ticker': symbol,
                            'Precio': symbol_analysis['price']['current'],
                            'Cambio %': symbol_analysis['price']['change_pct'],
                            'SMA20': symbol_processed['SMA20'].iloc[-1],
                            'SMA50': symbol_processed['SMA50'].iloc[-1],
                            'RSI': ind.get('rsi', 0),
                            'StochRSI': ind.get('stoch_rsi', 0),
                            'ADX': ind.get('adx', 0),
                            'ATR': ind.get('atr', 0),
                            'MACD_H': ind.get('macd_hist', 0),
                            'RVOL': ind.get('rvol', 0),
                            'BB_Up': ind.get('bb_upper', 0),
                            'BB_Low': ind.get('bb_lower', 0),
                            'Score': symbol_analysis['signals']['score'],
                            'Recomendación': symbol_analysis['signals']['recommendation']
                        })
            
            except Exception as e:
                st.warning(f"Error con {symbol}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(lista_completa))
        
        status_text.text("✅ Escaneo completo")
        
        if resultados:
            # Guardamos en session_state para que los datos no se borren al enviar el correo
            st.session_state.scanner_results = pd.DataFrame(resultados).sort_values('Score', ascending=False)

    # MOSTRAR RESULTADOS CON FORMATO DE 2 DECIMALES
    if 'scanner_results' in st.session_state:
        df_res = st.session_state.scanner_results
        st.markdown("---")
        st.subheader(f"📊 Reporte Detallado ({len(df_res)} activos)")
        
        def colorear_recomendacion(val):
            if 'COMPRA' in val: return 'background-color: #27ae60; color: white'
            if 'VENTA' in val: return 'background-color: #e74c3c; color: white'
            return 'background-color: #95a5a6; color: white'
            
        # Aplicamos precisión de 2 decimales a todas las columnas numéricas
        columnas_num = df_res.select_dtypes(include=['float64', 'int64']).columns
        
        st.dataframe(
            df_res.style.applymap(colorear_recomendacion, subset=['Recomendación'])
            .format(precision=2, subset=columnas_num), 
            use_container_width=True, 
            hide_index=True
        )
        
        # Botón de Email corregido para usar los datos guardados
        st.markdown("---")
        if st.button("📧 Enviar Reporte por Email"):
            with st.spinner("Enviando reporte..."):
                macro_info = fetcher.get_market_regime()
                notifier.send_full_report(df_summary=df_res, macro_info=macro_info)
                st.success("✅ ¡Reporte de 13 indicadores enviado!")

with tab6:
    st.header(f"🤖 Machine Learning - {ticker}")
    
    if ticker not in st.session_state.ml_models:
        st.info("ℹ️ No hay modelo entrenado para este ticker.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎓 Entrenar Modelo Ahora", use_container_width=True):
                with st.spinner("Entrenando modelo..."):
                    model = train_ml_model_for_ticker(ticker, data_processed, prediction_days=5)
                    
                    if model:
                        st.session_state.ml_models[ticker] = model
                        st.success("✅ Modelo entrenado!")
                        st.rerun()
    else:
        model = st.session_state.ml_models[ticker]
        
        # Información del modelo
        st.subheader("📊 Información del Modelo")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{model.model_metrics['accuracy']*100:.1f}%")
        
        with col2:
            st.metric("Precision", f"{model.model_metrics['precision']*100:.1f}%")
        
        with col3:
            st.metric("Recall", f"{model.model_metrics['recall']*100:.1f}%")
        
        with col4:
            st.metric("F1-Score", f"{model.model_metrics['f1_score']*100:.1f}%")
        
        st.caption(f"Entrenado el: {model.training_date.strftime('%Y-%m-%d %H:%M:%S')}")
        st.caption(f"Datos de entrenamiento: {model.model_metrics['train_size']} muestras")
        
        st.markdown("---")
        
        # Predicción actual
        st.subheader("🎯 Predicción Actual")
        
        ml_prediction = get_ml_prediction(model, data_processed)
        
        if ml_prediction:
            ml_output = format_ml_output(ml_prediction, ticker)
            st.markdown(ml_output)
            
            st.markdown("---")
            
            # Feature importance
            st.subheader("🏆 Features Más Importantes")
            
            feat_imp = model.get_feature_importance()
            
            # Gráfico de barras
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            top_10 = feat_imp.head(10)
            
            fig.add_trace(go.Bar(
                x=top_10['importance'] * 100,
                y=top_10['feature'],
                orientation='h',
                marker=dict(
                    color=top_10['importance'],
                    colorscale='Viridis'
                )
            ))
            
            fig.update_layout(
                title="Top 10 Features por Importancia",
                xaxis_title="Importancia (%)",
                yaxis_title="Feature",
                height=400,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla completa
            if st.expander("📋 Ver todas las features"):
                st.dataframe(feat_imp, use_container_width=True)
        
        st.markdown("---")
        
        # Opciones de re-entrenamiento
        st.subheader("⚙️ Opciones")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Re-entrenar Modelo"):
                with st.spinner("Re-entrenando..."):
                    model = train_ml_model_for_ticker(ticker, data_processed, prediction_days=5)
                    if model:
                        st.session_state.ml_models[ticker] = model
                        st.success("✅ Modelo re-entrenado!")
                        st.rerun()
        
        with col2:
            if st.button("🗑️ Eliminar Modelo"):
                del st.session_state.ml_models[ticker]
                st.success("✅ Modelo eliminado")
                st.rerun()

# ============================================================================
# MONITOR DE SEÑALES EN TIEMPO REAL (FUERA DE LAS TABS)
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.header("🔔 Alertas Proactivas")
auto_monitor = st.sidebar.checkbox("Activar Monitor en Vivo", value=False)

# El monitor debe ejecutarse siempre que esté activo, sin importar la pestaña
if auto_monitor:
    last_signal_key = f"last_alert_{ticker}"
    current_rec = analysis['signals']['recommendation']
    
    # IMPORTANTE: Todo este bloque debe estar indentado dentro de 'if auto_monitor'
    if "FUERTE" in current_rec:
        if st.session_state.get(last_signal_key) != current_rec:
            with st.sidebar:
                with st.spinner("Enviando alerta en tiempo real..."):
                    # Llamada al gestor de notificaciones modular
                    notifier.send_signal_alert(ticker, analysis)
                    st.session_state[last_signal_key] = current_rec
                    st.toast(f"🚀 Alerta enviada: {ticker} - {current_rec}")
                    st.success(f"🔔 Alerta de {current_rec} emitida.")
    else:
        st.sidebar.info("🛰️ Monitoreando... Esperando señal fuerte.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption(f"""
🦆 Pato Quant Terminal Pro v2.0 | 
📊 {len(lista_completa)} activos monitoreados | 
⏱️ Última actualización: {datetime.now(pytz.timezone('America/Monterrey')).strftime('%d/%m/%Y %H:%M:%S')} (Monterrey)
""")

