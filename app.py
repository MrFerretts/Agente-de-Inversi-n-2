

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
from portfolio_tracker import PortfolioTracker, display_portfolio_dashboard
from auto_monitoring import AutoMonitoringSystem, setup_auto_monitoring, display_monitoring_controls
from consensus_analyzer import ConsensusAnalyzer, get_consensus_analysis
from auto_trader import AutoTrader, AlpacaConnector, SafetyManager
from realtime_streamer import RealTimeStreamer, init_realtime_streamer

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

st.set_page_config(
    page_title="🦆 Pato Quant Terminal Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ESTILO VISUAL: DARK CRYPTO DASHBOARD
# ============================================================================
st.markdown("""
<style>
    .stApp { background-color: #0e0e0e; color: #e0e0e0; }
    .css-1r6slb0, .st-emotion-cache-1r6slb0, .st-emotion-cache-10trblm {
        background-color: #1a1a1a; border-radius: 12px; padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); border: 1px solid #2d2d2d;
    }
    .metric-card {
        background-color: #1a1a1a; border-radius: 10px; padding: 15px;
        border: 1px solid #2d2d2d; text-align: center;
    }

    /* El círculo alrededor del número */
    .metric-circle {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 90px;  /* Ancho fijo */
        height: 90px; /* Alto igual al ancho para que sea un círculo perfecto */
        border-radius: 50%; /* Esto hace la magia del círculo */
        background-color: #222; /* Un tono ligeramente más claro que la tarjeta */
        border: 2px solid #333; /* El anillo exterior */
        margin: 10px auto; /* Centrado horizontalmente */
        box-shadow: inset 0 0 10px rgba(0,0,0,0.5); /* Sombra interna para profundidad */
    }
    
    [data-testid="stMetricValue"] { font-size: 24px; font-weight: 700; color: #ffffff; }
    .stButton>button { border-radius: 8px; font-weight: 600; border: none; }
    .buy-button>button { background-color: #00c853; color: white; }
    .sell-button>button { background-color: #d32f2f; color: white; }
</style>
""", unsafe_allow_html=True)

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

# Portfolio Tracker
if 'portfolio_tracker' not in st.session_state:
    st.session_state.portfolio_tracker = PortfolioTracker(data_file="data/portfolio.json")

portfolio_tracker = st.session_state.portfolio_tracker



state_mgr = st.session_state.state_manager
risk_mgr = st.session_state.risk_manager
chart_builder = st.session_state.chart_builder
fetcher = st.session_state.fetcher
analyzer = st.session_state.analyzer
notifier = st.session_state.notifier

# --- MOVIDO HACIA ARRIBA PARA EVITAR NAMEERROR ---
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

# Definimos lista_completa AQUÍ para que el Streamer pueda leerla
lista_completa = st.session_state.mis_activos['stocks'] + st.session_state.mis_activos['crypto']

# ============================================================================
# INICIALIZAR AUTO-TRADER (CONEXIÓN ALPACA)
# ============================================================================
if 'auto_trader' not in st.session_state:
    try:
        # Busca las llaves en tus secrets de Streamlit
        alpaca_sec = st.secrets.get("ALPACA", {})
        if alpaca_sec:
            st.session_state.auto_trader = AutoTrader(
                alpaca_api_key=alpaca_sec.get("api_key"),
                alpaca_secret=alpaca_sec.get("api_secret"),
                consensus_analyzer=ConsensusAnalyzer(),
                portfolio_tracker=portfolio_tracker,
                paper_trading=alpaca_sec.get("paper_trading", True)
            )
        else:
            st.session_state.auto_trader = None
    except Exception as e:
        st.error(f"⚠️ Error inicializando Alpaca: {str(e)}")
        st.session_state.auto_trader = None

# Definir la variable global para que la Tab 8 la reconozca
auto_trader = st.session_state.auto_trader

# Inicializar (después de auto_trader)
if 'realtime_streamer' not in st.session_state:
    symbols = lista_completa[:5]  # Primeros 5
# Extraemos las llaves del bloque [ALPACA] que ya definimos arriba
    alpaca_sec = st.secrets.get("ALPACA", {})
    
    st.session_state.realtime_streamer = init_realtime_streamer(
        st, 
        alpaca_sec.get("api_key"), 
        alpaca_sec.get("api_secret"), 
        symbols, 
        paper=True
    )

# ============================================================================
# UI HELPER: CREAR TARJETAS MÉTRICAS
# ============================================================================
def crear_metric_card(titulo, valor, delta):
    color = "#00c853" if "+" in str(delta) or "COMPRA" in str(valor) else "#d32f2f"
    flecha = "↑" if color == "#00c853" else "↓"
    st.markdown(f"""
    <div class="metric-card">
        <p style="color: #a0a0a0; font-size: 14px; margin-bottom: 5px;">{titulo}</p>
        <h3 style="color: #ffffff; margin: 0; font-size: 26px;">{valor}</h3>
        <p style="color: {color}; font-size: 14px; margin-top: 5px;">{flecha} {delta}</p>
    </div>
    """, unsafe_allow_html=True)


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

def analizar_backtest_con_ia(ticker, resultados, trades):
    """
    Usa Llama 3.3 para realizar una autopsia profesional de los resultados del backtest.
    """
    try:
        from groq import Groq
        client = Groq(api_key=API_CONFIG['groq_api_key'])
        
        # Resumen de los trades para que la IA no se pierda en datos infinitos
        ultimos_trades = str(trades[-10:]) if trades else "Sin trades realizados"
        
        prompt = f"""
        Actúa como un Head of Trading analizando el desempeño de un algoritmo en {ticker}.
        
        RESULTADOS DEL BACKTEST:
        - Capital Inicial: ${resultados['inicial']:.2f}
        - Valor Final: ${resultados['final']:.2f}
        - Rendimiento Total: {resultados['rendimiento']:.2f}%
        - Número de Trades: {resultados['n_trades']}
        
        MUESTRA DE OPERACIONES:
        {ultimos_trades}
        
        TAREA:
        1. Explica brevemente por qué la estrategia tuvo éxito o fracasó en este activo.
        2. Analiza si el 'Motivo' de salida más común (TP o SL) sugiere que los parámetros están bien calibrados.
        3. Da una recomendación específica para mejorar el rendimiento (ej: ajustar el RSI, mover el Stop Loss, etc.).
        4. Tono crítico, constructivo y muy técnico. Máximo 3 párrafos.
        """
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", #
            messages=[{"role": "system", "content": "Eres un mentor de trading cuantitativo."},
                      {"role": "user", "content": prompt}],
            temperature=0.4
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ No se pudo generar la bitácora de IA: {str(e)}"

def consultar_ia_riesgo(ticker, risk_calc, position_calc, market_regime, ml_prediction=None):
    """
    Analiza la gestión de riesgo técnica vs el contexto macro e IA.
    """
    try:
        from groq import Groq
        client = Groq(api_key=API_CONFIG['groq_api_key'])
        
        ml_context = f"Prob. Alcista: {ml_prediction['probability_up']*100:.1f}% | Confianza: {ml_prediction['confidence_level']}" if ml_prediction else "No disponible"
        
        prompt = f"""
        Actúa como un Chief Risk Officer (CRO) de un fondo de cobertura.
        Analiza el riesgo para una posición en {ticker}:

        DATOS TÉCNICOS:
        - Stop Loss: ${risk_calc['stop_loss']:.2f} ({risk_calc['stop_loss_pct']:.2f}%)
        - Risk/Reward: {risk_calc['risk_reward_ratio']:.2f}:1
        - Tamaño sugerido: {position_calc['shares']} acciones (${position_calc['position_value']:.2f})

        CONTEXTO MACRO:
        - Régimen: {market_regime['regime']}
        - VIX: {market_regime['vix']:.2f}
        - Tendencia SPY: {market_regime['spy_trend']}

        INTELIGENCIA ARTIFICIAL:
        - {ml_context}

        TAREA:
        1. ¿El riesgo del {position_calc['max_loss_pct']:.2f}% es adecuado para este entorno?
        2. ¿Deberíamos ajustar el tamaño de la posición basado en el VIX y la probabilidad de la IA?
        3. Da un veredicto: [RIESGO ACEPTADO / REDUCIR POSICIÓN / CANCELAR OPERACIÓN]
        Responde en 3 párrafos cortos y directos.
        """
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Eres un experto en gestión de riesgos financieros."},
                      {"role": "user", "content": prompt}],
            temperature=0.3
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error en AI Risk Officer: {str(e)}"

def generar_top_picks_ia(df_scanner):
    """
    Analiza el dataframe del scanner y selecciona los 3 activos con mejor setup.
    """
    try:
        from groq import Groq
        client = Groq(api_key=API_CONFIG['groq_api_key'])
        
        # Tomamos los top 10 por score para que la IA elija entre lo mejor
        top_10_data = df_scanner.head(10).to_dict(orient='records')
        
        prompt = f"""
        Actúa como un Portfolio Manager analizando un escaneo de mercado.
        Aquí tienes los 10 activos con mejor Score Técnico hoy:
        {top_10_data}

        TAREA:
        1. Selecciona los 3 activos con el setup más explosivo (combina RSI, MACD y ADX).
        2. Explica brevemente el "Catalizador Técnico" de cada uno.
        3. Da un precio objetivo estimado (Target) basado en la volatilidad actual.
        
        Formato: Usa negritas para los Tickers y bullets para los puntos. 
        Máximo 3 párrafos en total.
        """
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Eres un experto en selección de activos cuantitativos."},
                      {"role": "user", "content": prompt}],
            temperature=0.4
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ No se pudo generar la selección: {str(e)}"
        
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
cached_data = state_mgr.get_cached_data(ticker, 'market_data', period='5y') # 👈 CAMBIA A 5y

if cached_data is not None:
    data = cached_data
else:
    with st.spinner(f"Descargando historial profundo de {ticker}..."):
        data = fetcher.get_portfolio_data([ticker], period='5y')[ticker] # 👈 CAMBIA A 5y
        if not data.empty:
            state_mgr.set_cached_data(ticker, 'market_data', data, period='5y')

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Dashboard Principal",
    "📈 Análisis Técnico Avanzado",
    "💰 Risk Management",
    "🧪 Backtesting Pro",
    "🔍 Scanner Multi-Activo",
    "🤖 Machine Learning",
    "💼 Mi Portfolio",
    "🤖 Auto-Trading"  # ← NUEVO
])

# ============================================================================
# TAB 1: DASHBOARD PRINCIPAL (Lógica Completa + Diseño Pro)
# ============================================================================
with tab1:
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: 
        # Verificamos si hay conexión con el flujo de datos de tu Mac
        if st.session_state.get('realtime_streamer'):
            # Obtenemos el último precio del ticker seleccionado
            price_live = st.session_state.realtime_streamer.get_latest_price(ticker)
            
            # Mostramos la tarjeta con el diseño de círculo y texto LIVE
            crear_metric_card("Precio LIVE", f"${price_live:.2f}", "STREAMING")
        else:
            # Si el streamer está apagado, mostramos el dato normal
            crear_metric_card("Precio", f"${signals['price']:.2f}", f"{signals['price_change_pct']:+.2f}%")
            crear_metric_card("Precio", f"${signals['price']:.2f}", f"{signals['price_change_pct']:+.2f}%")
    with c2: crear_metric_card("RSI", f"{signals['rsi']:.1f}", "Sobrecompra" if signals['rsi'] > 70 else "Neutral")
    with c3: crear_metric_card("ADX", f"{signals['adx']:.1f}", signals['trend_strength'])
    with c4: crear_metric_card("RVOL", f"{signals['rvol']:.2f}x", "Alto" if signals['rvol'] > 1.5 else "Normal")
    with c5: crear_metric_card("Señal", analysis['signals']['recommendation'], f"Score: {analysis['signals']['score']}")

    st.markdown("---")

    # 2. CUERPO: Gráfico (Izquierda) + Inteligencia Artificial (Derecha)
    col_main, col_side = st.columns([2.2, 1])

    with col_main:
        st.subheader(f"📊 Análisis Técnico: {ticker}")
        fig = chart_builder.create_multi_indicator_chart(data_processed, ticker, show_signals=False)
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Botón de Groq original
        if st.button("🔮 Consultar al Oráculo (Análisis Profundo)", use_container_width=True):
            with st.spinner("IA analizando contexto histórico..."):
                analisis_ia = consultar_ia_groq(ticker, analysis, signals, market_regime, data_processed)
                # Guardar en session_state para usar en Consensus
                st.session_state.last_groq_analysis = analisis_ia
                st.markdown(analisis_ia)

        st.markdown("---")
        # Resumen de señales original
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.write("### ✅ Señales de Compra")
            for s in analysis['signals'].get('buy_signals', [])[:4]: st.success(f"↑ {s}")
        with col_s2:
            st.write("### ❌ Señales de Venta")
            for s in analysis['signals'].get('sell_signals', [])[:4]: st.error(f"↓ {s}")

    with col_side:
        st.subheader("🧠 Modelos de IA")
        
        # --- SECCIÓN ML TRADICIONAL (Código intacto) ---
        if ticker in st.session_state.ml_models:
            model = st.session_state.ml_models[ticker]
            ml_prediction = get_ml_prediction(model, data_processed)
            if ml_prediction:
                st.markdown(f"""<div class="ia-panel">
                    <p style="margin:0; font-size:12px; color:#aaa;">RANDOM FOREST (ML)</p>
                    <h4 style="margin:0;">{ml_prediction['recommendation']}</h4>
                    <p style="margin:0; color:#ffcc00;">Prob. Alcista: {ml_prediction['probability_up']*100:.1f}%</p>
                </div>""", unsafe_allow_html=True)
                
                with st.expander("📊 Razonamiento ML"):
                    st.markdown(format_ml_output(ml_prediction, ticker))
                    if st.checkbox("Ver Features Importantes", key="feat_dash"):
                        st.dataframe(model.get_feature_importance().head(5), use_container_width=True)
        else:
            st.info("💡 Entrena ML en el sidebar")

        st.write("") # Espaciador

        # --- SECCIÓN LSTM DEEP LEARNING (Código intacto) ---
        lstm_key = f"{ticker}_lstm"
        if lstm_key in st.session_state.ml_models:
            lstm_model = st.session_state.ml_models[lstm_key]
            try:
                lstm_pred = lstm_model.predict(data_processed)
                st.markdown(f"""<div class="ia-panel" style="border-left-color: #00c853;">
                    <p style="margin:0; font-size:12px; color:#aaa;">DEEP LEARNING (LSTM)</p>
                    <h4 style="margin:0;">{lstm_pred['recommendation']}</h4>
                    <p style="margin:0; color:#00c853;">Prob. LSTM: {lstm_pred['probability_up']*100:.1f}%</p>
                </div>""", unsafe_allow_html=True)
                
                # Comparativa inteligente original
                if ticker in st.session_state.ml_models:
                    trad_pred = st.session_state.ml_models[ticker].predict(data_processed)
                    diff = lstm_pred['probability_up'] - trad_pred['probability_up']
                    if abs(diff) > 0.10:
                        st.warning(f"⚠️ Divergencia detectada: LSTM es {'más optimista' if diff > 0 else 'más pesimista'}")
                    else:
                        st.success("✅ Modelos en confluencia")
            except Exception as e:
                st.error(f"Error LSTM: {str(e)}")
        else:
            st.info("💡 Entrena LSTM en el sidebar")

        # Score Total original
        st.markdown("---")
        score = analysis['signals']['score']
        score_color = "green" if score > 0 else "red"
        st.markdown(f"### 🎯 Score Técnico: <span style='color:{score_color};'>{score}</span>", unsafe_allow_html=True)
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
    # 🧠 VISUALIZACIÓN LSTM (COMPARACIÓN DE MODELOS)
    # ============================================================================
    st.markdown("---")
    st.subheader("🧠 Predicción LSTM (Deep Learning)")

    lstm_key = f"{ticker}_lstm"
    if lstm_key in st.session_state.ml_models:
        lstm_model = st.session_state.ml_models[lstm_key]
        
        try:
            # Obtener predicción de Deep Learning
            lstm_pred = lstm_model.predict(data_processed)
            
            if lstm_pred:
                c_l1, c_l2, c_l3, c_l4 = st.columns(4)
                with c_l1:
                    st.metric("🧠 Prob. LSTM", f"{lstm_pred['probability_up']*100:.1f}%", 
                              delta=f"{(lstm_pred['probability_up'] - 0.5)*100:+.1f}%")
                with c_l2:
                    st.metric("Prob. Bajista", f"{lstm_pred['probability_down']*100:.1f}%")
                with c_l3:
                    c_icon = "🟢" if lstm_pred['confidence'] > 0.7 else "🟡"
                    st.metric("Confianza LSTM", f"{c_icon} {lstm_pred['confidence_level']}")
                with c_l4:
                    st.metric("Veredicto LSTM", lstm_pred['recommendation'])
                
                # Comparativa inteligente
                if ticker in st.session_state.ml_models:
                    trad_pred = st.session_state.ml_models[ticker].predict(data_processed)
                    if abs(lstm_pred['probability_up'] - trad_pred['probability_up']) > 0.10:
                        st.warning("⚠️ Los modelos divergen: LSTM ve patrones que el modelo básico ignora.")
                    else:
                        st.success("✅ Ambos cerebros (ML y LSTM) están alineados.")
        except Exception as e:
            st.error(f"Error en predicción LSTM: {str(e)}")
    else:
        st.info("💡 Entrena el cerebro LSTM en la barra lateral para ver este análisis.")

    # ============================================================================
    # 🎯 CONSENSUS SCORE - COMBINACIÓN DE TODOS LOS ANÁLISIS
    # ============================================================================
    
    st.markdown("---")
    st.markdown("## 🎯 Consensus Analysis")
    st.markdown("*Combinación inteligente de Score Técnico + ML + LSTM + Groq AI*")
    
    # Recopilar todas las predicciones disponibles
    ml_pred = None
    lstm_pred = None
    groq_text = None
    
    # 1. ML Prediction
    if ticker in st.session_state.ml_models:
        try:
            ml_pred = get_ml_prediction(st.session_state.ml_models[ticker], data_processed)
        except:
            pass
    
    # 2. LSTM Prediction
    lstm_key = f"{ticker}_lstm"
    if lstm_key in st.session_state.ml_models:
        try:
            lstm_pred = st.session_state.ml_models[lstm_key].predict(data_processed)
        except:
            pass
    
    # 3. Groq Analysis (si está en session_state)
    if 'last_groq_analysis' in st.session_state:
        groq_text = st.session_state.last_groq_analysis
    
    # Generar Consensus
    try:
        consensus_analyzer = ConsensusAnalyzer()
        consensus = consensus_analyzer.analyze_consensus(
            technical_score=analysis['signals']['score'],
            ml_prediction=ml_pred,
            lstm_prediction=lstm_pred,
            groq_analysis=groq_text
        )
        
        # Mostrar resultado en cards profesionales
        col_c1, col_c2, col_c3 = st.columns(3)
        
        with col_c1:
            score_color = "🟢" if consensus['consensus_score'] >= 70 else "🟡" if consensus['consensus_score'] >= 50 else "🔴"
            st.markdown(f"""
            <div class='metric-card'>
                <h4>Consensus Score</h4>
                <div style='font-size: 42px; font-weight: bold; color: #00ff88;'>
                    {score_color} {consensus['consensus_score']:.1f}<span style='font-size: 24px;'>/100</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_c2:
            conf_color = "#27ae60" if consensus['confidence'] >= 80 else "#f39c12" if consensus['confidence'] >= 60 else "#e74c3c"
            st.markdown(f"""
            <div class='metric-card'>
                <h4>Confianza</h4>
                <div style='font-size: 42px; font-weight: bold; color: {conf_color};'>
                    {consensus['confidence']:.0f}<span style='font-size: 24px;'>%</span>
                </div>
                <p style='font-size: 12px; color: #888;'>{len(consensus['sources_used'])}/4 fuentes</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_c3:
            rec = consensus['recommendation']
            rec_color = "#27ae60" if "COMPRA" in rec else "#e74c3c" if "VENTA" in rec else "#f39c12"
            rec_emoji = "🟢" if "COMPRA" in rec else "🔴" if "VENTA" in rec else "🟡"
            st.markdown(f"""
            <div class='metric-card'>
                <h4>Recomendación</h4>
                <div style='font-size: 24px; font-weight: bold; color: {rec_color}; margin-top: 15px;'>
                    {rec_emoji}<br>{rec}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Breakdown detallado
        with st.expander("📊 Ver Breakdown Completo del Consensus"):
            st.markdown("### Ponderación por Fuente")
            
            # Crear tabla de breakdown
            breakdown_data = []
            for source in consensus['sources_used']:
                source_score = consensus['source_scores'][source]
                weight = consensus['weights_used'][source]
                contribution = source_score * (weight / 100)
                
                source_names = {
                    'technical': '📊 Análisis Técnico',
                    'ml': '🤖 Machine Learning',
                    'lstm': '🧠 LSTM Deep Learning',
                    'groq': '💬 Groq AI'
                }
                
                breakdown_data.append({
                    'Fuente': source_names.get(source, source),
                    'Score': f"{source_score:.1f}/100",
                    'Peso': f"{weight:.0f}%",
                    'Contribución': f"{contribution:.1f}"
                })
            
            df_breakdown = pd.DataFrame(breakdown_data)
            st.dataframe(df_breakdown, use_container_width=True, hide_index=True)
            
            # Interpretación
            st.markdown("### 💡 Interpretación")
            
            if consensus['confidence'] >= 80:
                st.success("✅ **Alta confianza** - Las fuentes están muy alineadas. Esta es una señal fuerte.")
            elif consensus['confidence'] >= 60:
                st.info("ℹ️ **Confianza moderada** - Hay buen acuerdo entre las fuentes disponibles.")
            else:
                st.warning("⚠️ **Baja confianza** - Señales mixtas entre las fuentes. Proceder con precaución.")
            
            if consensus['consensus_score'] >= 70:
                st.markdown("📈 El consenso apunta firmemente hacia una **oportunidad de compra**.")
            elif consensus['consensus_score'] >= 60:
                st.markdown("↗️ El consenso sugiere un **ligero sesgo alcista**.")
            elif consensus['consensus_score'] <= 30:
                st.markdown("📉 El consenso apunta hacia una **oportunidad de venta**.")
            elif consensus['consensus_score'] <= 40:
                st.markdown("↘️ El consenso sugiere **precaución con sesgo bajista**.")
            else:
                st.markdown("↔️ El consenso sugiere **esperar por señales más claras**.")
            
            # Discrepancias
            if consensus['discrepancies']:
                st.markdown("### ⚠️ Señales Conflictivas Detectadas")
                for disc in consensus['discrepancies']:
                    st.warning(disc)
                st.caption("*Cuando hay discrepancias importantes, se recomienda análisis adicional.*")
        
        # Botón de actualización
        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            if st.button("🔄 Actualizar Análisis", use_container_width=True):
                st.cache_data.clear()
                st.success("✅ Datos actualizados")
                st.rerun()
        
    except Exception as e:
        st.error(f"Error generando consensus: {str(e)}")
        st.caption("Asegúrate de haber entrenado al menos el modelo ML para ver el consensus completo.")

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

    # ============================================================================
    # 🛡️ AI RISK OFFICER - BOTÓN DE ACTIVACIÓN
    # ============================================================================
    st.markdown("---")
    st.subheader("🛡️ AI Risk Officer - Validación Inteligente")
    
    col_ia1, col_ia2 = st.columns([1, 2])
    
    with col_ia1:
        st.write("Pulsa para que la IA valide tu gestión de riesgo basada en el VIX y el modelo ML.")
        # El botón clave que activa la consulta al CRO virtual
        btn_risk = st.button("⚖️ Validar Riesgo con IA", key="btn_cro_risk")
        
    with col_ia2:
        if btn_risk:
            with st.spinner("El CRO está evaluando la exposición..."):
                # Verificamos si hay un modelo de ML entrenado para darle más contexto a la IA
                ml_pred = None
                if ticker in st.session_state.ml_models:
                    ml_pred = get_ml_prediction(st.session_state.ml_models[ticker], data_processed)
                
                # Llamada a la función que pegamos en el Paso 1
                veredicto_ia = consultar_ia_riesgo(
                    ticker=ticker,
                    risk_calc=risk_calc,
                    position_calc=position_calc,
                    market_regime=market_regime,
                    ml_prediction=ml_pred
                )
                st.info(veredicto_ia)

    
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
# TAB 4: BACKTESTING (Versión Restaurada y Corregida)
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
    
    # --- BOTÓN DE EJECUCIÓN (Lógica original intacta) ---
    if st.button("▶️ Ejecutar Backtest"):
        with st.spinner("Ejecutando simulación con tu estrategia clásica..."):
            # Variables de simulación originales
            capital = backtest_capital
            posicion = 0
            precio_compra = 0
            historial_capital = []
            trades = []
            
            for i in range(1, len(data_processed)):
                precio = data_processed['Close'].iloc[i]
                rsi = data_processed['RSI'].iloc[i]
                macd_h = data_processed['MACD_Hist'].iloc[i] 
                
                # 1. Señal de COMPRA (Tu fórmula original)
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
                
                # 2. Señal de VENTA (Tu fórmula original)
                elif posicion > 0:
                    rendimiento = (precio - precio_compra) / precio_compra
                    
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
                
                # Registrar valor actual
                valor_actual = capital if posicion == 0 else posicion * precio
                historial_capital.append(valor_actual)
            
            # --- CÁLCULO FINAL Y GUARDADO EN SESIÓN ---
            valor_final = capital if posicion == 0 else posicion * data_processed['Close'].iloc[-1]
            rendimiento_total = ((valor_final - backtest_capital) / backtest_capital) * 100

            # Guardamos todo para que no se borre al picar la IA
            st.session_state.backtest_results = {
                'ticker': ticker,
                'capital_final': valor_final,
                'rendimiento': rendimiento_total,
                'trades': trades,
                'historial': historial_capital,
                'capital_inicial': backtest_capital
            }

    # --- VISUALIZACIÓN DE RESULTADOS (Fuera del botón para persistencia) ---
    if 'backtest_results' in st.session_state and st.session_state.backtest_results['ticker'] == ticker:
        res = st.session_state.backtest_results
        
        st.markdown("---")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Capital Inicial", f"${res['capital_inicial']:,.0f}")
        col_b.metric("Valor Final", f"${res['capital_final']:,.2f}")
        col_c.metric("Rendimiento", f"{res['rendimiento']:.2f}%", delta=f"{res['rendimiento']:.2f}%")
        col_d.metric("Trades Totales", len(res['trades']))
        
        # 🤖 BITÁCORA DE IA (Integrada profesionalmente)
        st.markdown("---")
        if st.button("🤖 Generar Bitácora de IA"):
            with st.spinner("La IA está realizando la autopsia del backtest..."):
                datos_ia = {
                    'inicial': res['capital_inicial'],
                    'final': res['capital_final'],
                    'rendimiento': res['rendimiento'],
                    'n_trades': len(res['trades'])
                }
                bitacora = analizar_backtest_con_ia(ticker, datos_ia, res['trades'])
                st.markdown("### 📜 Autopsia del Oráculo Quant")
                st.info(bitacora)

        # Gráfico de evolución original
        st.markdown("---")
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=data_processed.index[1:], 
            y=res['historial'], 
            fill='tozeroy', 
            line=dict(color='cyan')
        ))
        fig_bt.update_layout(title="Evolución de tu Capital ($)", template="plotly_dark", height=400)
        st.plotly_chart(fig_bt, use_container_width=True)
        
        # Bitácora de operaciones original
        if res['trades']:
            st.write("### 📜 Bitácora de Operaciones")
            st.dataframe(
                pd.DataFrame(res['trades']).sort_values(by="Fecha", ascending=False), 
                use_container_width=True, 
                hide_index=True
            )

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

# ============================================================================
        # 🤖 NUEVO: SELECCIÓN MAESTRA DE IA
        # ============================================================================
        st.markdown("---")
        st.subheader("🌟 Selección Maestra de la IA")
        if st.button("🤖 Analizar Top 3 Oportunidades"):
            with st.spinner("La IA está comparando setups técnicos..."):
                # Enviamos los resultados guardados en el session_state
                analisis_top = generar_top_picks_ia(st.session_state.scanner_results)
                st.markdown(analisis_top)

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
# TAB 7: MI PORTFOLIO
# ============================================================================

with tab7:
    st.header("💼 Mi Portfolio & Trading Journal")
    
    # Obtener precios actuales de los primeros 10 tickers de la watchlist
    current_prices = {}
    with st.spinner("Actualizando precios..."):
        for symbol in lista_completa[:10]:  # Solo primeros 10 para no tardar mucho
            try:
                data_symbol = fetcher.get_portfolio_data([symbol], period='1d')[symbol]
                if not data_symbol.empty:
                    current_prices[symbol] = data_symbol['Close'].iloc[-1]
            except:
                pass
    
    # Mostrar dashboard completo del portfolio
    display_portfolio_dashboard(portfolio_tracker, current_prices)
    
    # Sección para abrir nueva posición
    st.markdown("---")
    st.subheader("➕ Abrir Nueva Posición")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        new_ticker = st.selectbox("Ticker", lista_completa, key='new_position_ticker')
        new_shares = st.number_input("Número de Acciones", min_value=1, value=10, key='new_position_shares')
    
    with col2:
        # Obtener precio actual del ticker seleccionado
        try:
            data_current = fetcher.get_portfolio_data([new_ticker], period='1d')[new_ticker]
            current_price_new = data_current['Close'].iloc[-1]
        except:
            current_price_new = 100.0
        
        new_entry = st.number_input(
            "Precio de Entrada", 
            value=float(current_price_new), 
            format="%.2f",
            key='new_position_entry'
        )
        
        new_stop = st.number_input(
            "Stop Loss", 
            value=float(current_price_new * 0.95),  # -5% default
            format="%.2f",
            key='new_position_stop'
        )
    
    with col3:
        new_target = st.number_input(
            "Take Profit", 
            value=float(current_price_new * 1.10),  # +10% default
            format="%.2f",
            key='new_position_target'
        )
        
        new_strategy = st.text_input(
            "Estrategia/Notas", 
            value="Manual", 
            key='new_position_strategy'
        )
    
    # Botón para abrir posición
    if st.button("✅ Abrir Posición", use_container_width=True, type="primary"):
        try:
            position = portfolio_tracker.add_position(
                ticker=new_ticker,
                entry_price=new_entry,
                shares=new_shares,
                stop_loss=new_stop,
                take_profit=new_target,
                strategy=new_strategy,
                notes=f"Abierta desde Pato Terminal el {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            
            st.success(f"✅ Posición abierta: {new_ticker} ({new_shares} shares @ ${new_entry:.2f})")
            st.balloons()
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error abriendo posición: {str(e)}")
    
    # Cálculo de riesgo estimado
    risk_per_share = abs(new_entry - new_stop)
    total_risk = risk_per_share * new_shares
    position_size = new_entry * new_shares
    risk_pct = (total_risk / position_size) * 100 if position_size > 0 else 0
    
    st.caption(f"💡 Riesgo estimado: ${total_risk:.2f} ({risk_pct:.2f}% de la posición)")

# ============================================================================
# TAB 8: AUTO-TRADING
# ============================================================================

with tab8:
    st.header("🤖 Sistema de Auto-Trading")
    
    if auto_trader is None:
        st.error("❌ Auto-Trader no está configurado")
        st.markdown("""
        ### Pasos para configurar:
        1. Crear cuenta en Alpaca Markets
        2. Obtener API keys (Paper Trading)
        3. Agregar keys a Secrets:
           ```
           [ALPACA]
           api_key = "pk_xxx"
           api_secret = "xxx"
           paper_trading = true
           ```
        4. Reiniciar app
        """)
        st.stop()
    
    # Mostrar estado
    st.markdown("---")
    
    # Modo (Paper o Live)
    mode_emoji = "📝" if auto_trader.paper_mode else "💰"
    mode_text = "PAPER TRADING" if auto_trader.paper_mode else "LIVE TRADING"
    mode_color = "#f39c12" if auto_trader.paper_mode else "#e74c3c"
    
    st.markdown(f"""
    <div style='padding: 15px; background-color: {mode_color}20; border-left: 5px solid {mode_color}; margin-bottom: 20px;'>
        <h3 style='margin: 0;'>{mode_emoji} Modo: {mode_text}</h3>
        <p>{'Simulación - No se usa dinero real' if auto_trader.paper_mode else '⚠️ DINERO REAL - Ten precaución'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Estado de la cuenta
    status = auto_trader.get_status()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Valor de Cuenta", f"${status['account_value']:,.2f}")
    
    with col2:
        st.metric("💵 Poder de Compra", f"${status['buying_power']:,.2f}")
    
    with col3:
        st.metric("📊 Posiciones Abiertas", status['open_positions'])
    
    with col4:
        trades_today = status['safety_status']['daily_trades']
        max_trades = status['safety_status']['max_daily_trades']
        st.metric("📈 Trades Hoy", f"{trades_today}/{max_trades}")
    
    st.markdown("---")
    
    # Control del auto-trading
    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    
    with col_ctrl1:
        st.subheader("⚙️ Control del Sistema")
        
        # Toggle auto-trading
        auto_enabled = st.session_state.get('auto_trading_enabled', False)
        
        if auto_enabled:
            st.success("✅ Auto-Trading ACTIVO - El sistema está operando automáticamente")
            
            if st.button("⏸️ PAUSAR Auto-Trading", type="secondary", use_container_width=True):
                st.session_state.auto_trading_enabled = False
                st.rerun()
        else:
            st.warning("⏸️ Auto-Trading PAUSADO - No se ejecutarán trades automáticos")
            
            if st.button("▶️ ACTIVAR Auto-Trading", type="primary", use_container_width=True):
                st.session_state.auto_trading_enabled = True
                st.success("✅ Auto-Trading activado!")
                st.rerun()
    
    with col_ctrl2:
        st.subheader("🛑 Acciones de Emergencia")
        
        if st.button("❌ Cerrar TODAS las Posiciones", type="secondary"):
            if st.checkbox("⚠️ Confirmar cierre de todas las posiciones"):
                auto_trader.broker.close_all_positions()
                st.success("✅ Todas las posiciones cerradas")
                st.balloons()
        
        if st.button("🗑️ Cancelar TODAS las Órdenes"):
            auto_trader.broker.cancel_all_orders()
            st.success("✅ Todas las órdenes canceladas")
    
    st.markdown("---")
    
    # Configuración de seguridad
    with st.expander("⚙️ Configuración de Seguridad"):
        st.markdown("### Límites de Trading")
        
        safety_config = auto_trader.safety.config
        
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.number_input("Max Trades Diarios", 
                          value=safety_config['max_daily_trades'],
                          min_value=1, max_value=50,
                          key='safety_max_trades')
            
            st.number_input("Max Pérdida Diaria ($)", 
                          value=safety_config['max_daily_loss_usd'],
                          min_value=100, max_value=10000,
                          key='safety_max_loss')
            
            st.number_input("Consensus Score Mínimo", 
                          value=safety_config['min_consensus_score'],
                          min_value=50, max_value=95,
                          key='safety_min_score')
        
        with col_s2:
            st.number_input("Max Posiciones Abiertas", 
                          value=safety_config['max_open_positions'],
                          min_value=1, max_value=20,
                          key='safety_max_positions')
            
            st.number_input("Tamaño Max de Posición (%)", 
                          value=safety_config['max_position_size_pct'],
                          min_value=1.0, max_value=50.0,
                          key='safety_max_size')
            
            st.number_input("Confianza Mínima (%)", 
                          value=safety_config['min_confidence'],
                          min_value=50, max_value=95,
                          key='safety_min_conf')
        
        if st.button("💾 Guardar Configuración"):
            # Actualizar config
            auto_trader.safety.config['max_daily_trades'] = st.session_state.safety_max_trades
            auto_trader.safety.config['max_daily_loss_usd'] = st.session_state.safety_max_loss
            auto_trader.safety.config['min_consensus_score'] = st.session_state.safety_min_score
            auto_trader.safety.config['max_open_positions'] = st.session_state.safety_max_positions
            auto_trader.safety.config['max_position_size_pct'] = st.session_state.safety_max_size
            auto_trader.safety.config['min_confidence'] = st.session_state.safety_min_conf
            
            auto_trader.safety.save_config()
            st.success("✅ Configuración guardada")
    
    st.markdown("---")
    
    # Posiciones en Alpaca
    st.subheader("📊 Posiciones en Alpaca")
    
    alpaca_positions = auto_trader.broker.get_all_positions()
    
    if alpaca_positions:
        df_alpaca = pd.DataFrame(alpaca_positions)
        st.dataframe(df_alpaca, use_container_width=True, hide_index=True)
    else:
        st.info("No hay posiciones abiertas en Alpaca")
    
    st.markdown("---")
    
    # Log de trades
    st.subheader("📜 Historial de Auto-Trades")
    
    if auto_trader.trade_log:
        df_log = pd.DataFrame(auto_trader.trade_log)
        st.dataframe(df_log, use_container_width=True, hide_index=True)
    else:
        st.info("No hay trades ejecutados aún")

# ============================================================================
# 🤖 PASO C: MACHINE LEARNING (PÉGALO AQUÍ AHORA)
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.header("🤖 Machine Learning")

if st.sidebar.button("🎓 Entrenar Modelo ML"):
    with st.spinner(f"Entrenando cerebro para {ticker}..."):
        # Ahora sí, data_processed ya existe y el modelo puede aprender de él
        model = train_ml_model_for_ticker(ticker, data_processed, prediction_days=5)
        
        if model:
            st.session_state.ml_models[ticker] = model
            st.sidebar.success(f"✅ Modelo listo para {ticker}")
            st.sidebar.caption(f"Accuracy: {model.model_metrics['accuracy']*100:.1f}%")
        else:
            st.sidebar.error("❌ Error entrenando modelo")

# ============================================================================
# 🧠 NUEVO: BOTÓN LSTM (DEEP LEARNING)
# ============================================================================
st.sidebar.markdown("---")
if st.sidebar.button("🧠 Entrenar LSTM (Deep Learning)"):
    with st.spinner(f"🧠 Entrenando LSTM para {ticker}... (puede tardar 2-5 min)"):
        try:
            # Importamos el nuevo modelo avanzado
            from ml_model_lstm import train_lstm_model
            
            # Entrenar LSTM con ventana de 20 días
            lstm_model = train_lstm_model(
                ticker=ticker,
                data_processed=data_processed,
                prediction_days=5,
                lookback_window=20,
                epochs=50 
            )
            
            if lstm_model:
                # Guardamos con un nombre distinto para no borrar el modelo básico
                st.session_state.ml_models[f"{ticker}_lstm"] = lstm_model
                st.sidebar.success(f"✅ LSTM entrenado para {ticker}")
                st.sidebar.caption(f"Accuracy: {lstm_model.model_metrics['accuracy']*100:.1f}%")
            else:
                st.sidebar.error("❌ Error entrenando LSTM")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
            st.sidebar.caption("Verifica que ml_model_lstm.py esté en tu repo")

# ============================================================================
# SISTEMA AUTÓNOMO DE MONITOREO
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.header("🤖 Sistema Autónomo")

# Configurar sistema de monitoreo automático
if 'auto_monitor' not in st.session_state:
    st.session_state.auto_monitor = setup_auto_monitoring(
        st=st,
        watchlist=lista_completa,
        fetcher=fetcher,
        analyzer=analyzer,
        ml_models=st.session_state.ml_models,
        portfolio_tracker=portfolio_tracker
    )

# Mostrar controles del sistema autónomo
display_monitoring_controls(st, st.session_state.auto_monitor)

# Mostrar análisis en caché (opcional)
if st.sidebar.checkbox("📊 Ver Análisis en Caché"):
    st.sidebar.caption("Últimos análisis automáticos:")
    
    for ticker_cached in lista_completa[:5]:
        cached = st.session_state.auto_monitor.get_latest_analysis(ticker_cached)
        if cached:
            age_seconds = (datetime.now() - datetime.fromisoformat(cached['timestamp'])).seconds
            minutes_ago = age_seconds // 60
            seconds_ago = age_seconds % 60
            
            st.sidebar.caption(f"• {ticker_cached}: {minutes_ago}m {seconds_ago}s ago")

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



