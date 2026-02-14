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

state_mgr = st.session_state.state_manager
risk_mgr = st.session_state.risk_manager
chart_builder = st.session_state.chart_builder
fetcher = st.session_state.fetcher
analyzer = st.session_state.analyzer
notifier = st.session_state.notifier

def consultar_ia_groq(ticker, analysis, signals, market_regime):
    try:
        from groq import Groq
        client = Groq(api_key=API_CONFIG['groq_api_key'])
        
        # Extraemos datos profundos de tu motor de análisis 
        ind = analysis['indicators']
        contexto = f"VIX: {market_regime['vix']:.2f}, Régimen: {market_regime['regime']}"
        
        # Construimos un contexto técnico ultra-detallado [cite: 74, 86]
        prompt = f"""
        Actúa como un Senior Quantitative Researcher de un Hedge Fund. 
        Analiza el activo {ticker} con los siguientes datos técnicos reales:
        
        - CONTEXTO MACRO: {contexto}
        - PRECIO: ${signals['price']:.2f} (Cambio: {signals['price_change_pct']:.2f}%)
        - MOMENTUM: RSI({ind['rsi']:.1f}), StochRSI({ind['stoch_rsi']:.2f})
        - TENDENCIA: ADX({ind['adx']:.1f}), MACD Hist({ind['macd_hist']:.4f})
        - VOLATILIDAD/FLUJO: ATR(${ind['atr']:.2f}), RVOL({ind['rvol']:.2f}x)
        - SCORE TOTAL: {analysis['signals']['score']} ({analysis['signals']['recommendation']})
        
        INSTRUCCIONES:
        1. Realiza una síntesis profesional analizando convergencias/divergencias.
        2. Evalúa si el volumen (RVOL) valida el movimiento del precio.
        3. Da un veredicto de gestión de riesgo basado en la volatilidad actual.
        4. Usa un tono serio, técnico y directo. Máximo 4 párrafos cortos.
        """
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # Modelo potente para razonamiento complejo 
            messages=[{"role": "system", "content": "Eres un terminal quant de alta precisión."},
                      {"role": "user", "content": prompt}],
            temperature=0.3, # Menor temperatura para mayor precisión técnica
            max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error con Groq Pro: {str(e)}"
        
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

# ============================================================================
# TABS PRINCIPALES
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard Principal",
    "📈 Análisis Técnico Avanzado",
    "💰 Risk Management",
    "🧪 Backtesting Pro",
    "🔍 Scanner Multi-Activo"
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

    # Dentro de with tab1, donde estaba el botón del oráculo [cite: 93, 105]
if st.button("🔮 Consultar al Oráculo (Análisis Profundo)"):
    with st.spinner("Realizando análisis quant multidimensional..."):
        # Ahora pasamos el objeto completo de análisis y el régimen de mercado [cite: 1, 85]
        respuesta = consultar_ia_groq(
            ticker, 
            analysis,       # El diccionario completo con los 13 indicadores [cite: 85]
            signals,        # Precios y cambios recientes 
            market_regime   # VIX y tendencia de SPY 
        )
        st.markdown(f"### 🤖 Análisis Pro de Groq")
        st.info(respuesta)
    
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
# TAB 2: ANÁLISIS TÉCNICO AVANZADO
# ============================================================================

with tab2:
    st.header("📈 Análisis Técnico Detallado")
    
    # Información del régimen de mercado
    with st.spinner("Analizando contexto macro..."):
        market_regime = fetcher.get_market_regime()
    
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
