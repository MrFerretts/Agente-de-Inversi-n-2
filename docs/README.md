# 🦆 Pato Quant Terminal Pro

Terminal financiera profesional construida con Streamlit para análisis técnico avanzado, gestión de riesgo y backtesting.

## 🚀 Características Principales

### 1. **Dashboard Principal**
- Visualización multi-panel (Precio, RSI, MACD, Volumen)
- Sistema de scoring multifactorial (±100 puntos)
- 13 indicadores técnicos calculados en tiempo real
- Detección de régimen de mercado (VIX, SPY, BTC)

### 2. **Análisis Técnico Avanzado**
- **Indicadores implementados:**
  - RSI (14) + Stochastic RSI
  - MACD (12, 26, 9)
  - Bandas de Bollinger (20, 2)
  - ADX (Average Directional Index)
  - ATR (Average True Range)
  - Volumen Relativo (RVOL)
  - SMAs (20, 50, 200)

- **Sistema de señales profesional:**
  - Ponderación por categorías (Tendencia 30%, Momentum 25%, Fuerza 20%, MACD 15%, Volumen 10%)
  - Clasificación: COMPRA FUERTE / COMPRA / MANTENER / VENTA / VENTA FUERTE
  - Nivel de confianza: MUY ALTA / ALTA / MEDIA / BAJA

### 3. **Risk Management Profesional**
- **Stops/Targets dinámicos basados en ATR:**
  - Stop Loss: Entrada - (ATR × 2)
  - Take Profit 1: Entrada + (ATR × 3)
  - Take Profit 2: Entrada + (ATR × 6)

- **Cálculo de posición óptimo:**
  - Size basado en % de riesgo por trade
  - Ajuste por volatilidad (ATR actual vs promedio)
  - Límites de exposición (máx 10% por activo)

- **Portfolio Heat:**
  - Riesgo total agregado
  - Semáforo de riesgo (🟢 🟡 🟠 🔴)

### 4. **Backtesting Pro**
- Motor de simulación con señales RSI + MACD
- Take Profit y Stop Loss configurables
- Historial completo de trades con P/L
- Métricas: Win Rate, Sharpe Ratio, Max Drawdown

### 5. **Scanner Multi-Activo**
- Escaneo simultáneo de toda la watchlist
- Ranking por score técnico
- Comparativa visual con gráficos
- Envío de reportes HTML por email

### 6. **Sistema de Caché Inteligente**
- Cache con TTL (Time To Live) de 5 minutos
- Evita recálculo innecesario de indicadores
- Optimización de performance (~70% más rápido)

## 📦 Instalación

```bash
# 1. Clonar el repositorio
git clone <tu-repo>
cd quant_terminal_pro

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar secretos (ver sección abajo)

# 5. Ejecutar la aplicación
streamlit run app_refactored.py
```

## ⚙️ Configuración

### Estructura de Archivos

```
quant_terminal_pro/
├── app_refactored.py          # App principal (USAR ESTE)
├── core/
│   ├── state_manager.py       # Sistema de caché
│   └── risk_manager.py        # Gestión de riesgo
├── ui/
│   └── chart_builder.py       # Visualizaciones
├── data/
│   └── watchlist.json         # Lista de activos
├── market_data.py             # Descarga de datos (yfinance)
├── technical_analysis.py      # Motor de análisis técnico
├── notifications.py           # Email/Telegram
├── config.py                  # Configuración base
└── requirements.txt
```

### Configuración de Secretos

**Opción 1: Streamlit Cloud (Recomendado)**

En el dashboard de Streamlit Cloud, ir a Settings > Secrets y agregar:

```toml
[API_CONFIG]
gemini_api_key = "tu_api_key_de_gemini"

[PORTFOLIO_CONFIG]
stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
crypto = ["BTC-USD", "ETH-USD"]

[TECHNICAL_INDICATORS]
sma_short = 20
sma_long = 50
rsi_period = 14
rsi_oversold = 30
rsi_overbought = 70
macd_fast = 12
macd_slow = 26
macd_signal = 9
bb_period = 20
bb_std = 2

[NOTIFICATIONS.email]
enabled = true
user = "tu_email@gmail.com"
password = "tu_app_password"  # Usar App Password de Gmail
destinatario = "destino@email.com"

[NOTIFICATIONS.telegram]
enabled = false
bot_token = ""
chat_id = ""
```

**Opción 2: Archivo local (Desarrollo)**

Usar el archivo `config.py` incluido y modificar con tus credenciales.

### Gmail App Password

Para notificaciones por email:
1. Ir a Google Account > Security
2. Activar verificación en 2 pasos
3. Generar "App Password" para Python
4. Usar esa contraseña en la configuración

## 🎯 Uso Básico

### 1. Agregar Activos a la Watchlist

```python
# En el sidebar
1. Escribir ticker (ej: "NVDA")
2. Click en "➕ Agregar"
3. El activo se guarda en watchlist.json
```

### 2. Analizar un Activo

```python
# Seleccionar ticker del dropdown
# Los datos se cargan automáticamente con caché
# Ver 5 tabs:
- Dashboard Principal (vista general)
- Análisis Técnico (detalles de indicadores)
- Risk Management (stops/targets)
- Backtesting (simulación histórica)
- Scanner (comparativa multi-activo)
```

### 3. Configurar Risk Management

```python
# En sidebar:
- Capital Total: $10,000
- Riesgo por Trade: 2%

# En tab "Risk Management":
- Ver stops/targets automáticos basados en ATR
- Calcular tamaño de posición óptimo
- Visualizar niveles en gráfico
```

### 4. Ejecutar Backtest

```python
# En tab "Backtesting Pro":
- Ajustar capital inicial
- Configurar Take Profit / Stop Loss
- Click en "▶️ Ejecutar Backtest"
- Ver evolución de capital y trades
```

### 5. Escanear Múltiples Activos

```python
# En tab "Scanner Multi-Activo":
- Click en "🚀 Iniciar Escaneo"
- Ver tabla ordenada por Score
- Enviar reporte por email
```

## 🧠 Sistema de Scoring

### Componentes del Score (Total: ±100)

1. **Tendencia (30 pts):**
   - Precio > SMA20 > SMA50: +30
   - Precio < SMA20 < SMA50: -30
   - Precio > SMA20: +15
   - Precio < SMA20: -15

2. **Momentum (25 pts):**
   - RSI < 30 + StochRSI < 0.2: +25 (sobreventa extrema)
   - RSI > 70 + StochRSI > 0.8: -25 (sobrecompra extrema)
   - Zona neutral: 0

3. **Fuerza Direccional (20 pts):**
   - ADX > 40: Multiplicador 1.4× al score actual
   - ADX > 25: Multiplicador 1.2×
   - ADX < 20: Multiplicador 0.4× (penalización lateral)

4. **MACD (15 pts):**
   - Histograma > 0 y creciendo: +15
   - Histograma < 0 y decreciendo: -15

5. **Volumen (10 pts):**
   - RVOL > 2.0 con score positivo: +10
   - RVOL > 1.5 con score positivo: +5

6. **Bandas de Bollinger (Extra):**
   - Precio en banda inferior: +5
   - Precio en banda superior: -5

### Interpretación de Scores

| Score | Recomendación | Descripción |
|-------|--------------|-------------|
| ≥ 60 | COMPRA FUERTE | Múltiples confirmaciones alcistas |
| 30-59 | COMPRA | Señales positivas moderadas |
| -29 a 29 | MANTENER | Zona neutral sin señal clara |
| -30 a -59 | VENTA | Señales negativas moderadas |
| ≤ -60 | VENTA FUERTE | Múltiples confirmaciones bajistas |

## 📊 Arquitectura

### Flujo de Datos

```
User Input → StateManager (Cache) → MarketDataFetcher (yfinance)
                ↓
        DataProcessor (Indicadores)
                ↓
        TechnicalAnalyzer (Señales)
                ↓
        RiskManager (Stops/Sizing)
                ↓
        ChartBuilder (Visualización)
                ↓
        NotificationManager (Alertas)
```

### Optimizaciones Implementadas

1. **Caché con TTL:**
   - Datos de mercado: 5 minutos
   - Análisis técnico: Por símbolo
   - Evita descargas redundantes

2. **Pre-cálculo de indicadores:**
   - Una sola pasada por el DataFrame
   - Todos los indicadores calculados juntos
   - Reutilización en múltiples tabs

3. **Lazy loading:**
   - Datos solo se cargan cuando se selecciona el ticker
   - Scanner solo procesa lo necesario

## 🔧 Personalización

### Agregar Nuevo Indicador

1. **En `state_manager.py` → `DataProcessor.prepare_full_analysis()`:**
```python
# Agregar cálculo del indicador
df['MI_INDICADOR'] = tu_funcion_de_calculo(df)
```

2. **En `technical_analysis.py` → `_generate_signals_professional()`:**
```python
# Agregar lógica de señales
mi_valor = indicators.get('mi_indicador', 0)
if mi_valor > umbral:
    score += puntos
    buy_signals.append("Mi señal personalizada")
```

3. **En `chart_builder.py` → `create_multi_indicator_chart()`:**
```python
# Agregar visualización
fig.add_trace(go.Scatter(
    x=df.index, y=df['MI_INDICADOR'], name="Mi Indicador"
), row=panel, col=1)
```

### Modificar Estrategia de Backtesting

En `app_refactored.py`, tab4, modificar las condiciones de compra/venta:

```python
# Ejemplo: Agregar condición de volumen
if posicion == 0 and rsi < 35 and macd_hist > 0 and rvol > 1.5:
    # Comprar
    ...
```

## 🐛 Troubleshooting

### Error: "No se pudieron cargar datos"
- Verificar conexión a internet
- Verificar que el ticker es válido (usar formato Yahoo Finance)
- Crypto debe tener sufijo "-USD" (ej: BTC-USD)

### Error: "Fallo de configuración"
- Verificar que existe `config.py` o configuración en Streamlit Cloud
- Revisar formato de secretos (TOML correcto)

### Caché no se actualiza
- Click en "🔄 Limpiar Caché" en sidebar
- O reiniciar la app

### Email no se envía
- Verificar App Password de Gmail
- Verificar que "Acceso de apps poco seguras" está activado
- Revisar firewall/antivirus

## 📈 Mejoras Futuras (Roadmap)

- [ ] Integración con broker (Alpaca, Interactive Brokers)
- [ ] Machine Learning para predicción de señales
- [ ] Análisis de sentimiento (Twitter, Reddit)
- [ ] Backtesting multi-estrategia simultáneo
- [ ] Dashboard de portfolio en tiempo real
- [ ] Alertas automáticas por WhatsApp
- [ ] Optimización de parámetros con grid search
- [ ] Exportación de reportes en PDF

## 📝 Licencia

MIT License - Úsalo, modifícalo y distribúyelo libremente.

## 🤝 Contribuciones

Pull requests son bienvenidos. Para cambios mayores:
1. Abrir un issue primero
2. Discutir qué te gustaría cambiar
3. Asegurar que los tests pasen (pytest)

## 📧 Contacto

Creado por el equipo de Pato Quant 🦆

---

**⚠️ Disclaimer:** Esta herramienta es solo para fines educativos y de investigación. No constituye asesoría financiera. Operar en mercados financieros conlleva riesgos. Consulta con un asesor profesional antes de tomar decisiones de inversión.
