# 🚀 SETUP COMPLETO - Pato Quant Terminal Pro v2.0

Guía paso a paso para poner tu terminal en producción.

---

## 📋 TABLA DE CONTENIDOS

1. [Instalación Local](#instalación-local)
2. [Configuración](#configuración)
3. [Deploy en Streamlit Cloud](#deploy-en-streamlit-cloud)
4. [Verificación](#verificación)
5. [Troubleshooting](#troubleshooting)

---

## 1️⃣ Instalación Local

### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/pato-quant-pro-v2.git
cd pato-quant-pro-v2
```

### Paso 2: Crear entorno virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

Esto instalará:
- streamlit (UI)
- plotly (gráficos)
- pandas, numpy (análisis)
- yfinance (datos de mercado)
- google-generativeai (IA)
- pytz, schedule (time management)

---

## 2️⃣ Configuración

### Opción A: Desarrollo Local

Edita `config.py` directamente:

```python
# ============= API KEYS =============
API_CONFIG = {
    'gemini_api_key': 'TU_GEMINI_API_KEY_AQUI',
}

# ============= NOTIFICACIONES =============
NOTIFICATIONS = {
    'email': {
        'enabled': True,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender': 'tu_email@gmail.com',
        'password': 'tu_app_password',  # Ver sección abajo
        'recipient': 'destino@email.com'
    },
    'telegram': {
        'enabled': False,  # Cambiar a True si quieres Telegram
        'bot_token': '',
        'chat_id': ''
    }
}
```

### Opción B: Streamlit Cloud

**NO edites `config.py`**. En su lugar:

1. Ve a tu app en Streamlit Cloud
2. Settings → Secrets
3. Agrega esto:

```toml
[API_CONFIG]
gemini_api_key = "tu_gemini_key_aqui"

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
smtp_server = "smtp.gmail.com"
smtp_port = 587
sender = "tu_email@gmail.com"
password = "tu_app_password"
recipient = "destino@email.com"

[NOTIFICATIONS.telegram]
enabled = false
bot_token = ""
chat_id = ""

[NOTIFICATIONS.console]
enabled = true
```

### Obtener API Keys

#### 1. Gemini API Key (Gratis)

1. Ve a [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Crea una API key
3. Copia y pega en la configuración

#### 2. Gmail App Password (Gratis)

1. Ve a [Google Account Security](https://myaccount.google.com/security)
2. Habilita "Verificación en 2 pasos"
3. Busca "App Passwords"
4. Genera una para "Mail" + "Other (Python)"
5. Copia el password de 16 caracteres
6. **Usa ESTE password, no tu contraseña normal**

#### 3. Telegram (Opcional)

1. Abre Telegram y busca `@BotFather`
2. Envía `/newbot` y sigue instrucciones
3. Copia el `bot_token`
4. Para obtener `chat_id`:
   - Envía un mensaje a tu bot
   - Ve a: `https://api.telegram.org/bot<TU_TOKEN>/getUpdates`
   - Busca el `chat_id` en el JSON

---

## 3️⃣ Deploy en Streamlit Cloud

### Paso 1: Preparar repositorio

```bash
# Asegúrate de que estos archivos estén en tu repo
git add .
git commit -m "Initial commit - Pato Quant Pro v2"
git push origin main
```

### Paso 2: Conectar a Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Inicia sesión con GitHub
3. Click "New app"
4. Selecciona:
   - Repository: `tu-usuario/pato-quant-pro-v2`
   - Branch: `main`
   - Main file path: `app.py`

### Paso 3: Configurar Secrets

1. En tu app desplegada → Settings → Secrets
2. Pega la configuración TOML (ver Opción B arriba)
3. Click "Save"

### Paso 4: Deploy

Click "Deploy" y espera 2-3 minutos.

---

## 4️⃣ Verificación

### Test Local

```bash
streamlit run app.py
```

Deberías ver:
- ✅ App cargando en `http://localhost:8501`
- ✅ Sidebar con watchlist
- ✅ 5 tabs visibles
- ✅ Gráficos renderizando
- ✅ Sin errores en consola

### Test Producción

Después del deploy:
- ✅ URL funcionando (ej: `https://tu-app.streamlit.app`)
- ✅ Datos cargando desde yfinance
- ✅ Notificaciones enviándose (test con Scanner)

### Checklist de Funcionalidad

- [ ] **Agregar ticker**: Funciona en sidebar
- [ ] **Ver gráficos**: Multi-panel renderiza
- [ ] **Cambiar tabs**: Rápido (<0.5 seg)
- [ ] **Risk management**: Calcula stops/targets
- [ ] **Backtest**: Se ejecuta sin errores
- [ ] **Scanner**: Procesa múltiples activos
- [ ] **Email**: Se envía reporte (opcional)

---

## 5️⃣ Troubleshooting

### Error: `No module named 'streamlit'`

**Causa**: Dependencias no instaladas

**Solución**:
```bash
pip install -r requirements.txt
```

### Error: `No se pudieron cargar datos para AAPL`

**Causa**: Problema con yfinance o ticker inválido

**Solución**:
- Verifica conexión a internet
- Usa tickers válidos de Yahoo Finance
- Para crypto usa formato: `BTC-USD`, `ETH-USD`

### Error: `Fallo de configuración`

**Causa**: `config.py` no encontrado o mal formado

**Solución**:
- Verifica que `config.py` existe en la raíz
- O configura secrets en Streamlit Cloud
- Revisa formato TOML (no Python) en Streamlit Cloud

### Error: Email no se envía

**Causa**: App Password incorrecto o no habilitado

**Solución**:
1. Verifica que usas **App Password**, no tu contraseña normal
2. Verifica verificación en 2 pasos está activa
3. Genera un nuevo App Password
4. Desactiva temporalmente firewall/antivirus
5. Prueba con otro email si persiste

### Error: `ModuleNotFoundError: No module named 'core'`

**Causa**: Estructura de carpetas incorrecta

**Solución**:
```bash
# Verifica estructura
ls -la

# Deberías ver:
app.py
core/
  state_manager.py
  risk_manager.py
ui/
  chart_builder.py
```

### Error: Caché no funciona

**Causa**: Session state de Streamlit

**Solución**:
- Click "🔄 Limpiar Caché" en sidebar
- O reinicia la app: `Ctrl+C` y `streamlit run app.py`

### App muy lenta en Streamlit Cloud

**Causa**: Free tier tiene recursos limitados

**Solución**:
- El caché ayuda mucho (ya implementado)
- Reduce el número de activos en watchlist
- Considera upgrade a plan Pro ($20/mes)

---

## 📝 Notas Importantes

### Seguridad

- ⚠️ **NUNCA** subas `config.py` con credenciales a GitHub
- ✅ Usa `.gitignore` (ya incluido)
- ✅ En producción, usa Streamlit Secrets

### Performance

- ✅ Caché activo por defecto (5 minutos TTL)
- ✅ Pre-cálculo de indicadores
- ✅ Optimizado para 10-20 activos

### Límites

- yfinance: Sin límite oficial pero puede fallar con muchos requests
- Gemini API: 60 requests/minuto (plan gratuito)
- Streamlit Cloud: 1 GB RAM (free tier)

---

## 🎯 Próximos Pasos

1. ✅ Personaliza tu watchlist
2. ✅ Configura notificaciones
3. ✅ Prueba todas las features
4. ✅ Comparte con tu equipo
5. ✅ Da feedback o reporta bugs

---

## 💬 Soporte

¿Necesitas ayuda?

1. Revisa esta guía completa
2. Lee el [README.md](./README.md)
3. Abre un issue en GitHub
4. Contacta al equipo

---

**¡Felicidades! Tu terminal está lista 🦆📈**
