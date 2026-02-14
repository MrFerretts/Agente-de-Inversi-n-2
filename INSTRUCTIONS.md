# 🎯 INSTRUCCIONES COMPLETAS - Pato Quant Terminal Pro v2.0

## 📦 LO QUE DESCARGASTE

Has descargado la carpeta `pato-quant-pro-v2/` que contiene TODO lo necesario para tu terminal financiera profesional.

---

## 🚀 OPCIÓN 1: TESTING LOCAL (Recomendado primero)

### Paso 1: Descomprimir y abrir terminal

```bash
cd ruta/donde/descargaste/pato-quant-pro-v2
```

### Paso 2: Configurar credenciales

```bash
# Opción A: Copiar template y editar
cp config_template.py config.py
nano config.py  # O abrir con tu editor favorito

# Opción B: Editar config_template.py directamente y renombrar
# Agregar tus API keys:
# - gemini_api_key
# - email password (Gmail App Password)
```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Ejecutar

```bash
streamlit run app.py
```

### Paso 5: Abrir navegador

La app se abrirá automáticamente en `http://localhost:8501`

### Paso 6: Verificar

- ✅ Sidebar muestra watchlist
- ✅ Puedes agregar/eliminar tickers
- ✅ 5 tabs funcionan correctamente
- ✅ Gráficos se cargan
- ✅ Risk management calcula stops

**Si todo funciona → Pasar a Opción 2**

---

## 🌐 OPCIÓN 2: SUBIR A GITHUB Y DEPLOY

### A. Crear cuenta GitHub (si no tienes)

1. Ve a [github.com/signup](https://github.com/signup)
2. Crea tu cuenta
3. Verifica tu email

### B. Crear repositorio

1. Ve a [github.com/new](https://github.com/new)
2. Repository name: `pato-quant-pro-v2`
3. Description: "Terminal financiera profesional"
4. Public o Private (tu elección)
5. **NO marcar** ninguna de las casillas (README, gitignore, license)
6. Click "Create repository"

### C. Subir tu código

En tu terminal, dentro de la carpeta del proyecto:

```bash
# 1. Configurar git (primera vez)
git config --global user.name "Tu Nombre"
git config --global user.email "tu_email@gmail.com"

# 2. Inicializar repositorio
git init

# 3. Agregar archivos
git add .

# 4. Primer commit
git commit -m "Initial commit - Pato Quant Pro v2"

# 5. Conectar con GitHub
git remote add origin https://github.com/TU_USUARIO/pato-quant-pro-v2.git

# 6. Cambiar a rama main
git branch -M main

# 7. Subir
git push -u origin main
```

Cuando pida autenticación:
- **Username**: tu_usuario_github
- **Password**: Usar Personal Access Token
  1. GitHub → Settings → Developer settings → Personal access tokens
  2. Generate new token (classic)
  3. Seleccionar `repo`
  4. Copiar token y usarlo como password

### D. Deploy en Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Selecciona:
   - Repository: `tu-usuario/pato-quant-pro-v2`
   - Branch: `main`
   - Main file path: `app.py`

5. Click "Advanced settings" → "Secrets"
6. Pegar esto (con tus credenciales):

```toml
[API_CONFIG]
gemini_api_key = "tu_gemini_key"

[PORTFOLIO_CONFIG]
stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
crypto = ["BTC-USD"]

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
```

7. Click "Deploy"
8. Espera 2-3 minutos

Tu app estará en: `https://tu-usuario-pato-quant-pro-v2.streamlit.app`

---

## 📁 ESTRUCTURA DEL PROYECTO

```
pato-quant-pro-v2/
│
├── 📄 app.py                    ← App principal (EJECUTAR ESTE)
├── 📄 config_template.py        ← Template de configuración
├── 📄 config.py                 ← Tu config (crear desde template)
├── 📄 requirements.txt          ← Dependencias
│
├── 📁 core/                     ← Módulos principales
│   ├── state_manager.py        ← Sistema de caché
│   └── risk_manager.py         ← Risk management
│
├── 📁 ui/                       ← Visualizaciones
│   └── chart_builder.py        ← Constructor de gráficos
│
├── 📁 data/                     ← Datos
│   └── watchlist.json          ← Lista de activos
│
├── 📁 docs/                     ← Documentación extra
│   ├── README.md
│   └── COMPARISON.md
│
├── 📄 market_data.py           ← Descarga de datos
├── 📄 technical_analysis.py    ← Análisis técnico
├── 📄 notifications.py         ← Email/Telegram
│
└── 📚 GUÍAS
    ├── README.md               ← Documentación principal
    ├── SETUP.md                ← Guía de configuración
    ├── DEPLOYMENT.md           ← Guía GitHub/Deploy
    ├── LICENSE                 ← Licencia MIT
    └── .gitignore              ← Archivos a ignorar en Git
```

---

## 🔑 OBTENER API KEYS

### 1. Gemini API (Gratis)

1. Ve a [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
2. "Create API key"
3. Copiar y pegar en config

### 2. Gmail App Password (Gratis)

1. Ve a [myaccount.google.com/security](https://myaccount.google.com/security)
2. Habilita "Verificación en 2 pasos"
3. Busca "App passwords"
4. Genera una para "Mail" → "Other (Python)"
5. Copia el password de 16 caracteres
6. Úsalo en config (NO tu password normal)

---

## ✅ CHECKLIST DE VERIFICACIÓN

### Testing Local
- [ ] Descargado y descomprimido proyecto
- [ ] `config.py` creado con credenciales
- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] App ejecutándose (`streamlit run app.py`)
- [ ] Todos los tabs funcionan
- [ ] Gráficos cargan correctamente

### GitHub
- [ ] Cuenta GitHub creada
- [ ] Repositorio creado
- [ ] Código subido con `git push`
- [ ] README visible en GitHub
- [ ] NO hay credenciales expuestas

### Streamlit Cloud (Opcional)
- [ ] App deployada
- [ ] Secrets configurados
- [ ] URL funcionando
- [ ] Datos cargando correctamente

---

## 🆘 AYUDA RÁPIDA

### ❌ Error: "No module named 'streamlit'"
```bash
pip install -r requirements.txt
```

### ❌ Error: "Fallo de configuración"
- Verifica que `config.py` existe
- O que configuraste Secrets en Streamlit Cloud

### ❌ Error: "No se pudieron cargar datos"
- Verifica conexión a internet
- Usa tickers válidos (ej: AAPL, BTC-USD)

### ❌ Email no se envía
- Usa Gmail App Password, no tu contraseña normal
- Verifica verificación en 2 pasos activa
- Prueba con otro email

### ❌ Git authentication failed
- Usa Personal Access Token, no tu password de GitHub
- O configura SSH keys

---

## 📚 DOCUMENTACIÓN COMPLETA

Para más detalles, lee estos archivos dentro del proyecto:

1. **README.md** → Documentación completa del proyecto
2. **SETUP.md** → Guía detallada de configuración
3. **DEPLOYMENT.md** → Guía completa de GitHub y deploy

---

## 🎯 PRÓXIMOS PASOS

1. ✅ Testea localmente primero
2. ✅ Cuando funcione bien, sube a GitHub
3. ✅ Deploy en Streamlit Cloud
4. ✅ Personaliza tu watchlist
5. ✅ Configura notificaciones
6. ✅ ¡Empieza a analizar el mercado!

---

## 💬 SOPORTE

¿Necesitas ayuda?

1. Lee la documentación completa
2. Revisa SETUP.md y DEPLOYMENT.md
3. Abre un issue en GitHub (si tu repo es público)

---

## 🎉 ¡FELICIDADES!

Tienes una terminal financiera profesional lista para usar.

**Features principales:**
- 📊 13 indicadores técnicos
- 💰 Risk management con ATR
- 🎯 Position sizing óptimo
- 🔥 Portfolio heat monitoring
- 🧪 Backtesting avanzado
- 🔍 Scanner multi-activo
- ⚡ 60-80% más rápido

---

**Creado con 🦆 por el equipo Pato Quant**

¡Buena suerte con tus inversiones! 📈💰
