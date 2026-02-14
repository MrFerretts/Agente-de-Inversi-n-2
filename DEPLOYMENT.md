# 📦 DEPLOYMENT A GITHUB - Guía Paso a Paso

## 🎯 Objetivo

Subir tu proyecto `pato-quant-pro-v2` a GitHub para:
- Versionamiento
- Respaldo en la nube
- Deploy en Streamlit Cloud
- Colaboración

---

## 📋 PRERREQUISITOS

1. ✅ Cuenta de GitHub ([crear aquí](https://github.com/signup))
2. ✅ Git instalado en tu computadora
   ```bash
   # Verificar instalación
   git --version
   
   # Si no está instalado:
   # Windows: Descargar de https://git-scm.com/
   # Mac: brew install git
   # Linux: sudo apt-get install git
   ```

---

## 🚀 PASO 1: Configurar Git (Primera vez)

Si nunca has usado Git:

```bash
git config --global user.name "Tu Nombre"
git config --global user.email "tu_email@example.com"
```

---

## 🚀 PASO 2: Crear Repositorio en GitHub

### Opción A: Desde GitHub Web

1. Ve a [github.com](https://github.com)
2. Click en "+" (arriba derecha) → "New repository"
3. Configurar:
   - **Repository name**: `pato-quant-pro-v2`
   - **Description**: "Terminal financiera profesional con análisis técnico"
   - **Visibility**: Public (o Private si prefieres)
   - ❌ **NO marcar** "Add a README file"
   - ❌ **NO marcar** "Add .gitignore"
   - ❌ **NO marcar** "Choose a license"
4. Click "Create repository"

### Opción B: Desde GitHub CLI (si está instalado)

```bash
gh repo create pato-quant-pro-v2 --public --description "Terminal financiera profesional"
```

---

## 🚀 PASO 3: Inicializar Repositorio Local

En la carpeta de tu proyecto:

```bash
cd pato-quant-pro-v2

# Inicializar git
git init

# Verificar archivos
git status
```

Deberías ver todos tus archivos en rojo (untracked).

---

## 🚀 PASO 4: Agregar Archivos

```bash
# Agregar todos los archivos (excepto los de .gitignore)
git add .

# Verificar qué se agregará
git status
```

Deberías ver archivos en verde (staged).

⚠️ **IMPORTANTE**: Verifica que `config.py` NO aparezca si tiene credenciales.

Si aparece:
```bash
# Removerlo del staging
git reset config.py

# Crear un config_template.py sin credenciales
cp config.py config_template.py
# Editar config_template.py y borrar las API keys
# Luego:
git add config_template.py
```

---

## 🚀 PASO 5: Primer Commit

```bash
git commit -m "Initial commit - Pato Quant Terminal Pro v2.0"
```

---

## 🚀 PASO 6: Conectar con GitHub

Usa la URL de tu repositorio (la copiaste en Paso 2):

```bash
git remote add origin https://github.com/TU_USUARIO/pato-quant-pro-v2.git

# Verificar
git remote -v
```

Deberías ver:
```
origin  https://github.com/TU_USUARIO/pato-quant-pro-v2.git (fetch)
origin  https://github.com/TU_USUARIO/pato-quant-pro-v2.git (push)
```

---

## 🚀 PASO 7: Subir a GitHub

```bash
# Cambiar a rama main (si estás en master)
git branch -M main

# Push al repositorio
git push -u origin main
```

Te pedirá autenticación:

### Opción A: HTTPS (más fácil)
- Username: tu_usuario_github
- Password: Usar **Personal Access Token**
  1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
  2. "Generate new token" → Seleccionar `repo`
  3. Copiar token y usarlo como password

### Opción B: SSH (más seguro)
```bash
# Generar clave SSH
ssh-keygen -t ed25519 -C "tu_email@example.com"

# Agregar a GitHub
# Copiar contenido de ~/.ssh/id_ed25519.pub
# GitHub → Settings → SSH keys → New SSH key → Pegar

# Cambiar remote a SSH
git remote set-url origin git@github.com:TU_USUARIO/pato-quant-pro-v2.git
```

---

## ✅ PASO 8: Verificar

Ve a `https://github.com/TU_USUARIO/pato-quant-pro-v2`

Deberías ver:
- ✅ Todos tus archivos
- ✅ README.md renderizado
- ✅ Estructura de carpetas (core/, ui/, data/)

---

## 🔄 ACTUALIZACIONES FUTURAS

Cuando hagas cambios:

```bash
# 1. Ver qué cambió
git status

# 2. Agregar cambios
git add .

# 3. Commit con mensaje descriptivo
git commit -m "Agregar nueva estrategia de backtesting"

# 4. Push a GitHub
git push
```

---

## 🌐 PASO 9: Deploy en Streamlit Cloud

Ahora que tu código está en GitHub:

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. New app → Selecciona tu repo
4. Main file: `app.py`
5. Advanced: Configura Secrets (ver SETUP.md)
6. Deploy!

En 2-3 minutos tendrás tu app en:
```
https://tu-usuario-pato-quant-pro-v2.streamlit.app
```

---

## 🔐 SEGURIDAD - MUY IMPORTANTE

### ❌ NUNCA subas a GitHub:

- API keys
- Passwords
- Tokens
- Archivos con credenciales

### ✅ Usar en su lugar:

1. **Local**: `config.py` (en `.gitignore`)
2. **Producción**: Streamlit Secrets
3. **Template**: `config_template.py` (sin credenciales)

### Si accidentalmente subiste credenciales:

```bash
# 1. Cambiar las credenciales INMEDIATAMENTE en el servicio
# 2. Remover del historial de git:
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch config.py" \
  --prune-empty --tag-name-filter cat -- --all

# 3. Force push
git push origin --force --all
```

---

## 📝 COMANDOS ÚTILES

```bash
# Ver estado actual
git status

# Ver historial de commits
git log --oneline

# Ver diferencias antes de commit
git diff

# Deshacer cambios locales (no committed)
git checkout -- archivo.py

# Ver ramas
git branch

# Crear nueva rama
git checkout -b feature/nueva-funcionalidad

# Cambiar a main
git checkout main

# Mergear rama
git merge feature/nueva-funcionalidad

# Clonar tu repo en otra computadora
git clone https://github.com/TU_USUARIO/pato-quant-pro-v2.git
```

---

## 🐛 TROUBLESHOOTING

### Error: "remote origin already exists"

```bash
git remote remove origin
git remote add origin https://github.com/TU_USUARIO/pato-quant-pro-v2.git
```

### Error: "failed to push some refs"

```bash
# Alguien hizo cambios en GitHub
git pull origin main --rebase
git push
```

### Error: "Authentication failed"

- Verifica usuario/email
- Usa Personal Access Token (no tu password de GitHub)
- O configura SSH

### Conflictos en merge

```bash
# Git marcará los conflictos en los archivos
# Editar manualmente y resolver
# Luego:
git add archivo_resuelto.py
git commit -m "Resolver conflictos"
```

---

## 📚 RECURSOS

- [GitHub Docs](https://docs.github.com/)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)
- [Streamlit Deploy Docs](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app)

---

## ✅ CHECKLIST FINAL

- [ ] Repositorio creado en GitHub
- [ ] Git configurado localmente
- [ ] Archivos agregados y committed
- [ ] Remote conectado
- [ ] Push exitoso a GitHub
- [ ] README visible en GitHub
- [ ] NO hay credenciales expuestas
- [ ] App deployada en Streamlit Cloud (opcional)

---

**¡Felicidades! Tu proyecto está en GitHub 🎉**

Ahora puedes:
- ✅ Trabajar desde cualquier computadora
- ✅ Compartir con tu equipo
- ✅ Tener respaldo automático
- ✅ Deploy en Streamlit Cloud

---

**Creado por el equipo Pato Quant 🦆**
