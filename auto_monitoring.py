"""
SISTEMA AUTÓNOMO DE MONITOREO Y ALERTAS
Ejecuta análisis automáticamente en background y envía alertas
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import time
import threading
from typing import Dict, List, Optional, Callable
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
from consensus_analyzer import ConsensusAnalyzer #

class AutoMonitoringSystem:
    """
    Sistema que ejecuta análisis automáticos en background
    - Monitorea watchlist
    - Ejecuta modelos ML automáticamente
    - Detecta señales
    - Envía alertas
    """
    
    def __init__(self, 
                 watchlist: List[str],
                 fetcher,
                 analyzer,
                 ml_models: Dict = None,
                 portfolio_tracker = None,
                 check_interval: int = 300):  # 5 minutos
        """
        Args:
            watchlist: Lista de tickers a monitorear
            fetcher: MarketDataFetcher instance
            analyzer: TechnicalAnalyzer instance
            ml_models: Dict de modelos ML {ticker: model}
            portfolio_tracker: PortfolioTracker instance
            check_interval: Segundos entre chequeos (default 5 min)
        """
        self.watchlist = watchlist
        self.fetcher = fetcher
        self.analyzer = analyzer
        self.ml_models = ml_models or {}
        self.portfolio_tracker = portfolio_tracker
        self.check_interval = check_interval
        
        self.is_running = False
        self.monitoring_thread = None
        self.last_check = None
        self.alerts_sent = []
        
        # Configuración de alertas
        self.alert_config = {
            'ml_threshold': 0.75,  # Alertar si prob > 75%
            'score_threshold': 60,  # Alertar si score > 60
            'divergence': True,     # Alertar en divergencias
            'stop_loss': True,      # Alertar si stop activado
            'take_profit': True,    # Alertar si target alcanzado
            'ml_agreement': 0.85    # Alertar si modelos muy alineados
        }
        
        # Caché de análisis
        self.analysis_cache = {}
        
    def start_monitoring(self):
        """Inicia monitoreo en background"""
        if self.is_running:
            print("⚠️ Monitoreo ya está corriendo")
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        print("✅ Sistema de monitoreo iniciado")
        print(f"   Watchlist: {', '.join(self.watchlist)}")
        print(f"   Intervalo: {self.check_interval}s")
    
    def stop_monitoring(self):
        """Detiene monitoreo"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("⏸️ Sistema de monitoreo detenido")
    
    def _monitoring_loop(self):
        """Loop principal de monitoreo"""
        while self.is_running:
            try:
                print(f"\n🔄 [{datetime.now().strftime('%H:%M:%S')}] Ejecutando análisis automático...")
                
                # Analizar cada ticker
                for ticker in self.watchlist:
                    analysis = self._analyze_ticker(ticker)
                    
                    if analysis:
                        # Guardar en caché
                        self.analysis_cache[ticker] = {
                            'analysis': analysis,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Detectar señales y enviar alertas
                        self._check_alerts(ticker, analysis)
                
                # Chequear posiciones abiertas
                if self.portfolio_tracker:
                    self._check_positions()
                
                self.last_check = datetime.now()
                print(f"✅ Análisis completado. Próximo chequeo en {self.check_interval}s")
                
            except Exception as e:
                print(f"❌ Error en monitoreo: {str(e)}")
            
            # Esperar hasta próximo ciclo
            time.sleep(self.check_interval)
    
    def _analyze_ticker(self, ticker: str) -> Optional[Dict]:
        """
        Analiza un ticker completamente
        
        Args:
            ticker: Símbolo
        
        Returns:
            Dict con análisis completo
        """
        try:
            # Cargar datos
            data = self.fetcher.get_portfolio_data([ticker], period='1y')[ticker]
            
            if data.empty:
                return None
            
            # Procesar datos con indicadores
            from core.state_manager import DataProcessor
            data_processed = DataProcessor.prepare_full_analysis(data, self.analyzer)
            
            # Análisis técnico
            analysis = self.analyzer.analyze_asset(data_processed, ticker)
            
            # Predicción ML (si hay modelo)
            ml_prediction = None
            if ticker in self.ml_models:
                try:
                    model = self.ml_models[ticker]
                    ml_prediction = model.predict(data_processed)
                except:
                    pass
            
            # Precio actual
            current_price = data_processed['Close'].iloc[-1]
            
            # --- NUEVO: CÁLCULO DE CONSENSUS PARA EL BOT ---
            # Buscamos si existe modelo LSTM para este ticker
            lstm_prediction = None
            lstm_key = f"{ticker}_lstm"
            if lstm_key in self.ml_models:
                try: lstm_prediction = self.ml_models[lstm_key].predict(data_processed)
                except: pass

            # Generamos el veredicto unificado
            ca = ConsensusAnalyzer()
            consensus = ca.analyze_consensus(
                technical_score=analysis['signals']['score'],
                ml_prediction=ml_prediction,
                lstm_prediction=lstm_prediction
            )

            return {
                'ticker': ticker,
                'price': current_price,
                'technical': analysis,
                'ml_prediction': ml_prediction,
                'consensus': consensus,      # ← Requerido por el bot
                'data_processed': data_processed, # ← Requerido para ATR
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error analizando {ticker}: {str(e)}")
            return None
    
    def _check_alerts(self, ticker: str, analysis: Dict):
        """
        Revisa si hay alertas para enviar
        
        Args:
            ticker: Símbolo
            analysis: Análisis completo
        """
        alerts = []
        
        # 1. Score técnico alto
        score = analysis['technical']['signals']['score']
        if abs(score) >= self.alert_config['score_threshold']:
            if score > 0:
                alerts.append({
                    'type': 'technical_buy',
                    'priority': 'high' if score > 70 else 'medium',
                    'message': f"📈 {ticker}: Score técnico {score}/100 - {analysis['technical']['signals']['recommendation']}"
                })
            else:
                alerts.append({
                    'type': 'technical_sell',
                    'priority': 'high' if score < -70 else 'medium',
                    'message': f"📉 {ticker}: Score técnico {score}/100 - {analysis['technical']['signals']['recommendation']}"
                })
        
        # 2. Predicción ML alta
        if analysis['ml_prediction']:
            ml_pred = analysis['ml_prediction']
            
            if ml_pred['probability_up'] >= self.alert_config['ml_threshold']:
                alerts.append({
                    'type': 'ml_buy',
                    'priority': 'high',
                    'message': f"🤖 {ticker}: ML predice {ml_pred['probability_up']*100:.1f}% probabilidad de subida - {ml_pred['recommendation']}"
                })
            elif ml_pred['probability_up'] <= (1 - self.alert_config['ml_threshold']):
                alerts.append({
                    'type': 'ml_sell',
                    'priority': 'high',
                    'message': f"🤖 {ticker}: ML predice {ml_pred['probability_down']*100:.1f}% probabilidad de bajada - {ml_pred['recommendation']}"
                })
            
            # 3. Alto acuerdo entre modelos (si es ensemble)
            if ml_pred.get('model_agreement', 0) >= self.alert_config['ml_agreement']:
                alerts.append({
                    'type': 'ml_agreement',
                    'priority': 'high',
                    'message': f"🎯 {ticker}: Alto acuerdo ML ({ml_pred['model_agreement']*100:.0f}%) - Señal muy confiable"
                })
        
        # 4. Divergencias
        if self.alert_config['divergence']:
            buy_signals = analysis['technical']['signals'].get('buy_signals', [])
            for signal in buy_signals:
                if 'DIVERGENCIA' in signal.upper():
                    alerts.append({
                        'type': 'divergence',
                        'priority': 'high',
                        'message': f"⚡ {ticker}: {signal}"
                    })
        
        # Enviar alertas
        for alert in alerts:
            self._send_alert(alert)

    # ============================================================================
        # 📦 NUEVO: EJECUCIÓN DE AUTO-TRADING
        # ============================================================================
        if 'auto_trader' in st.session_state and st.session_state.get('auto_trading_enabled', False):
            auto_trader = st.session_state.auto_trader
            consensus = analysis['consensus'] # Obtenido en el Paso 2
            data_processed = analysis['data_processed']
            
            # Si hay consensus muy fuerte, ejecutar (Mínimo 75 Score y 80% Confianza)
            if (consensus['consensus_score'] >= 75 and consensus['confidence'] >= 80):
                try:
                    current_price = analysis['price']
                    atr = data_processed['ATR'].iloc[-1]
                    vix = 20  # Podríamos traer el VIX real desde fetcher si se requiere
                    
                    # El bot de Alpaca toma el control aquí
                    result = auto_trader.evaluate_and_execute(
                        ticker=ticker,
                        consensus=consensus,
                        current_price=current_price,
                        atr=atr,
                        vix=vix
                    )
                    
                    if result:
                        print(f"🚀 [AUTO-TRADE] Orden ejecutada con éxito para {ticker}")
                except Exception as e:
                    print(f"❌ Error en auto-trade {ticker}: {str(e)}")
    
    def _check_positions(self):
        """Revisa posiciones abiertas para stops/targets"""
        if not self.portfolio_tracker:
            return
        
        positions = self.portfolio_tracker.get_open_positions()
        
        for position in positions:
            ticker = position['ticker']
            current_price = position.get('current_price', 0)
            
            if current_price == 0:
                continue
            
            # Check stop loss
            if self.alert_config['stop_loss']:
                if current_price <= position['stop_loss']:
                    alert = {
                        'type': 'stop_loss',
                        'priority': 'critical',
                        'message': f"🛑 {ticker}: STOP LOSS ACTIVADO a ${current_price:.2f} (Stop: ${position['stop_loss']:.2f})"
                    }
                    self._send_alert(alert)
                    
                    # Auto-cerrar posición si configurado
                    # self.portfolio_tracker.close_position(position['id'], current_price, "Stop Loss")
            
            # Check take profit
            if self.alert_config['take_profit']:
                if current_price >= position['take_profit']:
                    alert = {
                        'type': 'take_profit',
                        'priority': 'high',
                        'message': f"🎯 {ticker}: TARGET ALCANZADO a ${current_price:.2f} (Target: ${position['take_profit']:.2f})"
                    }
                    self._send_alert(alert)
    
    def _send_alert(self, alert: Dict):
        """
        Envía alerta (email, log, etc)
        
        Args:
            alert: Dict con type, priority, message
        """
        # Evitar duplicados en corto tiempo
        alert_key = f"{alert['type']}_{alert['message'][:50]}"
        current_time = datetime.now()
        
        # Buscar si ya enviamos esta alerta recientemente (última hora)
        for sent_alert in self.alerts_sent[-50:]:  # Últimas 50 alertas
            if (sent_alert['key'] == alert_key and 
                (current_time - sent_alert['timestamp']).seconds < 3600):
                return  # No reenviar
        
        # Registrar alerta
        self.alerts_sent.append({
            'key': alert_key,
            'alert': alert,
            'timestamp': current_time
        })
        
        # Mostrar en consola
        priority_emoji = {
            'critical': '🚨',
            'high': '⚠️',
            'medium': 'ℹ️',
            'low': '💡'
        }
        emoji = priority_emoji.get(alert['priority'], '📢')
        
        print(f"\n{emoji} ALERTA [{alert['priority'].upper()}]")
        print(f"   {alert['message']}")
        print(f"   Timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # TODO: Enviar email, Telegram, etc.
        # self._send_email_alert(alert)
    
    def get_latest_analysis(self, ticker: str) -> Optional[Dict]:
        """
        Obtiene el último análisis en caché
        
        Args:
            ticker: Símbolo
        
        Returns:
            Dict con análisis o None
        """
        return self.analysis_cache.get(ticker)
    
    def get_monitoring_status(self) -> Dict:
        """
        Retorna estado del sistema
        
        Returns:
            Dict con estado
        """
        return {
            'is_running': self.is_running,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'watchlist_size': len(self.watchlist),
            'alerts_sent_today': len([a for a in self.alerts_sent 
                                     if (datetime.now() - a['timestamp']).days == 0]),
            'cache_size': len(self.analysis_cache),
            'next_check_in': self.check_interval if self.is_running else None
        }


class AlertManager:
    """
    Gestor de alertas (email, Telegram, etc)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.email_enabled = config.get('email', {}).get('enabled', False)
        
    def send_email_alert(self, subject: str, message: str):
        """Envía alerta por email"""
        if not self.email_enabled:
            return
        
        try:
            email_config = self.config['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['sender']
            msg['To'] = email_config['recipient']
            msg['Subject'] = subject
            
            body = f"""
            {message}
            
            ---
            Enviado por Pato Quant Terminal Pro
            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['sender'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            print(f"✅ Email enviado: {subject}")
            
        except Exception as e:
            print(f"❌ Error enviando email: {str(e)}")


# ============================================================================
# INTEGRACIÓN CON STREAMLIT
# ============================================================================

def setup_auto_monitoring(st, watchlist, fetcher, analyzer, ml_models=None, portfolio_tracker=None):
    """
    Configura sistema autónomo en Streamlit
    
    Args:
        st: Streamlit module
        watchlist: Lista de tickers
        fetcher: MarketDataFetcher
        analyzer: TechnicalAnalyzer
        ml_models: Dict de modelos ML
        portfolio_tracker: PortfolioTracker
    
    Returns:
        AutoMonitoringSystem instance
    """
    # Inicializar en session_state si no existe
    if 'auto_monitor' not in st.session_state:
        st.session_state.auto_monitor = AutoMonitoringSystem(
            watchlist=watchlist,
            fetcher=fetcher,
            analyzer=analyzer,
            ml_models=ml_models,
            portfolio_tracker=portfolio_tracker,
            check_interval=300  # 5 minutos
        )
    
    return st.session_state.auto_monitor


def display_monitoring_controls(st, monitor: AutoMonitoringSystem):
    """
    Muestra controles del sistema de monitoreo
    
    Args:
        st: Streamlit module
        monitor: AutoMonitoringSystem instance
    """
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 Sistema Autónomo")
    
    status = monitor.get_monitoring_status()
    
    if status['is_running']:
        st.sidebar.success("✅ Monitoreo ACTIVO")
        
        if status['last_check']:
            last_check = datetime.fromisoformat(status['last_check'])
            time_since = (datetime.now() - last_check).seconds
            st.sidebar.caption(f"Último chequeo: hace {time_since}s")
        
        st.sidebar.caption(f"Alertas hoy: {status['alerts_sent_today']}")
        
        if st.sidebar.button("⏸️ Detener Monitoreo"):
            monitor.stop_monitoring()
            st.rerun()
    else:
        st.sidebar.warning("⏸️ Monitoreo DETENIDO")
        
        if st.sidebar.button("▶️ Iniciar Monitoreo"):
            monitor.start_monitoring()
            st.rerun()
    
    # Configuración
    with st.sidebar.expander("⚙️ Configuración Alertas"):
        st.slider(
            "ML Threshold",
            0.0, 1.0, 
            monitor.alert_config['ml_threshold'],
            key='ml_threshold'
        )
        
        st.slider(
            "Score Threshold",
            0, 100,
            monitor.alert_config['score_threshold'],
            key='score_threshold'
        )
        
        st.checkbox(
            "Alertar en Divergencias",
            monitor.alert_config['divergence'],
            key='divergence_alert'
        )
