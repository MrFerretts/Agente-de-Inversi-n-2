"""
Módulo de Notificaciones
Soporta email, Telegram y consola
ACTUALIZADO: 2026-02-12 - Agregado soporte PRE-MARKET
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NotificationManager:
    """Gestiona notificaciones multi-canal"""
    
    def __init__(self, config: Dict):
        self.config = config.get('NOTIFICATIONS', {})
        self.email_config = self.config.get('email', {})
        self.telegram_config = self.config.get('telegram', {})
        self.console_enabled = self.config.get('console', {}).get('enabled', True)
    
    def send_notification(self, subject: str, message: str, 
                         priority: str = 'normal', channel: str = 'all'):
        """
        Envía notificación por los canales habilitados
        
        Args:
            subject: Asunto/título de la notificación
            message: Contenido del mensaje
            priority: 'low', 'normal', 'high', 'critical'
            channel: 'email', 'telegram', 'console', 'all'
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f"[{timestamp}] [{priority.upper()}] {subject}\n\n{message}"
        
        if channel == 'all' or channel == 'console':
            if self.console_enabled:
                self._send_console(formatted_message, priority)
        
        if channel == 'all' or channel == 'email':
            if self.email_config.get('enabled', False):
                self._send_email(subject, message)
        
        if channel == 'all' or channel == 'telegram':
            if self.telegram_config.get('enabled', False):
                self._send_telegram(formatted_message)
    
    def _send_console(self, message: str, priority: str):
        """Muestra notificación en consola"""
        separador = "=" * 80
        
        if priority == 'critical':
            print(f"\n{'!'*80}")
            print(f"🚨 ALERTA CRÍTICA 🚨")
            print(f"{'!'*80}")
            print(message)
            print(f"{'!'*80}\n")
        elif priority == 'high':
            print(f"\n{separador}")
            print(f"⚠️  ALERTA IMPORTANTE")
            print(separador)
            print(message)
            print(f"{separador}\n")
        else:
            print(f"\n{'-'*80}")
            print(message)
            print(f"{'-'*80}\n")
    
    def _send_email(self, subject: str, body: str, is_html: bool = False) -> bool:
        """Envía notificación por email (Soporta HTML) y devuelve éxito"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config.get('sender')
            msg['To'] = self.email_config.get('recipient')
            msg['Subject'] = f"🤖 {subject}"
            msg.attach(MIMEText(body, 'html' if is_html else 'plain'))
            
            server = smtplib.SMTP(self.email_config.get('smtp_server'), self.email_config.get('smtp_port'))
            server.starttls()
            server.login(self.email_config.get('sender'), self.email_config.get('password'))
            server.send_message(msg)
            server.quit()
            logger.info(f"📧 Email enviado: {subject}")
            return True # <--- Agregado
        except Exception as e:
            logger.error(f"Error enviando email: {str(e)}")
            return False # <--- Agregado

    def send_full_report(self, df_summary, macro_info=None, ai_comment=None) -> bool:
        """Envía la tabla y devuelve si el proceso fue exitoso"""
        macro_info = macro_info or {}
        regime = macro_info.get('regime', 'UNKNOWN')
        vix_raw = macro_info.get('vix')
        vix = f"{vix_raw:.2f}" if vix_raw is not None else 'N/A'
        desc = macro_info.get('description', 'Sin información')
        
        subject = f"📊 Reporte de Mercado - {regime} | VIX: {vix}"
        color_regime = "#27ae60" if "ON" in str(regime).upper() else "#e74c3c"
        html_table = df_summary.to_html(index=False, border=0, classes='table')
        
        # ... (Cuerpo HTML del correo queda igual) ...
        html_body = f"<html>... (mismo contenido) ...</html>" 

        if self.email_config.get('enabled', False):
            return self._send_email(subject, html_body, is_html=True) # <--- Ahora devuelve el resultado
        return False
    
    def _send_telegram(self, message: str):
        """Envía notificación por Telegram"""
        try:
            import requests
            
            bot_token = self.telegram_config.get('bot_token')
            chat_id = self.telegram_config.get('chat_id')
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data)
            
            if response.status_code == 200:
                logger.info("Mensaje de Telegram enviado")
            else:
                logger.error(f"Error en Telegram: {response.text}")
                
        except Exception as e:
            logger.error(f"Error enviando Telegram: {str(e)}")
    
    def send_price_alert(self, alert_data: Dict):
        """Envía alerta de precio"""
        symbol = alert_data.get('symbol')
        current_price = alert_data.get('current_price')
        change_pct = alert_data.get('change_pct')
        direction = alert_data.get('direction')
        
        subject = f"Alerta de Precio: {symbol} {direction}"
        
        message = f"""
Cambio significativo de precio detectado:

Símbolo: {symbol}
Precio Actual: ${current_price:.2f}
Cambio: {change_pct:+.2f}%
Dirección: {direction}

Revisa tu estrategia para este activo.
        """
        
        priority = 'high' if abs(change_pct) > 10 else 'normal'
        self.send_notification(subject, message, priority='normal', channel='console')
    
    def send_signal_alert(self, symbol: str, analysis: Dict):
        """Envía alerta de señal de trading"""
        signals = analysis.get('signals', {})
        recommendation = signals.get('recommendation', 'MANTENER')
        score = signals.get('score', 0)
        
        subject = f"Señal de Trading: {symbol} - {recommendation}"
        
        buy_signals = '\n'.join(f"  • {s}" for s in signals.get('buy_signals', []))
        sell_signals = '\n'.join(f"  • {s}" for s in signals.get('sell_signals', []))
        
        message = f"""
Nueva señal de trading generada:

Activo: {symbol}
Precio: ${analysis['price']['current']:.2f}
Recomendación: {recommendation}
Score: {score}/100

SEÑALES DE COMPRA:
{buy_signals if buy_signals else '  Ninguna'}

SEÑALES DE VENTA:
{sell_signals if sell_signals else '  Ninguna'}

Indicadores Clave:
  • RSI: {analysis['indicators']['rsi']:.1f}
  • MACD: {analysis['indicators']['macd']:.4f}
  • Volatilidad: {analysis['indicators']['volatility']:.4f}

Requiere aprobación manual para ejecutar.
        """
        
        priority = 'high' if 'FUERTE' in recommendation else 'normal'
        self.send_notification(subject, message, priority)
    
    def send_portfolio_summary(self, portfolio_data: Dict):
        """Envía resumen del portfolio"""
        subject = "Resumen Diario del Portfolio"
        
        message = f"""
RESUMEN DEL PORTFOLIO
{'='*50}

Valor Total: ${portfolio_data['total_value']:,.2f}
Efectivo: ${portfolio_data['cash']:,.2f}
Invertido: ${portfolio_data['invested']:,.2f}

P/L No Realizado: ${portfolio_data['unrealized_pnl']:,.2f} ({portfolio_data['unrealized_pnl_pct']:+.2f}%)
Número de Posiciones: {portfolio_data['num_positions']}

POSICIONES PRINCIPALES:
"""
        # Añadir top 5 posiciones
        positions = sorted(
            portfolio_data['positions'].items(),
            key=lambda x: x[1]['position_value'],
            reverse=True
        )[:5]
        
        for symbol, pos in positions:
            message += f"\n{symbol}: ${pos['position_value']:,.2f} ({pos['unrealized_pnl_pct']:+.1f}%)"
        
        self.send_notification(subject, message, priority='low')
    
    def send_rebalance_alert(self, suggestions: list):
        """Envía alerta de rebalanceo necesario"""
        if not suggestions or suggestions[0].get('message'):
            return
        
        subject = "Rebalanceo de Portfolio Recomendado"
        
        message = "Se detectaron desviaciones en la asignación de activos:\n\n"
        
        for suggestion in suggestions:
            action = suggestion['action']
            asset_type = suggestion['asset_type']
            deviation = suggestion['deviation']
            
            message += f"\n{action} {asset_type.upper()}:"
            message += f"\n  Actual: {suggestion['current']:.1f}%"
            message += f"\n  Objetivo: {suggestion['target']:.1f}%"
            message += f"\n  Desviación: {deviation:+.1f}%"
            message += f"\n  Razón: {suggestion['reason']}\n"
        
        self.send_notification(subject, message, priority='normal')
        
    def send_full_report(self, df_summary, macro_info=None, ai_comment=None):
        """Envía la tabla con el encabezado macro y el insight de la IA"""
        macro_info = macro_info or {}
        
        regime = macro_info.get('regime', 'UNKNOWN')
        vix_raw = macro_info.get('vix')
        vix = f"{vix_raw:.2f}" if vix_raw is not None else 'N/A'
        desc = macro_info.get('description', 'Sin información')
        
        subject = f"📊 Reporte de Mercado - {regime} | VIX: {vix}"
        color_regime = "#27ae60" if "ON" in str(regime).upper() else "#e74c3c"
        
        html_table = df_summary.to_html(index=False, border=0, classes='table')
        
        # --- BLOQUE DE IA (Ahora con identación correcta) ---
        ai_box = ""
        if ai_comment:
            ai_box = f"""
            <div style="padding: 15px; background-color: #e8f4fd; border-left: 5px solid #3498db; margin-bottom: 20px; font-family: sans-serif;">
                <h3 style="margin-top: 0; color: #2980b9;">🤖 IA Insight (Gemini Quant)</h3>
                <p style="font-style: italic; color: #2c3e50;">{ai_comment.replace('\\n', '<br>')}</p>
            </div>
            """
        
        html_body = f"""
        <html>
        <head>
            <style>
                .table {{ font-family: sans-serif; border-collapse: collapse; width: 100%; font-size: 12px; }}
                .table th {{ background-color: #2c3e50; color: white; padding: 10px; text-align: left; }}
                .table td {{ border: 1px solid #ecf0f1; padding: 8px; }}
                .table tr:nth-child(even) {{ background-color: #f8f9fa; }}
                .macro-box {{ padding: 15px; background-color: #f4f7f6; border-left: 5px solid {color_regime}; margin-bottom: 20px; font-family: sans-serif; }}
            </style>
        </head>
        <body>
            <div class="macro-box">
                <h2 style="margin: 0; color: #2c3e50;">🌍 Contexto de Mercado: <span style="color: {color_regime};">{regime}</span></h2>
                <p style="margin: 5px 0;"><strong>Índice VIX:</strong> {vix} | <strong>Estado:</strong> {desc}</p>
            </div>
            
            {ai_box}
            {html_table}
            
            <p style="font-size: 10px; color: #bdc3c7; margin-top: 20px;">
                Generado el: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
            </p>
        </body>
        </html>
        """
        
        if self.email_config.get('enabled', False):
            self._send_email(subject, html_body, is_html=True)
    
    # ===================================================================
    # 🌅 NUEVO: ALERTA PRE-MARKET
    # ===================================================================
    def send_premarket_alert(self, df_summary, macro_info=None):
        """
        Envía alerta de pre-market 1 hora antes de apertura
        Solo incluye activos con movimiento significativo (gap > 0.5%)
        """
        macro_info = macro_info or {}
        
        regime = macro_info.get('regime', 'UNKNOWN')
        vix_raw = macro_info.get('vix')
        vix = f"{vix_raw:.2f}" if vix_raw is not None else 'N/A'
        desc = macro_info.get('description', 'Sin información')
        
        subject = f"🌅 PRE-MARKET | {regime} | VIX: {vix}"
        
        # Filtrar solo activos con movimiento significativo
        if 'Gap %' in df_summary.columns:
            # Convertir 'Gap %' de string a float si es necesario
            df_summary['Gap_numeric'] = df_summary['Gap %'].apply(
                lambda x: float(str(x).replace('%', '').replace('+', '')) if isinstance(x, str) else x
            )
            significant = df_summary[abs(df_summary['Gap_numeric']) > 0.5].copy()
        else:
            significant = df_summary
        
        # Color del recuadro
        color_regime = "#27ae60" if "ON" in str(regime).upper() else "#e74c3c"
        
        # Generar tabla HTML
        if not significant.empty:
            html_table = significant.to_html(index=False, border=0, classes='table')
            gap_count = len(significant)
            gap_message = f"Se detectaron {gap_count} activos con gaps significativos"
        else:
            html_table = '<p style="color: #7f8c8d;">No hay movimientos significativos en pre-market (gaps < 0.5%)</p>'
            gap_message = "Mercado tranquilo en pre-market"
        
        html_body = f"""
        <html>
        <head>
            <style>
                .table {{ font-family: sans-serif; border-collapse: collapse; width: 100%; font-size: 12px; }}
                .table th {{ background-color: #34495e; color: white; padding: 10px; text-align: left; }}
                .table td {{ border: 1px solid #ecf0f1; padding: 8px; }}
                .table tr:nth-child(even) {{ background-color: #f8f9fa; }}
                .alert-box {{ padding: 15px; background-color: #fff3cd; border-left: 5px solid #ffc107; margin-bottom: 20px; }}
                .macro-box {{ padding: 15px; background-color: #f4f7f6; border-left: 5px solid {color_regime}; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="alert-box">
                <h2 style="margin: 0;">🌅 ANÁLISIS PRE-MARKET</h2>
                <p><strong>Hora:</strong> {datetime.now().strftime('%H:%M:%S')} (1 hora antes de apertura)</p>
                <p><strong>Estado:</strong> {gap_message}</p>
            </div>
            
            <div class="macro-box">
                <p><strong>Régimen:</strong> {regime} | <strong>VIX:</strong> {vix}</p>
                <p><strong>Contexto:</strong> {desc}</p>
            </div>
            
            <h3>Activos con Gap Significativo (&gt;0.5%):</h3>
            {html_table}
            
            <p style="font-size: 10px; color: #bdc3c7; margin-top: 20px;">
                <strong>⚠️ ADVERTENCIA:</strong> Datos pre-market tienen bajo volumen y pueden revertirse en apertura.
                Use esta información como guía, no como señal definitiva.
            </p>
            
            <p style="font-size: 10px; color: #bdc3c7;">
                Generado el: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
            </p>
        </body>
        </html>
        """
        
        if self.email_config.get('enabled', False):
            self._send_email(subject, html_body, is_html=True)
        
        # También log en consola
        logger.info(f"🌅 PRE-MARKET: {gap_message}")


def format_currency(value: float) -> str:
    """Formatea valor como moneda"""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Formatea valor como porcentaje"""
    return f"{value:+.2f}%"
