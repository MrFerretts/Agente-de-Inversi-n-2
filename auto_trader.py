"""
AUTO-TRADING SYSTEM
Sistema completo de trading automático integrado con Consensus Score
Broker: Alpaca (API gratuita)
"""

import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import pandas as pd
import time
from typing import Dict, Optional, List, Tuple # ← Agregamos Tuple aquí
import json
import os


class AlpacaConnector:
    """
    Conector con Alpaca Markets
    Paper Trading (simulación) o Live Trading
    """
    
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        """
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            paper: True para paper trading (recomendado), False para real
        """
        base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
        
        self.api = tradeapi.REST(
            api_key,
            api_secret,
            base_url,
            api_version='v2'
        )
        
        self.paper_mode = paper
        self.account = None
        self._update_account()
    
    def _update_account(self):
        """Actualiza información de la cuenta"""
        try:
            self.account = self.api.get_account()
        except Exception as e:
            print(f"Error actualizando cuenta: {str(e)}")
    
    def get_buying_power(self) -> float:
        """Retorna poder de compra disponible"""
        self._update_account()
        return float(self.account.buying_power)
    
    def get_portfolio_value(self) -> float:
        """Retorna valor total del portfolio"""
        self._update_account()
        return float(self.account.portfolio_value)
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Obtiene precio actual del símbolo"""
        try:
            quote = self.api.get_latest_trade(symbol)
            return float(quote.price)
        except Exception as e:
            print(f"Error obteniendo precio de {symbol}: {str(e)}")
            return None
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Obtiene posición actual de un símbolo"""
        try:
            position = self.api.get_position(symbol)
            return {
                'symbol': position.symbol,
                'qty': float(position.qty),
                'avg_entry_price': float(position.avg_entry_price),
                'current_price': float(position.current_price),
                'market_value': float(position.market_value),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc)
            }
        except:
            return None  # No hay posición
    
    def get_all_positions(self) -> List[Dict]:
        """Obtiene todas las posiciones abiertas"""
        try:
            positions = self.api.list_positions()
            return [{
                'symbol': p.symbol,
                'qty': float(p.qty),
                'avg_entry_price': float(p.avg_entry_price),
                'current_price': float(p.current_price),
                'unrealized_pl': float(p.unrealized_pl),
                'unrealized_plpc': float(p.unrealized_plpc)
            } for p in positions]
        except Exception as e:
            print(f"Error obteniendo posiciones: {str(e)}")
            return []
    
    def place_market_order(self, symbol: str, qty: int, side: str = 'buy') -> Optional[Dict]:
        """
        Coloca orden de mercado
        
        Args:
            symbol: Ticker
            qty: Cantidad de acciones
            side: 'buy' o 'sell'
        
        Returns:
            Dict con información de la orden o None
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            
            return {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'submitted_at': order.submitted_at
            }
        except Exception as e:
            print(f"Error colocando orden {side} {symbol}: {str(e)}")
            return None
    
    def place_bracket_order(self, symbol: str, qty: int, 
                           stop_loss: float, take_profit: float) -> Optional[Dict]:
        """
        Coloca orden bracket (con stop loss y take profit automáticos)
        
        Args:
            symbol: Ticker
            qty: Cantidad
            stop_loss: Precio de stop loss
            take_profit: Precio de take profit
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='day',
                order_class='bracket',
                stop_loss={'stop_price': stop_loss},
                take_profit={'limit_price': take_profit}
            )
            
            return {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'status': order.status,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
        except Exception as e:
            print(f"Error colocando bracket order {symbol}: {str(e)}")
            return None
    
    def cancel_all_orders(self):
        """Cancela todas las órdenes pendientes"""
        try:
            self.api.cancel_all_orders()
            print("✅ Todas las órdenes canceladas")
        except Exception as e:
            print(f"Error cancelando órdenes: {str(e)}")
    
    def close_position(self, symbol: str) -> bool:
        """Cierra una posición"""
        try:
            self.api.close_position(symbol)
            print(f"✅ Posición {symbol} cerrada")
            return True
        except Exception as e:
            print(f"Error cerrando posición {symbol}: {str(e)}")
            return False
    
    def close_all_positions(self):
        """Cierra todas las posiciones"""
        try:
            self.api.close_all_positions()
            print("✅ Todas las posiciones cerradas")
        except Exception as e:
            print(f"Error cerrando posiciones: {str(e)}")


class SafetyManager:
    """
    Sistema de seguridad para auto-trading
    Previene pérdidas excesivas y comportamiento errático
    """
    
    def __init__(self, config_file: str = "data/safety_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        
        # Estado diario
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_reset = datetime.now().date()
    
    def _load_config(self) -> Dict:
        """Carga configuración de seguridad"""
        default_config = {
            'max_daily_trades': 10,
            'max_daily_loss_usd': 500,
            'max_daily_loss_pct': 5.0,
            'max_position_size_pct': 10.0,
            'min_consensus_score': 75,
            'min_confidence': 80,
            'max_open_positions': 5,
            'required_buying_power_pct': 20.0,
            'trading_hours_start': '09:30',
            'trading_hours_end': '15:30',
            'avoid_high_vix': True,
            'max_vix': 35,
            'cooldown_after_loss_minutes': 30
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded = json.load(f)
                    default_config.update(loaded)
            except:
                pass
        
        return default_config
    
    def save_config(self):
        """Guarda configuración"""
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def reset_daily_counters(self):
        """Resetea contadores si es un nuevo día"""
        today = datetime.now().date()
        if today > self.last_reset:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset = today
            print(f"📅 Contadores reseteados para {today}")
    
    def can_trade(self, consensus: Dict, account_value: float, 
                  current_positions: int, vix: float = 20) -> Tuple[bool, str]:
        """
        Verifica si es seguro ejecutar un trade
        
        Returns:
            (puede_operar, razón)
        """
        self.reset_daily_counters()
        
        # Check 1: Límite de trades diarios
        if self.daily_trades >= self.config['max_daily_trades']:
            return False, f"❌ Límite de trades diarios alcanzado ({self.config['max_daily_trades']})"
        
        # Check 2: Pérdida máxima en USD
        if self.daily_pnl <= -self.config['max_daily_loss_usd']:
            return False, f"❌ Límite de pérdida diaria alcanzado (${-self.daily_pnl:.2f})"
        
        # Check 3: Pérdida máxima en %
        loss_pct = (self.daily_pnl / account_value) * 100
        if loss_pct <= -self.config['max_daily_loss_pct']:
            return False, f"❌ Límite de pérdida diaria % alcanzado ({loss_pct:.2f}%)"
        
        # Check 4: Consensus score mínimo
        if consensus['consensus_score'] < self.config['min_consensus_score']:
            return False, f"⚠️ Consensus score bajo ({consensus['consensus_score']:.1f} < {self.config['min_consensus_score']})"
        
        # Check 5: Confianza mínima
        if consensus['confidence'] < self.config['min_confidence']:
            return False, f"⚠️ Confianza baja ({consensus['confidence']:.0f}% < {self.config['min_confidence']}%)"
        
        # Check 6: Máximo de posiciones abiertas
        if current_positions >= self.config['max_open_positions']:
            return False, f"⚠️ Máximo de posiciones abiertas ({self.config['max_open_positions']})"
        
        # Check 7: Horario de trading
        if not self._is_trading_hours():
            return False, "⏰ Fuera de horario de trading"
        
        # Check 8: VIX alto (volatilidad extrema)
        if self.config['avoid_high_vix'] and vix > self.config['max_vix']:
            return False, f"📈 VIX muy alto ({vix:.1f} > {self.config['max_vix']}) - Mercado volátil"
        
        return True, "✅ Seguro para operar"
    
    def _is_trading_hours(self) -> bool:
        """Verifica si estamos en horario de trading"""
        now = datetime.now()
        
        # Solo días de semana
        if now.weekday() >= 5:  # Sábado o Domingo
            return False
        
        current_time = now.time()
        start = datetime.strptime(self.config['trading_hours_start'], '%H:%M').time()
        end = datetime.strptime(self.config['trading_hours_end'], '%H:%M').time()
        
        return start <= current_time <= end
    
    def calculate_position_size(self, consensus_score: float, 
                                account_value: float) -> int:
        """
        Calcula tamaño de posición basado en consensus y Kelly Criterion
        
        Args:
            consensus_score: 0-100
            account_value: Valor total de la cuenta
        
        Returns:
            Número de acciones a comprar
        """
        # Tamaño base: % del portfolio
        base_pct = self.config['max_position_size_pct']
        
        # Ajustar según consensus score
        if consensus_score >= 90:
            size_pct = base_pct * 1.0  # 100% del máximo
        elif consensus_score >= 80:
            size_pct = base_pct * 0.8  # 80%
        elif consensus_score >= 75:
            size_pct = base_pct * 0.6  # 60%
        else:
            size_pct = base_pct * 0.4  # 40%
        
        position_value = account_value * (size_pct / 100)
        
        return position_value
    
    def register_trade(self, pnl: float = 0):
        """Registra un trade ejecutado"""
        self.daily_trades += 1
        self.daily_pnl += pnl
    
    def get_status(self) -> Dict:
        """Retorna estado actual del safety manager"""
        return {
            'daily_trades': self.daily_trades,
            'max_daily_trades': self.config['max_daily_trades'],
            'daily_pnl': self.daily_pnl,
            'max_daily_loss': self.config['max_daily_loss_usd'],
            'trades_remaining': self.config['max_daily_trades'] - self.daily_trades,
            'trading_hours': self._is_trading_hours()
        }


class AutoTrader:
    """
    Sistema principal de Auto-Trading
    Integra Consensus Score + Alpaca + Safety Manager
    """
    
    def __init__(self, 
                 alpaca_api_key: str,
                 alpaca_secret: str,
                 consensus_analyzer,
                 portfolio_tracker,
                 paper_trading: bool = True):
        """
        Args:
            alpaca_api_key: Alpaca API key
            alpaca_secret: Alpaca API secret
            consensus_analyzer: ConsensusAnalyzer instance
            portfolio_tracker: PortfolioTracker instance
            paper_trading: True para simulación
        """
        self.broker = AlpacaConnector(alpaca_api_key, alpaca_secret, paper=paper_trading)
        self.safety = SafetyManager()
        self.consensus_analyzer = consensus_analyzer
        self.portfolio_tracker = portfolio_tracker
        
        self.paper_mode = paper_trading
        self.trade_log = []
    
    def evaluate_and_execute(self, ticker: str, consensus: Dict, 
                            current_price: float, atr: float, vix: float = 20) -> Optional[Dict]:
        """
        Evalúa señal de consensus y ejecuta trade si cumple condiciones
        
        Args:
            ticker: Símbolo
            consensus: Dict del consensus analyzer
            current_price: Precio actual
            atr: ATR del activo
            vix: VIX actual (volatilidad del mercado)
        
        Returns:
            Dict con resultado o None
        """
        # 1. Verificar safety
        account_value = self.broker.get_portfolio_value()
        current_positions = len(self.broker.get_all_positions())
        
        can_trade, reason = self.safety.can_trade(
            consensus=consensus,
            account_value=account_value,
            current_positions=current_positions,
            vix=vix
        )
        
        if not can_trade:
            print(f"🛑 No se puede operar {ticker}: {reason}")
            return None
        
        print(f"✅ {ticker} - Safety check passed: {reason}")
        
        # 2. Calcular tamaño de posición
        position_value = self.safety.calculate_position_size(
            consensus['consensus_score'],
            account_value
        )
        
        shares = int(position_value / current_price)
        
        if shares < 1:
            print(f"⚠️ {ticker} - Shares calculados < 1, cancelando")
            return None
        
        # 3. Calcular stops dinámicos
        stop_loss = current_price - (atr * 2)
        
        # Take profit dinámico según consensus
        if consensus['consensus_score'] >= 85:
            tp_multiplier = 4  # R/R 2:1
        elif consensus['consensus_score'] >= 75:
            tp_multiplier = 3  # R/R 1.5:1
        else:
            tp_multiplier = 2.5  # R/R 1.25:1
        
        take_profit = current_price + (atr * tp_multiplier)
        
        # 4. Ejecutar trade
        print(f"\n{'='*60}")
        print(f"🤖 EJECUTANDO AUTO-TRADE")
        print(f"{'='*60}")
        print(f"Ticker: {ticker}")
        print(f"Consensus Score: {consensus['consensus_score']:.1f}/100")
        print(f"Confianza: {consensus['confidence']:.0f}%")
        print(f"Precio: ${current_price:.2f}")
        print(f"Shares: {shares}")
        print(f"Valor: ${current_price * shares:.2f}")
        print(f"Stop Loss: ${stop_loss:.2f} (-{((current_price - stop_loss) / current_price) * 100:.1f}%)")
        print(f"Take Profit: ${take_profit:.2f} (+{((take_profit - current_price) / current_price) * 100:.1f}%)")
        print(f"R/R Ratio: {((take_profit - current_price) / (current_price - stop_loss)):.2f}:1")
        print(f"{'='*60}\n")
        
        # Ejecutar bracket order (con stops automáticos)
        order = self.broker.place_bracket_order(
            symbol=ticker,
            qty=shares,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        if order:
            # Registrar en safety manager
            self.safety.register_trade()
            
            # Registrar en portfolio tracker
            self.portfolio_tracker.add_position(
                ticker=ticker,
                entry_price=current_price,
                shares=shares,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy=f"Auto-Trading Consensus ({consensus['consensus_score']:.0f})",
                notes=f"Confianza: {consensus['confidence']:.0f}% | Order ID: {order['id']}"
            )
            
            # Log del trade
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'action': 'BUY',
                'shares': shares,
                'price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'consensus_score': consensus['consensus_score'],
                'confidence': consensus['confidence'],
                'order_id': order['id']
            }
            
            self.trade_log.append(trade_record)
            self._save_trade_log()
            
            print(f"✅ Trade ejecutado exitosamente!")
            print(f"Order ID: {order['id']}\n")
            
            return trade_record
        else:
            print(f"❌ Error ejecutando trade\n")
            return None
    
    def _save_trade_log(self):
        """Guarda log de trades"""
        os.makedirs('data', exist_ok=True)
        with open('data/auto_trade_log.json', 'w') as f:
            json.dump(self.trade_log, f, indent=2)
    
    def get_status(self) -> Dict:
        """Retorna estado completo del auto-trader"""
        return {
            'paper_mode': self.paper_mode,
            'account_value': self.broker.get_portfolio_value(),
            'buying_power': self.broker.get_buying_power(),
            'open_positions': len(self.broker.get_all_positions()),
            'safety_status': self.safety.get_status(),
            'trades_today': self.safety.daily_trades,
            'pnl_today': self.safety.daily_pnl
        }
