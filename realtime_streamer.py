"""
REAL-TIME WEBSOCKET STREAMER
Sistema optimizado de streaming en tiempo real con gestión de memoria
Previene saturación de RAM y fugas de memoria
"""

import asyncio
import threading
from collections import deque
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Deque
import logging
from alpaca_trade_api.rest import REST # ← REST es la clave para evitar bloqueos en la nube
import time # Para el control de velocidad

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryOptimizedBuffer:
    """
    Buffer circular con límite de memoria
    Automáticamente descarta datos viejos
    """
    
    def __init__(self, max_size: int = 1000, max_age_seconds: int = 3600):
        """
        Args:
            max_size: Máximo número de elementos en buffer
            max_age_seconds: Máxima edad de datos en segundos (default 1 hora)
        """
        self.buffer: Deque = deque(maxlen=max_size)
        self.max_age = timedelta(seconds=max_age_seconds)
        self.lock = threading.Lock()
    
    def append(self, data: Dict):
        """Agrega dato con timestamp"""
        with self.lock:
            data['_timestamp'] = datetime.now()
            self.buffer.append(data)
            
            # Limpiar datos viejos
            self._clean_old_data()
    
    def _clean_old_data(self):
        """Elimina datos más viejos que max_age"""
        cutoff_time = datetime.now() - self.max_age
        
        # Eliminar desde el inicio (más viejos)
        while self.buffer and self.buffer[0]['_timestamp'] < cutoff_time:
            self.buffer.popleft()
    
    def get_latest(self, n: int = 1) -> List[Dict]:
        """Obtiene los últimos N elementos"""
        with self.lock:
            return list(self.buffer)[-n:] if self.buffer else []
    
    def get_all(self) -> List[Dict]:
        """Obtiene todos los elementos actuales"""
        with self.lock:
            return list(self.buffer)
    
    def clear(self):
        """Limpia el buffer"""
        with self.lock:
            self.buffer.clear()
    
    def size(self) -> int:
        """Retorna tamaño actual del buffer"""
        return len(self.buffer)
    
    def memory_usage_mb(self) -> float:
        """Estima uso de memoria en MB"""
        import sys
        total_size = sum(sys.getsizeof(item) for item in self.buffer)
        return total_size / (1024 * 1024)


class RealTimeStreamer:
    """
    Streamer de datos en tiempo real optimizado para Streamlit
    
    Características:
    - Buffers circulares (no crece infinitamente)
    - Auto-limpieza de datos viejos
    - Agregación de datos para reducir memoria
    - Límites de memoria configurables
    """
    
    def __init__(self, 
                 alpaca_key: str,
                 alpaca_secret: str,
                 paper: bool = True,
                 max_buffer_size: int = 1000,
                 data_retention_seconds: int = 3600):
        """
        Args:
            alpaca_key: Alpaca API key
            alpaca_secret: Alpaca API secret
            paper: True para paper trading
            max_buffer_size: Tamaño máximo de buffer por símbolo
            data_retention_seconds: Tiempo máximo de retención (default 1 hora)
        """
        base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
        
        self.api = REST(
            alpaca_key,
            alpaca_secret,
            base_url=base_url
        )  # O 'sip' para datos más completos ($$$)
        
        # Buffers por símbolo (gestión de memoria)
        self.quotes_buffer: Dict[str, MemoryOptimizedBuffer] = {}
        self.trades_buffer: Dict[str, MemoryOptimizedBuffer] = {}
        self.bars_buffer: Dict[str, MemoryOptimizedBuffer] = {}
        
        # Configuración
        self.max_buffer_size = max_buffer_size
        self.data_retention = data_retention_seconds
        
        # Estado
        self.is_running = False
        self.subscribed_symbols = set()
        self.stream_thread = None
        
        # Estadísticas
        self.stats = {
            'messages_received': 0,
            'last_update': None,
            'symbols_active': 0
        }
    
    def _init_buffers(self, symbol: str):
        """Inicializa buffers para un símbolo"""
        if symbol not in self.quotes_buffer:
            self.quotes_buffer[symbol] = MemoryOptimizedBuffer(
                max_size=self.max_buffer_size,
                max_age_seconds=self.data_retention
            )
            self.trades_buffer[symbol] = MemoryOptimizedBuffer(
                max_size=self.max_buffer_size,
                max_age_seconds=self.data_retention
            )
            self.bars_buffer[symbol] = MemoryOptimizedBuffer(
                max_size=100,  # Menos bars (más pesados)
                max_age_seconds=self.data_retention
            )
    
    async def _handle_quote(self, q):
        """Handler para quotes (bid/ask)"""
        try:
            symbol = q.symbol
            self._init_buffers(symbol)
            
            # Solo guardar datos esenciales (optimización de memoria)
            data = {
                'bid': float(q.bid_price),
                'ask': float(q.ask_price),
                'bid_size': int(q.bid_size),
                'ask_size': int(q.ask_size),
                'spread': float(q.ask_price - q.bid_price)
            }
            
            self.quotes_buffer[symbol].append(data)
            self.stats['messages_received'] += 1
            self.stats['last_update'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error handling quote: {e}")
    
    async def _handle_trade(self, t):
        """Handler para trades (ejecuciones)"""
        try:
            symbol = t.symbol
            self._init_buffers(symbol)
            
            data = {
                'price': float(t.price),
                'size': int(t.size),
                'conditions': t.conditions
            }
            
            self.trades_buffer[symbol].append(data)
            self.stats['messages_received'] += 1
            self.stats['last_update'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error handling trade: {e}")
    
    async def _handle_bar(self, bar):
        """Handler para bars (OHLCV agregados)"""
        try:
            symbol = bar.symbol
            self._init_buffers(symbol)
            
            data = {
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume),
                'vwap': float(bar.vwap) if hasattr(bar, 'vwap') else None
            }
            
            self.bars_buffer[symbol].append(data)
            
        except Exception as e:
            logger.error(f"Error handling bar: {e}")
    
    def subscribe(self, symbols: List[str], data_types: List[str] = ['quotes', 'trades']):
        """
        Suscribe a símbolos para streaming
        
        Args:
            symbols: Lista de símbolos ['AAPL', 'TSLA', etc]
            data_types: Tipos de datos ['quotes', 'trades', 'bars']
        """
        for symbol in symbols:
            self._init_buffers(symbol)
            self.subscribed_symbols.add(symbol)
        
        # Subscribir según tipo
        if 'quotes' in data_types:
            self.stream.subscribe_quotes(self._handle_quote, *symbols)
        
        if 'trades' in data_types:
            self.stream.subscribe_trades(self._handle_trade, *symbols)
        
        if 'bars' in data_types:
            self.stream.subscribe_bars(self._handle_bar, *symbols)
        
        logger.info(f"✅ Subscribed to {len(symbols)} symbols: {data_types}")
    
    def unsubscribe(self, symbols: List[str] = None):
        """Cancela suscripción y limpia buffers"""
        if symbols is None:
            symbols = list(self.subscribed_symbols)
        
        for symbol in symbols:
            if symbol in self.quotes_buffer:
                self.quotes_buffer[symbol].clear()
                del self.quotes_buffer[symbol]
            
            if symbol in self.trades_buffer:
                self.trades_buffer[symbol].clear()
                del self.trades_buffer[symbol]
            
            if symbol in self.bars_buffer:
                self.bars_buffer[symbol].clear()
                del self.bars_buffer[symbol]
            
            self.subscribed_symbols.discard(symbol)
        
        logger.info(f"🗑️ Unsubscribed from {len(symbols)} symbols")
    
    def start(self):
        """Inicia el loop de consulta en background thread"""
        if self.is_running:
            return
        
        self.is_running = True
        
        def run_stream():
            logger.info("🚀 Iniciando motor de consulta REST (Anti-Bloqueos)...")
            while self.is_running:
                for symbol in list(self.subscribed_symbols):
                    try:
                        # 1. Pedir último Trade
                        trade = self.api.get_latest_trade(symbol)
                        # Reutilizamos tus handlers originales para no romper tu lógica
                        self.trades_buffer[symbol].append({
                            'price': float(trade.price),
                            'size': int(trade.size),
                            'conditions': getattr(trade, 'conditions', [])
                        })
                        
                        # 2. Pedir último Quote
                        quote = self.api.get_latest_quote(symbol)
                        self.quotes_buffer[symbol].append({
                            'bid': float(quote.bid_price),
                            'ask': float(quote.ask_price),
                            'bid_size': int(quote.bid_size),
                            'ask_size': int(quote.ask_size),
                            'spread': float(quote.ask_price - quote.bid_price)
                        })
                        
                        self.stats['messages_received'] += 1
                        self.stats['last_update'] = datetime.now()
                        
                    except Exception as e:
                        logger.error(f"Error actualizando {symbol}: {e}")
                
                # Pausa de seguridad para evitar Error 429
                time.sleep(1.5) 
        
        self.stream_thread = threading.Thread(target=run_stream, daemon=True)
        self.stream_thread.start()
        logger.info("✅ Motor iniciado correctamente.")
    
    def stop(self):
        """Detiene streaming"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        try:
            self.stream.stop()
        except:
            pass
        
        logger.info("⏸️ Stream stopped")
    
    # ============================================================================
    # MÉTODOS DE ACCESO A DATOS (Thread-safe)
    # ============================================================================
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Obtiene último precio (del último trade)"""
        if symbol not in self.trades_buffer:
            return None
        
        latest = self.trades_buffer[symbol].get_latest(1)
        if latest:
            return latest[0]['price']
        return None
    
    def get_latest_quote(self, symbol: str) -> Optional[Dict]:
        """Obtiene último bid/ask"""
        if symbol not in self.quotes_buffer:
            return None
        
        latest = self.quotes_buffer[symbol].get_latest(1)
        if latest:
            return {
                'bid': latest[0]['bid'],
                'ask': latest[0]['ask'],
                'spread': latest[0]['spread'],
                'mid': (latest[0]['bid'] + latest[0]['ask']) / 2
            }
        return None
    
    def get_recent_trades(self, symbol: str, n: int = 100) -> List[Dict]:
        """Obtiene últimos N trades"""
        if symbol not in self.trades_buffer:
            return []
        
        return self.trades_buffer[symbol].get_latest(n)
    
    def get_ohlcv_bars(self, symbol: str) -> pd.DataFrame:
        """Obtiene bars como DataFrame"""
        if symbol not in self.bars_buffer:
            return pd.DataFrame()
        
        bars = self.bars_buffer[symbol].get_all()
        if not bars:
            return pd.DataFrame()
        
        df = pd.DataFrame(bars)
        df['timestamp'] = pd.to_datetime([b['_timestamp'] for b in bars])
        df.set_index('timestamp', inplace=True)
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def get_aggregated_data(self, symbol: str, interval_seconds: int = 60) -> Dict:
        """
        Agrega datos en intervalos para reducir volumen
        
        Args:
            symbol: Símbolo
            interval_seconds: Intervalo de agregación (default 1 min)
        
        Returns:
            Dict con OHLCV agregado del intervalo
        """
        trades = self.get_recent_trades(symbol, n=1000)
        
        if not trades:
            return None
        
        # Filtrar últimos N segundos
        cutoff_time = datetime.now() - timedelta(seconds=interval_seconds)
        recent_trades = [
            t for t in trades 
            if t['_timestamp'] >= cutoff_time
        ]
        
        if not recent_trades:
            return None
        
        prices = [t['price'] for t in recent_trades]
        volumes = [t['size'] for t in recent_trades]
        
        return {
            'open': prices[0],
            'high': max(prices),
            'low': min(prices),
            'close': prices[-1],
            'volume': sum(volumes),
            'vwap': sum(p*v for p, v in zip(prices, volumes)) / sum(volumes) if sum(volumes) > 0 else None,
            'trades_count': len(recent_trades)
        }
    
    # ============================================================================
    # MONITOREO Y GESTIÓN DE MEMORIA
    # ============================================================================
    
    def get_memory_stats(self) -> Dict:
        """Obtiene estadísticas de uso de memoria"""
        total_memory = 0
        symbol_memory = {}
        
        for symbol in self.subscribed_symbols:
            memory = 0
            
            if symbol in self.quotes_buffer:
                memory += self.quotes_buffer[symbol].memory_usage_mb()
            
            if symbol in self.trades_buffer:
                memory += self.trades_buffer[symbol].memory_usage_mb()
            
            if symbol in self.bars_buffer:
                memory += self.bars_buffer[symbol].memory_usage_mb()
            
            symbol_memory[symbol] = memory
            total_memory += memory
        
        return {
            'total_mb': total_memory,
            'by_symbol': symbol_memory,
            'symbols_count': len(self.subscribed_symbols),
            'avg_per_symbol_mb': total_memory / len(self.subscribed_symbols) if self.subscribed_symbols else 0
        }
    
    def get_buffer_stats(self) -> Dict:
        """Obtiene estadísticas de buffers"""
        stats = {}
        
        for symbol in self.subscribed_symbols:
            stats[symbol] = {
                'quotes': self.quotes_buffer[symbol].size() if symbol in self.quotes_buffer else 0,
                'trades': self.trades_buffer[symbol].size() if symbol in self.trades_buffer else 0,
                'bars': self.bars_buffer[symbol].size() if symbol in self.bars_buffer else 0
            }
        
        return stats
    
    def cleanup_old_data(self):
        """Fuerza limpieza de datos viejos en todos los buffers"""
        for symbol in self.subscribed_symbols:
            if symbol in self.quotes_buffer:
                self.quotes_buffer[symbol]._clean_old_data()
            
            if symbol in self.trades_buffer:
                self.trades_buffer[symbol]._clean_old_data()
            
            if symbol in self.bars_buffer:
                self.bars_buffer[symbol]._clean_old_data()
        
        logger.info("🧹 Cleanup completed")
    
    def get_status(self) -> Dict:
        """Obtiene estado completo del streamer"""
        memory = self.get_memory_stats()
        buffers = self.get_buffer_stats()
        
        return {
            'is_running': self.is_running,
            'subscribed_symbols': list(self.subscribed_symbols),
            'symbols_count': len(self.subscribed_symbols),
            'messages_received': self.stats['messages_received'],
            'last_update': self.stats['last_update'].isoformat() if self.stats['last_update'] else None,
            'memory_usage_mb': memory['total_mb'],
            'avg_memory_per_symbol_mb': memory['avg_per_symbol_mb'],
            'buffer_sizes': buffers
        }


# ============================================================================
# FUNCIONES HELPER PARA STREAMLIT
# ============================================================================

def init_realtime_streamer(st, alpaca_key: str, alpaca_secret: str, 
                          symbols: List[str], paper: bool = True):
    """
    Inicializa streamer en Streamlit session_state
    
    Args:
        st: Streamlit module
        alpaca_key: API key
        alpaca_secret: API secret
        symbols: Lista de símbolos a monitorear
        paper: True para paper trading
    
    Returns:
        RealTimeStreamer instance
    """
    if 'realtime_streamer' not in st.session_state:
        # Inicializar con límites de memoria conservadores
        st.session_state.realtime_streamer = RealTimeStreamer(
            alpaca_key=alpaca_key,
            alpaca_secret=alpaca_secret,
            paper=paper,
            max_buffer_size=500,      # 500 elementos por símbolo
            data_retention_seconds=1800  # 30 minutos
        )
        
        # Subscribir a símbolos
        st.session_state.realtime_streamer.subscribe(
            symbols=symbols,
            data_types=['quotes', 'trades']  # No bars para ahorrar memoria
        )
        
        # Iniciar stream
        st.session_state.realtime_streamer.start()
        
        logger.info(f"✅ RealTime streamer initialized for {len(symbols)} symbols")
    
    return st.session_state.realtime_streamer


def display_realtime_metrics(st, streamer: RealTimeStreamer, symbol: str):
    """
    Muestra métricas en tiempo real en Streamlit
    
    Args:
        st: Streamlit module
        streamer: RealTimeStreamer instance
        symbol: Símbolo a mostrar
    """
    # Obtener datos
    price = streamer.get_latest_price(symbol)
    quote = streamer.get_latest_quote(symbol)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if price:
            st.metric("💵 Último Precio", f"${price:.2f}")
        else:
            st.metric("💵 Último Precio", "Esperando...")
    
    with col2:
        if quote:
            st.metric("📊 Bid", f"${quote['bid']:.2f}")
    
    with col3:
        if quote:
            st.metric("📊 Ask", f"${quote['ask']:.2f}")
    
    with col4:
        if quote:
            spread_bps = (quote['spread'] / quote['mid']) * 10000
            st.metric("📏 Spread", f"{spread_bps:.1f} bps")


def display_memory_monitor(st, streamer: RealTimeStreamer):
    """
    Muestra monitor de memoria del streamer
    
    Args:
        st: Streamlit module
        streamer: RealTimeStreamer instance
    """
    status = streamer.get_status()
    memory = status['memory_usage_mb']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Color según uso de memoria
        color = "🟢" if memory < 50 else "🟡" if memory < 100 else "🔴"
        st.metric(f"{color} Memoria Total", f"{memory:.2f} MB")
    
    with col2:
        st.metric("📊 Símbolos Activos", status['symbols_count'])
    
    with col3:
        st.metric("📨 Mensajes Recibidos", f"{status['messages_received']:,}")
    
    # Warning si memoria alta
    if memory > 100:
        st.warning("⚠️ Alto uso de memoria. Considera reducir símbolos o retención de datos.")
