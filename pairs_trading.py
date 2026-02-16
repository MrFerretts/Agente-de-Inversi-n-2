"""
PAIRS TRADING SYSTEM
Sistema completo de arbitraje estadístico mediante pares cointegrados
Estrategia market-neutral de bajo riesgo
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.stattools import coint, adfuller
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PairsFinder:
    """
    Encuentra pares de acciones cointegradas
    Cointegración = Relación estadística de largo plazo
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Args:
            significance_level: Nivel de significancia para test (default 5%)
        """
        self.significance_level = significance_level
        self.pairs_found = []
        
    def find_cointegrated_pairs(self, 
                                prices_dict: Dict[str, pd.Series],
                                min_correlation: float = 0.7) -> List[Dict]:
        """
        Encuentra todos los pares cointegrados en un universo de activos
        
        Args:
            prices_dict: Dict {ticker: price_series}
            min_correlation: Correlación mínima requerida (filtro previo)
        
        Returns:
            Lista de pares con estadísticas
        """
        tickers = list(prices_dict.keys())
        pairs = []
        
        logger.info(f"🔍 Buscando pares cointegrados en {len(tickers)} activos...")
        
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                ticker1 = tickers[i]
                ticker2 = tickers[j]
                
                series1 = prices_dict[ticker1]
                series2 = prices_dict[ticker2]
                
                # Filtro 1: Correlación mínima
                correlation = series1.corr(series2)
                
                if abs(correlation) < min_correlation:
                    continue
                
                # Test de cointegración
                score, pvalue, _ = coint(series1, series2)
                
                # Si pvalue < 0.05, hay cointegración
                if pvalue < self.significance_level:
                    # Calcular estadísticas adicionales
                    pair_stats = self._calculate_pair_stats(series1, series2, ticker1, ticker2)
                    pair_stats.update({
                        'coint_pvalue': pvalue,
                        'coint_score': score,
                        'correlation': correlation
                    })
                    
                    pairs.append(pair_stats)
                    
                    logger.info(f"✅ Par encontrado: {ticker1} <-> {ticker2} (p={pvalue:.4f})")
        
        # Ordenar por p-value (menor = mejor)
        pairs.sort(key=lambda x: x['coint_pvalue'])
        
        self.pairs_found = pairs
        logger.info(f"🎯 Total de pares encontrados: {len(pairs)}")
        
        return pairs
    
    def _calculate_pair_stats(self, series1: pd.Series, series2: pd.Series,
                             ticker1: str, ticker2: str) -> Dict:
        """Calcula estadísticas adicionales del par"""
        
        # Calcular hedge ratio (beta)
        hedge_ratio = np.polyfit(series2, series1, 1)[0]
        
        # Calcular spread
        spread = series1 - hedge_ratio * series2
        
        # Estadísticas del spread
        spread_mean = spread.mean()
        spread_std = spread.std()
        
        # Half-life (velocidad de reversión)
        half_life = self._calculate_half_life(spread)
        
        # Volatilidad del spread
        spread_returns = spread.pct_change().dropna()
        spread_volatility = spread_returns.std() * np.sqrt(252)  # Anualizada
        
        return {
            'ticker1': ticker1,
            'ticker2': ticker2,
            'hedge_ratio': hedge_ratio,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'half_life': half_life,
            'spread_volatility': spread_volatility,
            'current_spread': spread.iloc[-1]
        }
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calcula half-life del spread (velocidad de reversión a la media)
        Half-life bajo = reversión rápida = mejor para trading
        """
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        
        # Regresión: spread_diff = lambda * spread_lag + epsilon
        spread_lag = spread_lag.dropna()
        spread_diff = spread_diff.dropna()
        
        # Alinear series
        spread_diff = spread_diff.iloc[1:]
        spread_lag = spread_lag.iloc[1:]
        
        # Regresión lineal
        slope = np.polyfit(spread_lag, spread_diff, 1)[0]
        
        if slope >= 0:
            return np.inf  # No hay reversión
        
        half_life = -np.log(2) / slope
        
        return half_life
    
    def get_best_pairs(self, n: int = 5) -> List[Dict]:
        """
        Retorna los N mejores pares
        
        Criterios:
        1. p-value más bajo (cointegración más fuerte)
        2. Half-life razonable (2-30 días)
        3. Volatilidad del spread manejable
        """
        if not self.pairs_found:
            return []
        
        # Filtrar por half-life razonable
        good_pairs = [
            p for p in self.pairs_found 
            if 2 <= p['half_life'] <= 30
        ]
        
        return good_pairs[:n]


class PairsTrader:
    """
    Ejecuta estrategia de pairs trading
    """
    
    def __init__(self, 
                 entry_z_score: float = 2.0,
                 exit_z_score: float = 0.5,
                 stop_loss_z_score: float = 3.5):
        """
        Args:
            entry_z_score: Z-score para abrir posición (default 2.0)
            exit_z_score: Z-score para cerrar posición (default 0.5)
            stop_loss_z_score: Z-score de stop loss (default 3.5)
        """
        self.entry_z = entry_z_score
        self.exit_z = exit_z_score
        self.stop_z = stop_loss_z_score
        
    def calculate_spread(self, 
                        price1: pd.Series, 
                        price2: pd.Series,
                        hedge_ratio: float) -> pd.Series:
        """
        Calcula spread ajustado por hedge ratio
        
        Spread = Stock1 - (hedge_ratio * Stock2)
        """
        return price1 - hedge_ratio * price2
    
    def calculate_z_score(self, 
                         spread: pd.Series,
                         lookback: int = 20) -> pd.Series:
        """
        Calcula Z-score del spread
        
        Z-score = (spread - mean) / std
        
        Interpretación:
        - Z > 2: Spread muy alto → SHORT Stock1, LONG Stock2
        - Z < -2: Spread muy bajo → LONG Stock1, SHORT Stock2
        - |Z| < 0.5: Convergencia → CERRAR posición
        """
        rolling_mean = spread.rolling(window=lookback).mean()
        rolling_std = spread.rolling(window=lookback).std()
        
        z_score = (spread - rolling_mean) / (rolling_std + 1e-9)
        
        return z_score
    
    def generate_signals(self,
                        ticker1: str,
                        ticker2: str,
                        price1: pd.Series,
                        price2: pd.Series,
                        hedge_ratio: float,
                        lookback: int = 20) -> Dict:
        """
        Genera señales de trading para un par
        
        Returns:
            Dict con señales y estadísticas
        """
        # Calcular spread y z-score
        spread = self.calculate_spread(price1, price2, hedge_ratio)
        z_score = self.calculate_z_score(spread, lookback)
        
        current_z = z_score.iloc[-1]
        prev_z = z_score.iloc[-2] if len(z_score) > 1 else 0
        
        # Generar señales
        signal = self._determine_signal(current_z, prev_z)
        
        # Calcular tamaño de posición sugerido
        position_size = self._calculate_position_size(current_z)
        
        return {
            'ticker1': ticker1,
            'ticker2': ticker2,
            'signal': signal['action'],
            'signal_strength': signal['strength'],
            'current_z_score': current_z,
            'spread': spread.iloc[-1],
            'spread_mean': spread.mean(),
            'spread_std': spread.std(),
            'position_size_pct': position_size,
            'hedge_ratio': hedge_ratio,
            'details': signal['details']
        }
    
    def _determine_signal(self, current_z: float, prev_z: float) -> Dict:
        """Determina la señal de trading basada en z-score"""
        
        # SEÑAL DE APERTURA
        
        # Spread muy alto → SHORT par (vender Stock1, comprar Stock2)
        if current_z > self.entry_z:
            return {
                'action': 'SHORT_PAIR',
                'strength': 'STRONG' if current_z > 2.5 else 'MODERATE',
                'details': f'Spread {current_z:.2f} desviaciones sobre la media. SHORT {ticker1}, LONG {ticker2}'
            }
        
        # Spread muy bajo → LONG par (comprar Stock1, vender Stock2)
        elif current_z < -self.entry_z:
            return {
                'action': 'LONG_PAIR',
                'strength': 'STRONG' if current_z < -2.5 else 'MODERATE',
                'details': f'Spread {abs(current_z):.2f} desviaciones bajo la media. LONG {ticker1}, SHORT {ticker2}'
            }
        
        # SEÑAL DE CIERRE (convergencia)
        
        elif abs(current_z) < self.exit_z:
            return {
                'action': 'CLOSE',
                'strength': 'TAKE_PROFIT',
                'details': f'Spread convergió (Z={current_z:.2f}). Cerrar posición.'
            }
        
        # STOP LOSS
        
        elif abs(current_z) > self.stop_z:
            return {
                'action': 'STOP_LOSS',
                'strength': 'EMERGENCY',
                'details': f'Stop loss activado (Z={current_z:.2f}). Spread divergió demasiado.'
            }
        
        # SIN SEÑAL
        
        else:
            return {
                'action': 'HOLD',
                'strength': 'NEUTRAL',
                'details': f'Z-score en zona neutral ({current_z:.2f}). Esperar.'
            }
    
    def _calculate_position_size(self, z_score: float) -> float:
        """
        Calcula tamaño de posición basado en z-score
        Mayor z-score = mayor confianza = mayor tamaño
        """
        abs_z = abs(z_score)
        
        if abs_z < self.entry_z:
            return 0  # No operar
        
        # Tamaño base: 5% del capital
        base_size = 5.0
        
        # Ajustar según fuerza de señal
        if abs_z > 3.0:
            return base_size * 1.5  # 7.5% del capital
        elif abs_z > 2.5:
            return base_size * 1.2  # 6% del capital
        else:
            return base_size  # 5% del capital
    
    def backtest_pair(self,
                     ticker1: str,
                     ticker2: str,
                     price1: pd.Series,
                     price2: pd.Series,
                     hedge_ratio: float,
                     initial_capital: float = 10000) -> Dict:
        """
        Backtest de estrategia de pairs trading
        
        Returns:
            Métricas de performance
        """
        spread = self.calculate_spread(price1, price2, hedge_ratio)
        z_score = self.calculate_z_score(spread, lookback=20)
        
        # Simular trades
        positions = []  # Lista de (entry_z, entry_date, direction)
        trades = []
        capital = initial_capital
        
        position_open = False
        entry_z = 0
        direction = 0  # 1 = LONG pair, -1 = SHORT pair
        
        for i in range(20, len(z_score)):
            current_z = z_score.iloc[i]
            
            # Lógica de entrada
            if not position_open:
                if current_z > self.entry_z:
                    # SHORT pair
                    position_open = True
                    entry_z = current_z
                    direction = -1
                
                elif current_z < -self.entry_z:
                    # LONG pair
                    position_open = True
                    entry_z = current_z
                    direction = 1
            
            # Lógica de salida
            else:
                exit_signal = False
                profit = 0
                
                # Take profit (convergencia)
                if abs(current_z) < self.exit_z:
                    profit = (entry_z - current_z) * direction
                    exit_signal = True
                
                # Stop loss
                elif abs(current_z) > self.stop_z:
                    profit = (entry_z - current_z) * direction
                    exit_signal = True
                
                if exit_signal:
                    # Registrar trade
                    trades.append({
                        'entry_z': entry_z,
                        'exit_z': current_z,
                        'profit_z': profit,
                        'direction': 'LONG' if direction == 1 else 'SHORT'
                    })
                    
                    # Actualizar capital (simplificado)
                    capital += profit * 100  # Cada punto de Z = $100
                    
                    position_open = False
        
        # Calcular métricas
        if not trades:
            return {'error': 'No trades executed'}
        
        profits = [t['profit_z'] for t in trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(trades) * 100,
            'avg_profit': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'profit_factor': sum(wins) / abs(sum(losses)) if losses else np.inf,
            'total_return': (capital - initial_capital) / initial_capital * 100,
            'sharpe_ratio': np.mean(profits) / np.std(profits) if np.std(profits) > 0 else 0,
            'max_consecutive_losses': self._calc_max_consecutive_losses(profits)
        }
    
    def _calc_max_consecutive_losses(self, profits: List[float]) -> int:
        """Calcula máxima racha de pérdidas consecutivas"""
        max_streak = 0
        current_streak = 0
        
        for p in profits:
            if p < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak


# ============================================================================
# PARES PRE-DEFINIDOS (Clásicos que funcionan bien)
# ============================================================================

CLASSIC_PAIRS = [
    # Bebidas
    ('KO', 'PEP'),      # Coca-Cola vs Pepsi
    
    # Fast Food
    ('MCD', 'YUM'),     # McDonald's vs Yum Brands
    
    # Tech
    ('AAPL', 'MSFT'),   # Apple vs Microsoft
    ('GOOGL', 'META'),  # Google vs Meta
    
    # Retail
    ('WMT', 'TGT'),     # Walmart vs Target
    ('HD', 'LOW'),      # Home Depot vs Lowe's
    
    # Finance
    ('JPM', 'BAC'),     # JP Morgan vs Bank of America
    ('GS', 'MS'),       # Goldman Sachs vs Morgan Stanley
    
    # Airlines
    ('AAL', 'UAL'),     # American vs United
    ('DAL', 'LUV'),     # Delta vs Southwest
    
    # Oil & Gas
    ('XOM', 'CVX'),     # Exxon vs Chevron
    
    # Telecom
    ('VZ', 'T'),        # Verizon vs AT&T
]


def get_classic_pairs() -> List[Tuple[str, str]]:
    """Retorna lista de pares clásicos que históricamente funcionan bien"""
    return CLASSIC_PAIRS


# ============================================================================
# FUNCIONES HELPER
# ============================================================================

def format_pairs_report(pairs: List[Dict]) -> str:
    """Formatea reporte de pares encontrados"""
    
    if not pairs:
        return "❌ No se encontraron pares cointegrados"
    
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║          PARES COINTEGRADOS ENCONTRADOS ({len(pairs)})                     
║
╚══════════════════════════════════════════════════════════════╝

"""
    
    for i, pair in enumerate(pairs[:10], 1):  # Top 10
        report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PAR #{i}: {pair['ticker1']} <-> {pair['ticker2']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Estadísticas:
   • P-value:        {pair['coint_pvalue']:.4f} {'✅ EXCELENTE' if pair['coint_pvalue'] < 0.01 else '✅ BUENO'}
   • Correlación:    {pair['correlation']:.3f}
   • Hedge Ratio:    {pair['hedge_ratio']:.4f}
   • Half-life:      {pair['half_life']:.1f} días {'✅ ÓPTIMO' if 2 <= pair['half_life'] <= 30 else '⚠️ SUBÓPTIMO'}
   
💹 Spread Actual:
   • Valor:          {pair['current_spread']:.4f}
   • Mean:           {pair['spread_mean']:.4f}
   • Std Dev:        {pair['spread_std']:.4f}
   • Volatilidad:    {pair['spread_volatility']*100:.1f}%
"""
    
    return report
