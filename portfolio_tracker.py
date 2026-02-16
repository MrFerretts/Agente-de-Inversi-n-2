"""
PORTFOLIO TRACKING SYSTEM
Sistema completo de seguimiento de portfolio con trades, P&L, métricas
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PortfolioTracker:
    """
    Sistema de tracking de portfolio
    Maneja posiciones abiertas, historial de trades, P&L, métricas
    """
    
    def __init__(self, data_file: str = "data/portfolio.json"):
        self.data_file = data_file
        self.portfolio = self._load_portfolio()
        
    def _load_portfolio(self) -> Dict:
        """Carga portfolio desde archivo"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'positions': [],
                'closed_trades': [],
                'initial_capital': 10000,
                'current_capital': 10000,
                'start_date': datetime.now().isoformat()
            }
    
    def _save_portfolio(self):
        """Guarda portfolio a archivo"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        with open(self.data_file, 'w') as f:
            json.dump(self.portfolio, f, indent=2)
    
    def add_position(self, ticker: str, entry_price: float, shares: int,
                    stop_loss: float, take_profit: float, 
                    strategy: str = "Manual", notes: str = "") -> Dict:
        """
        Abre una nueva posición
        
        Args:
            ticker: Símbolo
            entry_price: Precio de entrada
            shares: Número de acciones
            stop_loss: Stop loss
            take_profit: Take profit
            strategy: Nombre de la estrategia
            notes: Notas adicionales
        
        Returns:
            Dict con la posición creada
        """
        position = {
            'id': len(self.portfolio['positions']) + len(self.portfolio['closed_trades']) + 1,
            'ticker': ticker,
            'entry_price': entry_price,
            'shares': shares,
            'entry_date': datetime.now().isoformat(),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': entry_price * shares,
            'strategy': strategy,
            'notes': notes,
            'status': 'open'
        }
        
        self.portfolio['positions'].append(position)
        self.portfolio['current_capital'] -= position['position_size']
        self._save_portfolio()
        
        return position
    
    def close_position(self, position_id: int, exit_price: float, 
                      reason: str = "Manual") -> Dict:
        """
        Cierra una posición
        
        Args:
            position_id: ID de la posición
            exit_price: Precio de salida
            reason: Razón del cierre
        
        Returns:
            Dict con el trade cerrado
        """
        # Buscar posición
        position = None
        for i, pos in enumerate(self.portfolio['positions']):
            if pos['id'] == position_id:
                position = self.portfolio['positions'].pop(i)
                break
        
        if position is None:
            raise ValueError(f"Posición {position_id} no encontrada")
        
        # Calcular P&L
        exit_value = exit_price * position['shares']
        pnl = exit_value - position['position_size']
        pnl_pct = (pnl / position['position_size']) * 100
        
        # Calcular duración
        entry_date = datetime.fromisoformat(position['entry_date'])
        exit_date = datetime.now()
        duration_days = (exit_date - entry_date).days
        
        # Crear trade cerrado
        closed_trade = {
            **position,
            'exit_price': exit_price,
            'exit_date': exit_date.isoformat(),
            'exit_value': exit_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'duration_days': duration_days,
            'exit_reason': reason,
            'status': 'closed'
        }
        
        self.portfolio['closed_trades'].append(closed_trade)
        self.portfolio['current_capital'] += exit_value
        self._save_portfolio()
        
        return closed_trade
    
    def update_positions(self, current_prices: Dict[str, float]):
        """
        Actualiza todas las posiciones con precios actuales
        
        Args:
            current_prices: Dict {ticker: price}
        """
        for position in self.portfolio['positions']:
            if position['ticker'] in current_prices:
                current_price = current_prices[position['ticker']]
                position['current_price'] = current_price
                position['current_value'] = current_price * position['shares']
                position['unrealized_pnl'] = position['current_value'] - position['position_size']
                position['unrealized_pnl_pct'] = (position['unrealized_pnl'] / position['position_size']) * 100
        
        self._save_portfolio()
    
    def get_open_positions(self) -> List[Dict]:
        """Retorna posiciones abiertas"""
        return self.portfolio['positions']
    
    def get_closed_trades(self) -> List[Dict]:
        """Retorna trades cerrados"""
        return self.portfolio['closed_trades']
    
    def calculate_metrics(self) -> Dict:
        """
        Calcula métricas del portfolio
        
        Returns:
            Dict con métricas completas
        """
        closed = self.portfolio['closed_trades']
        
        if not closed:
            return {
                'total_trades': 0,
                'winning_trades': 0, # ← Agregamos esta
                'losing_trades': 0,  # ← Agregamos esta
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_win_pct': 0.0,  # ← Agregamos esta
                'avg_loss': 0.0,
                'avg_loss_pct': 0.0, # ← Agregamos esta
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0,
                'total_pnl': 0.0,
                'initial_capital': self.portfolio['initial_capital'],
                'current_capital': self.portfolio['current_capital']
            }
        
        # Calcular wins y losses
        wins = [t['pnl'] for t in closed if t['pnl'] > 0]
        losses = [t['pnl'] for t in closed if t['pnl'] < 0]
        
        win_rate = len(wins) / len(closed) * 100 if closed else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Returns para Sharpe
        returns = [t['pnl_pct'] for t in closed]
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Max drawdown
        equity_curve = []
        running_capital = self.portfolio['initial_capital']
        for trade in closed:
            running_capital += trade['pnl']
            equity_curve.append(running_capital)
        
        if equity_curve:
            peak = equity_curve[0]
            max_dd = 0
            for value in equity_curve:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak * 100
                if dd > max_dd:
                    max_dd = dd
        else:
            max_dd = 0
        
        # Total return
        total_return = ((self.portfolio['current_capital'] - self.portfolio['initial_capital']) / 
                       self.portfolio['initial_capital'] * 100)
        
        return {
            'total_trades': len(closed),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_win_pct': np.mean([t['pnl_pct'] for t in closed if t['pnl'] > 0]) if wins else 0,
            'avg_loss': avg_loss,
            'avg_loss_pct': np.mean([t['pnl_pct'] for t in closed if t['pnl'] < 0]) if losses else 0,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'total_return': total_return,
            'total_pnl': sum([t['pnl'] for t in closed]),
            'initial_capital': self.portfolio['initial_capital'],
            'current_capital': self.portfolio['current_capital']
        }
    
    def get_equity_curve(self) -> Tuple[List, List]:
        """
        Genera equity curve
        
        Returns:
            Tuple (dates, equity_values)
        """
        closed = self.portfolio['closed_trades']
        
        if not closed:
            return [datetime.now()], [self.portfolio['initial_capital']]
        
        dates = []
        equity = []
        
        running_capital = self.portfolio['initial_capital']
        dates.append(datetime.fromisoformat(self.portfolio['start_date']))
        equity.append(running_capital)
        
        for trade in closed:
            exit_date = datetime.fromisoformat(trade['exit_date'])
            running_capital += trade['pnl']
            dates.append(exit_date)
            equity.append(running_capital)
        
        return dates, equity
    
    def create_equity_chart(self) -> go.Figure:
        """Crea gráfico de equity curve"""
        dates, equity = self.get_equity_curve()
        
        fig = go.Figure()
        
        # Equity curve
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity,
            mode='lines',
            name='Equity',
            line=dict(color='#00ff88', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.1)'
        ))
        
        # Línea de capital inicial
        fig.add_hline(
            y=self.portfolio['initial_capital'],
            line_dash="dash",
            line_color="gray",
            annotation_text="Capital Inicial"
        )
        
        fig.update_layout(
            title="📈 Equity Curve",
            xaxis_title="Fecha",
            yaxis_title="Capital ($)",
            template="plotly_dark",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def create_pnl_distribution(self) -> go.Figure:
        """Crea histograma de distribución de P&L"""
        closed = self.portfolio['closed_trades']
        
        if not closed:
            return None
        
        pnl_pcts = [t['pnl_pct'] for t in closed]
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=pnl_pcts,
            nbinsx=20,
            marker_color='#00ff88',
            name='Trades'
        ))
        
        fig.update_layout(
            title="📊 Distribución de Returns (%)",
            xaxis_title="Return (%)",
            yaxis_title="Frecuencia",
            template="plotly_dark",
            height=300
        )
        
        return fig
    
    def get_summary(self) -> str:
        """Genera resumen del portfolio en texto"""
        metrics = self.calculate_metrics()
        positions = self.get_open_positions()
        
        summary = f"""
## 💼 Portfolio Summary

### Capital
- **Inicial:** ${self.portfolio['initial_capital']:,.2f}
- **Actual:** ${self.portfolio['current_capital']:,.2f}
- **Total Return:** {metrics['total_return']:.2f}%

### Posiciones
- **Abiertas:** {len(positions)}
- **Cerradas:** {metrics['total_trades']}

### Performance
- **Win Rate:** {metrics['win_rate']:.1f}%
- **Profit Factor:** {metrics['profit_factor']:.2f}
- **Sharpe Ratio:** {metrics['sharpe_ratio']:.2f}
- **Max Drawdown:** {metrics['max_drawdown']:.2f}%

### Trades
- **Ganadores:** {metrics['winning_trades']} (Avg: {metrics['avg_win_pct']:.2f}%)
- **Perdedores:** {metrics['losing_trades']} (Avg: {metrics['avg_loss_pct']:.2f}%)
"""
        return summary


# ============================================================================
# FUNCIONES HELPER PARA STREAMLIT
# ============================================================================

def display_portfolio_dashboard(tracker: PortfolioTracker, current_prices: Dict[str, float]):
    """
    Muestra dashboard completo del portfolio en Streamlit
    
    Args:
        tracker: PortfolioTracker instance
        current_prices: Dict con precios actuales
    """
    import streamlit as st
    
    # Actualizar posiciones con precios actuales
    tracker.update_positions(current_prices)
    
    # Métricas generales
    metrics = tracker.calculate_metrics()
    
    st.header("💼 Mi Portfolio")
    
    # Métricas principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        current_capital = tracker.portfolio['current_capital']
        st.metric(
            "Capital Total",
            f"${current_capital:,.0f}",
            delta=f"{metrics['total_return']:+.2f}%"
        )
    
    with col2:
       # Usamos .get() para evitar KeyErrors si una métrica falta por error
        win_trades = metrics.get('winning_trades', 0)
        total_trades = metrics.get('total_trades', 0)
        win_rate = metrics.get('win_rate', 0.0)
        
        delta_text = f"{win_trades}/{total_trades}" if total_trades > 0 else "Sin trades"
        
        st.metric(
            "Win Rate",
            f"{win_rate:.1f}%",
            delta=delta_text
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
            delta="🟢" if metrics['sharpe_ratio'] > 1.5 else "🟡" if metrics['sharpe_ratio'] > 1.0 else "🔴"
        )
    
    with col4:
        st.metric(
            "Profit Factor",
            f"{metrics['profit_factor']:.2f}",
            delta="🟢" if metrics['profit_factor'] > 2.0 else "🟡" if metrics['profit_factor'] > 1.5 else "🔴"
        )
    
    with col5:
        st.metric(
            "Max Drawdown",
            f"{metrics['max_drawdown']:.1f}%",
            delta="🟢" if metrics['max_drawdown'] < 10 else "🟡" if metrics['max_drawdown'] < 20 else "🔴"
        )
    
    st.markdown("---")
    
    # Equity curve
    fig = tracker.create_equity_chart()
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Tabs para diferentes vistas
    tab1, tab2, tab3 = st.tabs(["📊 Posiciones Abiertas", "📜 Historial", "📈 Análisis"])
    
    with tab1:
        positions = tracker.get_open_positions()
        
        if positions:
            # Crear DataFrame con nombres mejorados
            df_positions = pd.DataFrame([{
                'Ticker': p['ticker'],
                'Entrada': f"${p['entry_price']:.2f}",
                '💵 Precio Actual': f"${p.get('current_price', 0):.2f}",
                'Shares': p['shares'],
                '💰 Valor Total': f"${p.get('current_value', 0):.2f}",
                'P&L $': f"${p.get('unrealized_pnl', 0):.2f}",
                'P&L %': f"{p.get('unrealized_pnl_pct', 0):.2f}%",
                'Stop': f"${p['stop_loss']:.2f}",
                'Target': f"${p['take_profit']:.2f}",
                'Días': (datetime.now() - datetime.fromisoformat(p['entry_date'])).days,
                'Estrategia': p['strategy']
            } for p in positions])
            
            st.dataframe(df_positions, use_container_width=True, hide_index=True)
            
            st.caption("💵 **Precio Actual** = Precio de la acción ahora | 💰 **Valor Total** = Precio Actual × Shares")
            
            # Botones de acción por posición
            st.markdown("### ⚙️ Acciones")
            
            for pos in positions:
                with st.expander(f"🔧 {pos['ticker']} - {pos['strategy']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button(f"✅ Cerrar Posición", key=f"close_{pos['id']}", use_container_width=True):
                            current_price = current_prices.get(pos['ticker'], pos['entry_price'])
                            tracker.close_position(pos['id'], current_price, "Manual close")
                            st.success(f"✅ Posición {pos['ticker']} cerrada")
                            st.balloons()
                    
                    with col2:
                        if st.button(f"🗑️ Eliminar Sin Registrar", key=f"delete_{pos['id']}", use_container_width=True):
                            # Eliminar sin registrar en historial
                            tracker.portfolio['positions'] = [
                                p for p in tracker.portfolio['positions'] 
                                if p['id'] != pos['id']
                            ]
                            # Devolver capital
                            tracker.portfolio['current_capital'] += pos['position_size']
                            tracker._save_portfolio()
                            st.success(f"🗑️ {pos['ticker']} eliminada (sin registrar trade)")
                    
                    with col3:
                        st.caption(f"**Entrada:** ${pos['entry_price']:.2f}")
                        st.caption(f"**Stop:** ${pos['stop_loss']:.2f}")
                        st.caption(f"**Target:** ${pos['take_profit']:.2f}")
                        pnl = pos.get('unrealized_pnl', 0)
                        pnl_pct = pos.get('unrealized_pnl_pct', 0)
                        pnl_color = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                        st.caption(f"**P&L:** {pnl_color} ${pnl:.2f} ({pnl_pct:+.2f}%)")
        else:
            st.info("📭 No hay posiciones abiertas")
    
    with tab2:
        closed = tracker.get_closed_trades()
        
        if closed:
            df_closed = pd.DataFrame([{
                'Fecha': datetime.fromisoformat(t['exit_date']).strftime('%Y-%m-%d'),
                'Ticker': t['ticker'],
                'Entrada': f"${t['entry_price']:.2f}",
                'Salida': f"${t['exit_price']:.2f}",
                'P&L $': f"${t['pnl']:.2f}",
                'P&L %': f"{t['pnl_pct']:.2f}%",
                'Días': t['duration_days'],
                'Razón': t['exit_reason'],
                'Estrategia': t['strategy']
            } for t in sorted(closed, key=lambda x: x['exit_date'], reverse=True)])
            
            st.dataframe(df_closed, use_container_width=True, hide_index=True)
        else:
            st.info("📭 No hay trades cerrados")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Métricas Detalladas")
            
            st.metric("Total Trades", metrics['total_trades'])
            
            # Separar el cálculo del formato para evitar problemas
            win_text = f"{metrics['winning_trades']} ({metrics['win_rate']:.1f}%)"
            st.metric("Ganadores", win_text)
            
            loss_pct = 100 - metrics['win_rate']
            loss_text = f"{metrics['losing_trades']} ({loss_pct:.1f}%)"
            st.metric("Perdedores", loss_text)
            
            avg_win_text = f"${metrics['avg_win']:.2f} ({metrics['avg_win_pct']:.2f}%)"
            st.metric("Avg Win", avg_win_text)
            
            avg_loss_text = f"${metrics['avg_loss']:.2f} ({metrics['avg_loss_pct']:.2f}%)"
            st.metric("Avg Loss", avg_loss_text)
        
        with col2:
            # Distribución de P&L
            fig_dist = tracker.create_pnl_distribution()
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)

