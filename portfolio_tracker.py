"""
PORTFOLIO TRACKING SYSTEM - VERSIÓN DEFINITIVA BLINDADA
Solución al KeyError: 'winning_trades' y Diseño de Círculos
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from typing import Dict, List, Tuple
import plotly.graph_objects as go

class PortfolioTracker:
    def __init__(self, data_file: str = "data/portfolio.json"):
        self.data_file = data_file
        self.portfolio = self._load_portfolio()
        
    def _load_portfolio(self) -> Dict:
        """Carga el portfolio asegurando que existan todas las llaves necesarias"""
        defaults = {
            'positions': [],
            'closed_trades': [],
            'initial_capital': 10000.0,
            'current_capital': 10000.0,
            'start_date': datetime.now().isoformat()
        }
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                for key, val in defaults.items():
                    if key not in data: data[key] = val
                return data
            except: return defaults
        return defaults

    def _save_portfolio(self):
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        with open(self.data_file, 'w') as f:
            json.dump(self.portfolio, f, indent=2)

    def add_position(self, ticker, entry_price, shares, stop_loss, take_profit, strategy="Manual", notes=""):
        """Abre posición y descuenta capital"""
        pos_size = float(entry_price * shares)
        position = {
            'id': len(self.portfolio['positions']) + len(self.portfolio['closed_trades']) + 1,
            'ticker': ticker, 'entry_price': entry_price, 'shares': shares,
            'entry_date': datetime.now().isoformat(), 'stop_loss': stop_loss,
            'take_profit': take_profit, 'position_size': pos_size,
            'strategy': strategy, 'notes': notes, 'status': 'open'
        }
        self.portfolio['positions'].append(position)
        self.portfolio['current_capital'] -= pos_size
        self._save_portfolio()
        return position

    def close_position(self, position_id, exit_price, reason="Manual"):
        """Cierra posición y calcula P&L final"""
        for i, pos in enumerate(self.portfolio['positions']):
            if pos['id'] == position_id:
                position = self.portfolio['positions'].pop(i)
                exit_value = exit_price * position['shares']
                pnl = exit_value - position['position_size']
                closed_trade = {
                    **position, 'exit_price': exit_price, 'exit_date': datetime.now().isoformat(),
                    'pnl': pnl, 'pnl_pct': (pnl / position['position_size']) * 100,
                    'duration_days': (datetime.now() - datetime.fromisoformat(position['entry_date'])).days,
                    'exit_reason': reason, 'status': 'closed'
                }
                self.portfolio['closed_trades'].append(closed_trade)
                self.portfolio['current_capital'] += exit_value
                self._save_portfolio()
                return closed_trade
        return None

    def calculate_metrics(self) -> Dict:
        """Calcula todas las métricas garantizando que no falte ninguna llave"""
        closed = self.portfolio['closed_trades']
        initial = float(self.portfolio.get('initial_capital', 10000.0))
        
        # Valor total = Efectivo + Valor actual de posiciones
        open_val = sum([p.get('current_value', p['position_size']) for p in self.portfolio['positions']])
        total_equity = self.portfolio['current_capital'] + open_val
        
        # INICIALIZACIÓN COMPLETA (Evita KeyError)
        metrics = {
            'total_trades': len(closed), 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0.0, 'avg_win': 0.0, 'avg_win_pct': 0.0,
            'avg_loss': 0.0, 'avg_loss_pct': 0.0, 'profit_factor': 0.0,
            'total_return': ((total_equity - initial) / initial) * 100 if initial > 0 else 0,
            'current_equity': total_equity, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0
        }

        if not closed: return metrics

        # Cálculos si hay historial
        wins = [t['pnl'] for t in closed if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in closed if t['pnl'] < 0]
        
        metrics['winning_trades'] = len(wins)
        metrics['losing_trades'] = len(losses)
        metrics['win_rate'] = (len(wins) / len(closed)) * 100
        metrics['avg_win'] = np.mean(wins) if wins else 0
        metrics['avg_loss'] = np.mean(losses) if losses else 0
        metrics['avg_win_pct'] = np.mean([t['pnl_pct'] for t in closed if t['pnl'] > 0]) if wins else 0
        metrics['avg_loss_pct'] = np.mean([t['pnl_pct'] for t in closed if t['pnl'] < 0]) if losses else 0
        metrics['profit_factor'] = sum(wins) / sum(losses) if losses else 1.0
        
        return metrics

    def update_positions(self, current_prices: Dict[str, float]):
        for pos in self.portfolio['positions']:
            if pos['ticker'] in current_prices:
                price = current_prices[pos['ticker']]
                pos['current_price'] = price
                pos['current_value'] = price * pos['shares']
                pos['unrealized_pnl'] = pos['current_value'] - pos['position_size']
                pos['unrealized_pnl_pct'] = (pos['unrealized_pnl'] / pos['position_size']) * 100
        self._save_portfolio()

    def create_equity_chart(self) -> go.Figure:
        """Crea el gráfico de crecimiento de balance"""
        dates = [datetime.fromisoformat(self.portfolio['start_date'])]
        equity = [self.portfolio['initial_capital']]
        running = self.portfolio['initial_capital']
        for t in self.portfolio['closed_trades']:
            running += t['pnl']
            dates.append(datetime.fromisoformat(t['exit_date']))
            equity.append(running)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=equity, mode='lines+markers', name='Equity', line=dict(color='#00ff88', width=3), fill='tozeroy'))
        fig.update_layout(title="📈 Evolución del Capital", template="plotly_dark", height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig

# --- INTERFAZ DE DASHBOARD CON CÍRCULOS ---
def display_portfolio_dashboard(tracker, current_prices):
    import streamlit as st
    tracker.update_positions(current_prices)
    metrics = tracker.calculate_metrics()
    
    # UI Helper: Círculos Visuales
    def metric_circle_ui(titulo, valor, delta, sub):
        color = "#00c853" if "+" in str(delta) or "🟢" in str(delta) else "#d32f2f"
        st.markdown(f"""
        <div class="metric-card">
            <p style="color: #a0a0a0; font-size: 12px; margin-bottom: 5px; text-transform: uppercase;">{titulo}</p>
            <div class="metric-circle">
                <h3 style="color: #ffffff; margin: 0; font-size: 18px;">{valor}</h3>
            </div>
            <p style="color: {color}; font-size: 13px; margin-top: 5px; font-weight: bold;">{delta}</p>
            <p style="color: #666; font-size: 10px; margin:0;">{sub}</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("💼 Terminal de Gestión de Activos")
    
    # 1. KPIs EN CÍRCULOS
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_circle_ui("EQUITY TOTAL", f"${metrics['current_equity']:,.0f}", f"{metrics['total_return']:+.2f}%", "Retorno Total")
    with c2: metric_circle_ui("WIN RATE", f"{metrics['win_rate']:.1f}%", f"🟢 {metrics['winning_trades']}", f"de {metrics['total_trades']} trades")
    with c3: metric_circle_ui("PROFIT FACTOR", f"{metrics['profit_factor']:.2f}", "Factor", "Eficiencia")
    with c4: metric_circle_ui("MAX DD", f"{metrics['max_drawdown']:.1f}%", "Riesgo", "Drawdown Máx")
    with c5: metric_circle_ui("OPEN POS", f"{len(tracker.portfolio['positions'])}", "Activas", "Exposición")

    st.markdown("---")
    st.plotly_chart(tracker.create_equity_chart(), use_container_width=True)

    # 2. SECCIÓN DE POSICIONES Y HISTORIAL
    tab_a, tab_b, tab_c = st.tabs(["📊 Posiciones Activas", "📜 Historial de Trades", "🧠 Estadísticas IA"])
    
    with tab_a:
        if tracker.portfolio['positions']:
            df_p = pd.DataFrame([{
                'ID': p['id'], 'Ticker': p['ticker'], 'Entrada': f"${p['entry_price']:.2f}",
                'Actual': f"${p.get('current_price', 0):.2f}", 'Shares': p['shares'],
                'P&L $': f"${p.get('unrealized_pnl', 0):.2f}", 'P&L %': f"{p.get('unrealized_pnl_pct', 0):.2f}%",
                'Notas': p['notes']
            } for p in tracker.portfolio['positions']])
            st.dataframe(df_p, use_container_width=True, hide_index=True)
            
            # Botones de Cierre
            st.markdown("### ⚙️ Gestión Directa")
            for p in tracker.portfolio['positions']:
                with st.expander(f"🔧 Operar {p['ticker']} (ID: {p['id']})"):
                    col_x, col_y = st.columns(2)
                    if col_x.button(f"✅ Cerrar Posición {p['ticker']}", key=f"close_btn_{p['id']}", use_container_width=True):
                        tracker.close_position(p['id'], p.get('current_price', p['entry_price']))
                        st.success(f"Posición {p['ticker']} cerrada con éxito.")
                        st.rerun()
                    if col_y.button(f"🗑️ Eliminar Sin Registro", key=f"del_btn_{p['id']}", use_container_width=True):
                        tracker.portfolio['positions'] = [pos for pos in tracker.portfolio['positions'] if pos['id'] != p['id']]
                        tracker.portfolio['current_capital'] += p['position_size']
                        tracker._save_portfolio()
                        st.rerun()
        else:
            st.info("No tienes posiciones abiertas.")

    with tab_b:
        if tracker.portfolio['closed_trades']:
            df_hist = pd.DataFrame(tracker.portfolio['closed_trades']).sort_values('exit_date', ascending=False)
            st.dataframe(df_hist[['exit_date', 'ticker', 'entry_price', 'exit_price', 'pnl', 'pnl_pct']], use_container_width=True)
        else:
            st.info("El historial de trading está vacío.")

    with tab_c:
        c_ia1, c_ia2 = st.columns(2)
        c_ia1.metric("Promedio Ganador", f"${metrics['avg_win']:,.2f}", f"{metrics['avg_win_pct']:.2f}%")
        c_ia2.metric("Promedio Perdedor", f"${metrics['avg_loss']:,.2f}", f"{metrics['avg_loss_pct']:.2f}%")
