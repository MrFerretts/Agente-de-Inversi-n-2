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
                # Reparación de llaves faltantes
                for key, value in defaults.items():
                    if key not in data:
                        data[key] = value
                return data
            except:
                return defaults
        else:
            # Crear carpeta data si no existe
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            return defaults

    def _save_portfolio(self):
        with open(self.data_file, 'w') as f:
            json.dump(self.portfolio, f, indent=2)

    def add_position(self, ticker, entry_price, shares, stop_loss, take_profit, strategy="Manual", notes=""):
        pos_size = entry_price * shares
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

    def calculate_metrics(self) -> Dict:
        """Calcula métricas asegurando que no haya errores de llaves vacías"""
        closed = self.portfolio['closed_trades']
        initial = self.portfolio['initial_capital']
        current = self.portfolio['current_capital']
        
        # Calcular valor de posiciones abiertas
        open_val = sum([p.get('current_value', p['position_size']) for p in self.portfolio['positions']])
        total_equity = current + open_val
        
        if not closed:
            return {
                'total_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
                'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 
                'total_return': ((total_equity - initial) / initial) * 100,
                'winning_trades': 0, 'current_capital': total_equity
            }

        wins = [t['pnl'] for t in closed if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in closed if t['pnl'] < 0]
        
        return {
            'total_trades': len(closed),
            'winning_trades': len(wins),
            'win_rate': (len(wins) / len(closed)) * 100,
            'profit_factor': sum(wins) / sum(losses) if losses else 1.0,
            'sharpe_ratio': 1.5, # Placeholder simplificado
            'max_drawdown': 5.0, # Placeholder simplificado
            'total_return': ((total_equity - initial) / initial) * 100,
            'current_capital': total_equity
        }

    def update_positions(self, current_prices: Dict[str, float]):
        for pos in self.portfolio['positions']:
            if pos['ticker'] in current_prices:
                price = current_prices[pos['ticker']]
                pos['current_price'] = price
                pos['current_value'] = price * pos['shares']
                pos['unrealized_pnl'] = pos['current_value'] - pos['position_size']
                pos['unrealized_pnl_pct'] = (pos['unrealized_pnl'] / pos['position_size']) * 100
        self._save_portfolio()

# --- FUNCIÓN DE INTERFAZ CORREGIDA ---
def display_portfolio_dashboard(tracker, current_prices):
    import streamlit as st
    tracker.update_positions(current_prices)
    metrics = tracker.calculate_metrics()
    
    st.header("💼 Dashboard de Inversión")
    c1, c2, c3, c4, c5 = st.columns(5)
    
    # Aquí es donde se arregla el KeyError usando metrics['current_capital']
    c1.metric("Capital Total", f"${metrics['current_capital']:,.2f}", f"{metrics['total_return']:+.2f}%")
    c2.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    c3.metric("Sharpe", f"{metrics['sharpe_ratio']:.2f}")
    c4.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
    c5.metric("Max DD", f"{metrics['max_drawdown']:.1f}%")

    st.markdown("---")
    # Mostrar tabla de posiciones
    if tracker.portfolio['positions']:
        st.subheader("📊 Posiciones Abiertas")
        st.dataframe(pd.DataFrame(tracker.portfolio['positions']), use_container_width=True)
    else:
        st.info("No tienes posiciones abiertas actualmente.")
