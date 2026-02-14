"""
Enhanced Visualization Module - Gráficos profesionales para terminal quant
Incluye heatmaps, comparativas y señales históricas
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class ChartBuilder:
    """Constructor de gráficos profesionales"""
    
    @staticmethod
    def create_multi_indicator_chart(df: pd.DataFrame, 
                                    symbol: str,
                                    show_signals: bool = True) -> go.Figure:
        """
        Crea gráfico principal con 4 paneles:
        1. Precio + Bandas + SMAs
        2. RSI + Stoch RSI
        3. MACD
        4. Volumen + RVOL
        """
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.45, 0.2, 0.2, 0.15],
            subplot_titles=(
                f"{symbol} - Precio & Tendencia",
                "Momentum (RSI)",
                "MACD",
                "Volumen"
            )
        )
        
        # ========== PANEL 1: PRECIO ==========
        # Velas
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Precio",
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # SMAs
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA20'],
                name="SMA20",
                line=dict(color='orange', width=1.5)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA50'],
                name="SMA50",
                line=dict(color='blue', width=1.5)
            ),
            row=1, col=1
        )
        
        # Bandas de Bollinger
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Upper'],
                name="BB Superior",
                line=dict(color='rgba(173,216,230,0.3)', width=1),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Lower'],
                name="BB Inferior",
                line=dict(color='rgba(173,216,230,0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(173,216,230,0.1)',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Señales de compra/venta (si aplica)
        if show_signals and 'Signal' in df.columns:
            buy_signals = df[df['Signal'] == 1]
            sell_signals = df[df['Signal'] == -1]
            
            if not buy_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['Low'] * 0.99,
                        mode='markers',
                        name='Compra',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color='lime',
                            line=dict(color='darkgreen', width=1)
                        )
                    ),
                    row=1, col=1
                )
            
            if not sell_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['High'] * 1.01,
                        mode='markers',
                        name='Venta',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color='red',
                            line=dict(color='darkred', width=1)
                        )
                    ),
                    row=1, col=1
                )
        
        # ========== PANEL 2: RSI ==========
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                name="RSI",
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        
        # Líneas de referencia RSI
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1, opacity=0.5)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1, opacity=0.5)
        fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1, opacity=0.3)
        
        # Stochastic RSI (área sombreada)
        if 'StochRSI' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['StochRSI'] * 100,  # Escalar a 0-100
                    name="Stoch RSI",
                    line=dict(color='orange', width=1, dash='dot'),
                    opacity=0.5
                ),
                row=2, col=1
            )
        
        # ========== PANEL 3: MACD ==========
        # Histograma MACD
        colors = ['green' if val > 0 else 'red' for val in df['MACD_Hist']]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['MACD_Hist'],
                name="MACD Hist",
                marker_color=colors,
                opacity=0.6
            ),
            row=3, col=1
        )
        
        # Líneas MACD
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_Line'],
                name="MACD",
                line=dict(color='blue', width=1.5)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_Signal'],
                name="Signal",
                line=dict(color='orange', width=1.5)
            ),
            row=3, col=1
        )
        
        # ========== PANEL 4: VOLUMEN ==========
        # Volumen como barras
        vol_colors = ['green' if df['Close'].iloc[i] > df['Open'].iloc[i] else 'red' 
                     for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name="Volumen",
                marker_color=vol_colors,
                opacity=0.5
            ),
            row=4, col=1
        )
        
        # RVOL como línea
        if 'RVOL' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RVOL'],
                    name="RVOL",
                    line=dict(color='cyan', width=2),
                    yaxis='y2'
                ),
                row=4, col=1
            )
        
        # ========== LAYOUT ==========
        fig.update_layout(
            height=1000,
            template="plotly_dark",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified',
            xaxis_rangeslider_visible=False
        )
        
        # Configurar ejes
        fig.update_yaxes(title_text="Precio ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_yaxes(title_text="Volumen", row=4, col=1)
        fig.update_xaxes(title_text="Fecha", row=4, col=1)
        
        return fig
    
    @staticmethod
    def create_correlation_heatmap(symbols: List[str], 
                                   returns_dict: Dict[str, pd.Series]) -> go.Figure:
        """
        Crea heatmap de correlación entre activos
        """
        # Construir DataFrame de retornos
        returns_df = pd.DataFrame(returns_dict)
        
        # Calcular matriz de correlación
        corr_matrix = returns_df.corr()
        
        # Crear heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdYlGn',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlación")
            )
        )
        
        fig.update_layout(
            title="Matriz de Correlación (Retornos Diarios)",
            height=600,
            template="plotly_dark"
        )
        
        return fig
    
    @staticmethod
    def create_performance_comparison(analyses: Dict[str, Dict]) -> go.Figure:
        """
        Crea gráfico de barras comparando performance de múltiples activos
        """
        symbols = []
        scores = []
        colors = []
        
        for symbol, analysis in analyses.items():
            if not analysis or 'signals' not in analysis:
                continue
            
            symbols.append(symbol)
            score = analysis['signals']['score']
            scores.append(score)
            
            # Color basado en score
            if score >= 60:
                colors.append('#27ae60')  # Verde fuerte
            elif score >= 30:
                colors.append('#2ecc71')  # Verde claro
            elif score <= -60:
                colors.append('#e74c3c')  # Rojo fuerte
            elif score <= -30:
                colors.append('#ec7063')  # Rojo claro
            else:
                colors.append('#95a5a6')  # Gris
        
        fig = go.Figure(
            data=[
                go.Bar(
                    x=symbols,
                    y=scores,
                    marker_color=colors,
                    text=scores,
                    textposition='outside',
                    texttemplate='%{text:.0f}'
                )
            ]
        )
        
        fig.update_layout(
            title="Comparativa de Señales (Score Total)",
            xaxis_title="Activo",
            yaxis_title="Score",
            height=500,
            template="plotly_dark",
            showlegend=False
        )
        
        # Líneas de referencia
        fig.add_hline(y=60, line_dash="dash", line_color="green", 
                     annotation_text="Compra Fuerte", annotation_position="right")
        fig.add_hline(y=30, line_dash="dot", line_color="lightgreen",
                     annotation_text="Compra", annotation_position="right")
        fig.add_hline(y=-30, line_dash="dot", line_color="salmon",
                     annotation_text="Venta", annotation_position="right")
        fig.add_hline(y=-60, line_dash="dash", line_color="red",
                     annotation_text="Venta Fuerte", annotation_position="right")
        
        return fig
    
    @staticmethod
    def create_risk_metrics_gauge(risk_metrics: Dict) -> go.Figure:
        """
        Crea dashboard de medidores para métricas de riesgo
        """
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=("Riesgo del Portfolio", "Exposición Total", "Posiciones Abiertas")
        )
        
        # Gauge 1: Riesgo
        risk_pct = risk_metrics.get('total_risk_pct', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=risk_pct,
                title={'text': "Riesgo (%)"},
                delta={'reference': 5},
                gauge={
                    'axis': {'range': [None, 15]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 3], 'color': "lightgreen"},
                        {'range': [3, 6], 'color': "yellow"},
                        {'range': [6, 10], 'color': "orange"},
                        {'range': [10, 15], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 10
                    }
                }
            ),
            row=1, col=1
        )
        
        # Gauge 2: Exposición
        exposure_pct = risk_metrics.get('total_exposure_pct', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=exposure_pct,
                title={'text': "Exposición (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "blue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightblue"},
                        {'range': [50, 80], 'color': "cornflowerblue"},
                        {'range': [80, 100], 'color': "darkblue"}
                    ]
                }
            ),
            row=1, col=2
        )
        
        # Gauge 3: Número de posiciones
        num_positions = risk_metrics.get('num_positions', 0)
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=num_positions,
                title={'text': "Posiciones"},
                delta={'reference': 5},
                number={'font': {'size': 60}}
            ),
            row=1, col=3
        )
        
        fig.update_layout(
            height=300,
            template="plotly_dark"
        )
        
        return fig
