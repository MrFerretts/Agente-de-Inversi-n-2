"""
CONSENSUS SCORING SYSTEM
Sistema que combina todos los análisis (Técnico, ML, LSTM, Groq) en una recomendación única ponderada
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


class ConsensusAnalyzer:
    """
    Combina múltiples fuentes de análisis en un score de consenso
    
    Fuentes:
    1. Análisis Técnico (Score -100 a +100)
    2. ML Prediction (Probabilidad 0-1)
    3. LSTM Prediction (Probabilidad 0-1, si disponible)
    4. Groq AI Analysis (Sentimiento del análisis)
    
    Output: Consensus Score 0-100 con nivel de confianza
    """
    
    def __init__(self, weights: Optional[Dict] = None):
        """
        Args:
            weights: Dict con pesos para cada fuente
                     Si None, usa pesos default balanceados
        """
        # Pesos default (suman 100)
        self.weights = weights or {
            'technical': 30,  # 30% peso al análisis técnico
            'ml': 25,         # 25% peso al ML
            'lstm': 25,       # 25% peso al LSTM
            'groq': 20        # 20% peso al análisis de Groq
        }
        
        # Validar que sumen 100
        total = sum(self.weights.values())
        if total != 100:
            # Normalizar
            for key in self.weights:
                self.weights[key] = (self.weights[key] / total) * 100
    
    def analyze_consensus(self, 
                         technical_score: float,
                         ml_prediction: Optional[Dict] = None,
                         lstm_prediction: Optional[Dict] = None,
                         groq_analysis: Optional[str] = None) -> Dict:
        """
        Genera consensus score combinando todas las fuentes
        
        Args:
            technical_score: Score técnico (-100 a +100)
            ml_prediction: Dict con predicción ML
            lstm_prediction: Dict con predicción LSTM
            groq_analysis: String con análisis de Groq
        
        Returns:
            Dict con consensus score, confianza, y breakdown
        """
        scores = {}
        available_sources = []
        
        # ====================================================================
        # 1. ANÁLISIS TÉCNICO
        # ====================================================================
        # Normalizar de -100/+100 a 0-100
        technical_normalized = ((technical_score + 100) / 200) * 100
        scores['technical'] = technical_normalized
        available_sources.append('technical')
        
        # ====================================================================
        # 2. ML PREDICTION
        # ====================================================================
        if ml_prediction and ml_prediction.get('probability_up'):
            # Convertir probabilidad (0-1) a score (0-100)
            ml_score = ml_prediction['probability_up'] * 100
            scores['ml'] = ml_score
            available_sources.append('ml')
        
        # ====================================================================
        # 3. LSTM PREDICTION
        # ====================================================================
        if lstm_prediction and lstm_prediction.get('probability_up'):
            lstm_score = lstm_prediction['probability_up'] * 100
            scores['lstm'] = lstm_score
            available_sources.append('lstm')
        
        # ====================================================================
        # 4. GROQ AI ANALYSIS
        # ====================================================================
        if groq_analysis:
            groq_score = self._extract_groq_sentiment(groq_analysis)
            scores['groq'] = groq_score
            available_sources.append('groq')
        
        # ====================================================================
        # CALCULAR CONSENSUS SCORE
        # ====================================================================
        total_weight = sum([self.weights[src] for src in available_sources])
        
        consensus_score = 0
        for source in available_sources:
            # Peso proporcional basado en fuentes disponibles
            weight = (self.weights[source] / total_weight) * 100
            consensus_score += scores[source] * (weight / 100)
        
        # ====================================================================
        # CALCULAR CONFIANZA
        # ====================================================================
        # Confianza basada en:
        # 1. Número de fuentes disponibles
        # 2. Acuerdo entre fuentes
        
        confidence = self._calculate_confidence(scores, available_sources)
        
        # ====================================================================
        # DETERMINAR RECOMENDACIÓN
        # ====================================================================
        recommendation = self._get_recommendation(consensus_score, confidence)
        
        # ====================================================================
        # ANÁLISIS DE DISCREPANCIAS
        # ====================================================================
        discrepancies = self._analyze_discrepancies(scores)
        
        return {
            'consensus_score': round(consensus_score, 1),
            'confidence': round(confidence, 1),
            'recommendation': recommendation,
            'sources_used': available_sources,
            'source_scores': scores,
            'discrepancies': discrepancies,
            'weights_used': {k: self.weights[k] for k in available_sources}
        }
    
    def _extract_groq_sentiment(self, groq_text: str) -> float:
        """
        Extrae sentimiento del análisis de Groq
        
        Args:
            groq_text: Texto del análisis
        
        Returns:
            Score 0-100
        """
        # Keywords alcistas
        bullish_keywords = [
            'COMPRA FUERTE', 'COMPRA AGRESIVA', 'alta confianza en una subida',
            'condiciones técnicas favorecen', 'momentum alcista', 'señal muy confiable',
            'alta probabilidad', 'convergencia alcista', 'sesgo alcista fuerte'
        ]
        
        # Keywords bajistas
        bearish_keywords = [
            'VENTA FUERTE', 'VENTA', 'alta confianza en una bajada', 
            'evitar posiciones largas', 'momentum bajista', 'tendencia bajista',
            'sesgo bajista', 'reducir exposición'
        ]
        
        # Keywords neutrales
        neutral_keywords = [
            'MANTENER', 'ESPERAR', 'neutral', 'sin operación',
            'señal más clara', 'precaución'
        ]
        
        text_upper = groq_text.upper()
        
        # Contar coincidencias
        bullish_count = sum(1 for kw in bullish_keywords if kw.upper() in text_upper)
        bearish_count = sum(1 for kw in bearish_keywords if kw.upper() in text_upper)
        neutral_count = sum(1 for kw in neutral_keywords if kw.upper() in text_upper)
        
        # Calcular score
        if bullish_count > bearish_count and bullish_count > neutral_count:
            # Alcista fuerte
            return 75 + (bullish_count * 5)  # 75-100
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            # Bajista fuerte
            return 25 - (bearish_count * 5)  # 0-25
        elif bullish_count > bearish_count:
            # Alcista moderado
            return 60
        elif bearish_count > bullish_count:
            # Bajista moderado
            return 40
        else:
            # Neutral
            return 50
    
    def _calculate_confidence(self, scores: Dict, sources: list) -> float:
        """
        Calcula nivel de confianza basado en acuerdo entre fuentes
        
        Args:
            scores: Dict con scores de cada fuente
            sources: Lista de fuentes disponibles
        
        Returns:
            Confianza 0-100
        """
        if len(sources) < 2:
            # Con una sola fuente, confianza media
            return 60
        
        # Base de confianza según número de fuentes
        base_confidence = {
            2: 60,  # 2 fuentes
            3: 70,  # 3 fuentes
            4: 80   # 4 fuentes (todas)
        }
        confidence = base_confidence.get(len(sources), 60)
        
        # Calcular variación entre fuentes (desviación estándar)
        score_values = list(scores.values())
        std_dev = np.std(score_values)
        
        # Ajustar confianza basado en acuerdo
        # Menor desviación = mayor confianza
        if std_dev < 10:
            # Excelente acuerdo
            confidence += 15
        elif std_dev < 20:
            # Buen acuerdo
            confidence += 10
        elif std_dev < 30:
            # Acuerdo moderado
            confidence += 0
        else:
            # Poco acuerdo
            confidence -= 15
        
        # Limitar a 0-100
        return max(0, min(100, confidence))
    
    def _get_recommendation(self, score: float, confidence: float) -> str:
        """
        Determina recomendación basada en score y confianza
        
        Args:
            score: Consensus score 0-100
            confidence: Nivel de confianza 0-100
        
        Returns:
            String con recomendación
        """
        # Ajustar umbrales según confianza
        if confidence >= 80:
            # Alta confianza - umbrales más agresivos
            if score >= 70:
                return "COMPRA FUERTE"
            elif score >= 60:
                return "COMPRA"
            elif score <= 30:
                return "VENTA FUERTE"
            elif score <= 40:
                return "VENTA"
            else:
                return "MANTENER"
        elif confidence >= 60:
            # Confianza media - umbrales estándar
            if score >= 75:
                return "COMPRA FUERTE"
            elif score >= 65:
                return "COMPRA"
            elif score <= 25:
                return "VENTA FUERTE"
            elif score <= 35:
                return "VENTA"
            else:
                return "MANTENER"
        else:
            # Baja confianza - más conservador
            if score >= 80:
                return "COMPRA (Baja Confianza)"
            elif score <= 20:
                return "VENTA (Baja Confianza)"
            else:
                return "ESPERAR - Señales Mixtas"
    
    def _analyze_discrepancies(self, scores: Dict) -> list:
        """
        Analiza discrepancias importantes entre fuentes
        
        Args:
            scores: Dict con scores de cada fuente
        
        Returns:
            Lista de mensajes sobre discrepancias
        """
        discrepancies = []
        
        if len(scores) < 2:
            return discrepancies
        
        score_items = list(scores.items())
        
        # Comparar cada par de fuentes
        for i in range(len(score_items)):
            for j in range(i + 1, len(score_items)):
                source1, score1 = score_items[i]
                source2, score2 = score_items[j]
                
                diff = abs(score1 - score2)
                
                # Discrepancia significativa (>30 puntos)
                if diff > 30:
                    direction1 = "alcista" if score1 > 50 else "bajista" if score1 < 50 else "neutral"
                    direction2 = "alcista" if score2 > 50 else "bajista" if score2 < 50 else "neutral"
                    
                    discrepancies.append(
                        f"⚠️ {source1.upper()} ({direction1}: {score1:.0f}) vs "
                        f"{source2.upper()} ({direction2}: {score2:.0f}) - "
                        f"Diferencia de {diff:.0f} puntos"
                    )
        
        return discrepancies
    
    def format_consensus_output(self, consensus: Dict, ticker: str) -> str:
        """
        Formatea el output del consensus para mostrar en Streamlit
        
        Args:
            consensus: Dict retornado por analyze_consensus()
            ticker: Símbolo del activo
        
        Returns:
            String formateado en markdown
        """
        score = consensus['consensus_score']
        confidence = consensus['confidence']
        recommendation = consensus['recommendation']
        
        # Color según recomendación
        if "COMPRA" in recommendation:
            color = "🟢"
        elif "VENTA" in recommendation:
            color = "🔴"
        else:
            color = "🟡"
        
        output = f"""
## 🎯 Consensus Score - {ticker}

### Recomendación Final
# {color} **{recommendation}**

### Métricas
- **Consensus Score:** {score:.1f}/100
- **Nivel de Confianza:** {confidence:.1f}%
- **Fuentes Utilizadas:** {len(consensus['sources_used'])}/4

### Breakdown por Fuente
"""
        
        # Mostrar cada fuente
        for source, source_score in consensus['source_scores'].items():
            weight = consensus['weights_used'][source]
            emoji = self._get_source_emoji(source)
            
            output += f"- {emoji} **{source.upper()}:** {source_score:.1f}/100 (Peso: {weight:.0f}%)\n"
        
        # Discrepancias
        if consensus['discrepancies']:
            output += "\n### ⚠️ Señales Conflictivas\n"
            for disc in consensus['discrepancies']:
                output += f"{disc}\n"
        
        # Interpretación
        output += "\n### 💡 Interpretación\n"
        
        if confidence >= 80:
            output += f"✅ **Alta confianza** - Las fuentes están muy alineadas. "
        elif confidence >= 60:
            output += f"ℹ️ **Confianza moderada** - Hay buen acuerdo entre fuentes. "
        else:
            output += f"⚠️ **Baja confianza** - Señales mixtas entre fuentes. "
        
        if score >= 70:
            output += "El consenso apunta firmemente hacia una oportunidad de compra."
        elif score >= 60:
            output += "El consenso sugiere una ligera inclinación alcista."
        elif score <= 30:
            output += "El consenso apunta hacia una oportunidad de venta."
        elif score <= 40:
            output += "El consenso sugiere precaución con sesgo bajista."
        else:
            output += "El consenso sugiere esperar por señales más claras."
        
        return output
    
    def _get_source_emoji(self, source: str) -> str:
        """Retorna emoji según la fuente"""
        emojis = {
            'technical': '📊',
            'ml': '🤖',
            'lstm': '🧠',
            'groq': '💬'
        }
        return emojis.get(source, '📈')


# ============================================================================
# FUNCIÓN HELPER PARA STREAMLIT
# ============================================================================

def get_consensus_analysis(ticker: str,
                          technical_analysis: Dict,
                          ml_prediction: Optional[Dict] = None,
                          lstm_prediction: Optional[Dict] = None,
                          groq_analysis: Optional[str] = None,
                          custom_weights: Optional[Dict] = None) -> Dict:
    """
    Genera análisis de consenso completo
    
    Args:
        ticker: Símbolo
        technical_analysis: Dict del análisis técnico
        ml_prediction: Dict de predicción ML
        lstm_prediction: Dict de predicción LSTM
        groq_analysis: String con análisis Groq
        custom_weights: Dict con pesos personalizados
    
    Returns:
        Dict con consensus completo
    """
    analyzer = ConsensusAnalyzer(weights=custom_weights)
    
    # Extraer score técnico
    technical_score = technical_analysis['signals']['score']
    
    # Generar consensus
    consensus = analyzer.analyze_consensus(
        technical_score=technical_score,
        ml_prediction=ml_prediction,
        lstm_prediction=lstm_prediction,
        groq_analysis=groq_analysis
    )
    
    # Agregar ticker
    consensus['ticker'] = ticker
    
    return consensus
