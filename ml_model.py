"""
MACHINE LEARNING MODULE - Random Forest Predictor
Sistema de predicción de probabilidad de subida/bajada usando indicadores técnicos
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from typing import Dict, Tuple, Optional
import pickle
import os
from datetime import datetime, timedelta


class TradingMLModel:
    """
    Modelo de Machine Learning para predecir movimientos de precio
    Usa Random Forest para clasificación binaria
    """
    
    def __init__(self, prediction_days: int = 5, threshold: float = 2.0):
        """
        Args:
            prediction_days: Días hacia adelante para predecir
            threshold: % mínimo de cambio para considerar "subida"
        """
        self.prediction_days = prediction_days
        self.threshold = threshold
        self.model = None
        self.feature_importance = None
        self.is_trained = False
        self.training_date = None
        self.model_metrics = {}
        
    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara datos para entrenamiento
        
        Args:
            data: DataFrame con indicadores técnicos ya calculados
        
        Returns:
            X (features), y (labels)
        """
        df = data.copy()
        
        # ====================================================================
        # CREAR ETIQUETAS (TARGET)
        # ====================================================================
        # Calcular retorno futuro
        df['Future_Return'] = df['Close'].pct_change(self.prediction_days).shift(-self.prediction_days) * 100
        
        # Etiquetar: 1 si sube más del threshold%, 0 si no
        df['Target'] = (df['Future_Return'] > self.threshold).astype(int)
        
        # ====================================================================
        # FEATURES (VARIABLES DE ENTRADA)
        # ====================================================================
        feature_columns = [
            # Momentum
            'RSI',
            'StochRSI',
            
            # Trend
            'MACD_Hist',
            'ADX',
            
            # Volatilidad
            'ATR',
            'BB_Width',
            
            # Volumen
            'RVOL',
            
            # Price action
            'Returns',  # Retorno 1 día
            
            # SMAs - posición relativa
            'SMA20',
            'SMA50',
        ]
        
        # Agregar features derivados
        df['Price_to_SMA20'] = (df['Close'] / df['SMA20'] - 1) * 100
        df['Price_to_SMA50'] = (df['Close'] / df['SMA50'] - 1) * 100
        df['SMA20_to_SMA50'] = (df['SMA20'] / df['SMA50'] - 1) * 100
        
        # Volatilidad relativa
        df['ATR_Normalized'] = df['ATR'] / df['Close']
        
        # Momentum de corto plazo
        df['Returns_5D'] = df['Close'].pct_change(5) * 100
        df['Returns_20D'] = df['Close'].pct_change(20) * 100
        
        # Actualizar features
        feature_columns.extend([
            'Price_to_SMA20',
            'Price_to_SMA50',
            'SMA20_to_SMA50',
            'ATR_Normalized',
            'Returns_5D',
            'Returns_20D'
        ])
        
        # Remover NaN (filas sin datos completos)
        df_clean = df[feature_columns + ['Target']].dropna()
        
        X = df_clean[feature_columns]
        y = df_clean['Target']
        
        return X, y
    
    def train(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Entrena el modelo con datos históricos
        
        Args:
            data: DataFrame con indicadores técnicos
            test_size: Proporción de datos para testing (default 20%)
        
        Returns:
            Dict con métricas de entrenamiento
        """
        print("📚 Preparando datos de entrenamiento...")
        X, y = self.prepare_training_data(data)
        
        if len(X) < 100:
            raise ValueError(f"Datos insuficientes. Mínimo 100 muestras, tienes {len(X)}")
        
        print(f"✅ Datos preparados: {len(X)} muestras")
        print(f"   - Clase 1 (subida): {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
        print(f"   - Clase 0 (no subida): {len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.1f}%)")
        
        # Dividir train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42,
            shuffle=False  # No shuffle para series temporales
        )
        
        print("\n🤖 Entrenando modelo Random Forest...")
        
        # Entrenar modelo
        self.model = RandomForestClassifier(
            n_estimators=100,      # 100 árboles
            max_depth=10,          # Profundidad máxima (evita overfitting)
            min_samples_split=20,  # Mínimo de muestras para dividir
            min_samples_leaf=10,   # Mínimo de muestras en hoja
            random_state=42,
            n_jobs=-1              # Usar todos los cores
        )
        
        self.model.fit(X_train, y_train)
        
        print("✅ Modelo entrenado!")
        
        # ====================================================================
        # EVALUAR MODELO
        # ====================================================================
        print("\n📊 Evaluando modelo...")
        
        # Predicciones en test
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Métricas
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc_roc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc_roc = 0.5
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Cross-validation
        print("\n🔄 Validación cruzada (5-fold)...")
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Guardar métricas
        self.model_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'features': list(X.columns)
        }
        
        self.is_trained = True
        self.training_date = datetime.now()
        
        # Imprimir resultados
        print("\n" + "="*60)
        print("📈 RESULTADOS DEL ENTRENAMIENTO")
        print("="*60)
        print(f"Accuracy (Precisión total):     {accuracy*100:.1f}%")
        print(f"Precision (Acierto en compras): {precision*100:.1f}%")
        print(f"Recall (Captura oportunidades): {recall*100:.1f}%")
        print(f"F1-Score (Balance):             {f1*100:.1f}%")
        print(f"AUC-ROC:                        {auc_roc:.3f}")
        print(f"\nCross-Validation (5-fold):      {cv_scores.mean()*100:.1f}% (±{cv_scores.std()*100:.1f}%)")
        print("="*60)
        
        print("\n🏆 Top 5 Features más importantes:")
        for idx, row in self.feature_importance.head(5).iterrows():
            print(f"   {row['feature']:20s} {row['importance']*100:5.1f}%")
        
        return self.model_metrics
    
    def predict(self, data: pd.DataFrame) -> Dict:
        """
        Predice probabilidad de subida para datos actuales
        
        Args:
            data: DataFrame con indicadores técnicos (última fila = datos actuales)
        
        Returns:
            Dict con predicción y confianza
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado. Ejecuta train() primero.")
        
        # Preparar features de la última fila
        df = data.copy()
        
        # Features derivados (mismos que en training)
        df['Price_to_SMA20'] = (df['Close'] / df['SMA20'] - 1) * 100
        df['Price_to_SMA50'] = (df['Close'] / df['SMA50'] - 1) * 100
        df['SMA20_to_SMA50'] = (df['SMA20'] / df['SMA50'] - 1) * 100
        df['ATR_Normalized'] = df['ATR'] / df['Close']
        df['Returns_5D'] = df['Close'].pct_change(5) * 100
        df['Returns_20D'] = df['Close'].pct_change(20) * 100
        
        # Seleccionar features
        feature_columns = self.model_metrics['features']
        X_current = df[feature_columns].iloc[[-1]]  # Última fila
        
        # Predecir
        proba = self.model.predict_proba(X_current)[0]
        pred_class = self.model.predict(X_current)[0]
        
        prob_down = proba[0]
        prob_up = proba[1]
        
        # Calcular confianza
        confidence = max(prob_up, prob_down)
        
        if confidence > 0.75:
            confidence_level = "MUY ALTA"
        elif confidence > 0.65:
            confidence_level = "ALTA"
        elif confidence > 0.55:
            confidence_level = "MEDIA"
        else:
            confidence_level = "BAJA"
        
        # Recomendación
        if prob_up > 0.65:
            recommendation = "COMPRA FUERTE"
        elif prob_up > 0.55:
            recommendation = "COMPRA"
        elif prob_up < 0.35:
            recommendation = "VENTA FUERTE"
        elif prob_up < 0.45:
            recommendation = "VENTA"
        else:
            recommendation = "MANTENER"
        
        return {
            'probability_up': prob_up,
            'probability_down': prob_down,
            'predicted_class': pred_class,
            'recommendation': recommendation,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'prediction_days': self.prediction_days,
            'threshold': self.threshold,
            'model_accuracy': self.model_metrics['accuracy']
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Retorna DataFrame con importancia de features"""
        if self.feature_importance is None:
            raise ValueError("Modelo no entrenado.")
        return self.feature_importance
    
    def save_model(self, filepath: str):
        """Guarda el modelo entrenado"""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado.")
        
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'model_metrics': self.model_metrics,
            'training_date': self.training_date,
            'prediction_days': self.prediction_days,
            'threshold': self.threshold
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Modelo guardado en: {filepath}")
    
    def load_model(self, filepath: str):
        """Carga un modelo previamente entrenado"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_importance = model_data['feature_importance']
        self.model_metrics = model_data['model_metrics']
        self.training_date = model_data['training_date']
        self.prediction_days = model_data['prediction_days']
        self.threshold = model_data['threshold']
        self.is_trained = True
        
        print(f"✅ Modelo cargado desde: {filepath}")
        print(f"   Entrenado el: {self.training_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Accuracy: {self.model_metrics['accuracy']*100:.1f}%")


# ============================================================================
# FUNCIONES HELPER PARA STREAMLIT
# ============================================================================

def train_ml_model_for_ticker(ticker: str, data_processed: pd.DataFrame, 
                              prediction_days: int = 5) -> TradingMLModel:
    """
    Entrena modelo ML para un ticker específico
    
    Args:
        ticker: Símbolo del activo
        data_processed: DataFrame con indicadores ya calculados
        prediction_days: Días hacia adelante para predecir
    
    Returns:
        Modelo entrenado
    """
    print(f"\n{'='*60}")
    print(f"🤖 ENTRENANDO MODELO ML PARA {ticker}")
    print(f"{'='*60}\n")
    
    model = TradingMLModel(prediction_days=prediction_days, threshold=2.0)
    
    try:
        metrics = model.train(data_processed, test_size=0.2)
        return model
    except Exception as e:
        print(f"❌ Error entrenando modelo: {str(e)}")
        return None


def get_ml_prediction(model: TradingMLModel, data_processed: pd.DataFrame) -> Dict:
    """
    Obtiene predicción ML para datos actuales
    
    Args:
        model: Modelo entrenado
        data_processed: DataFrame con indicadores
    
    Returns:
        Dict con predicción
    """
    if model is None or not model.is_trained:
        return None
    
    try:
        prediction = model.predict(data_processed)
        return prediction
    except Exception as e:
        print(f"❌ Error en predicción: {str(e)}")
        return None


def format_ml_output(prediction: Dict, ticker: str) -> str:
    """
    Formatea output del ML para mostrar en Streamlit
    
    Args:
        prediction: Dict con predicción
        ticker: Símbolo del activo
    
    Returns:
        String formateado en markdown
    """
    if prediction is None:
        return "⚠️ No hay predicción disponible"
    
    # Emoji según recomendación
    if "COMPRA" in prediction['recommendation']:
        emoji = "🟢"
    elif "VENTA" in prediction['recommendation']:
        emoji = "🔴"
    else:
        emoji = "🟡"
    
    output = f"""
## 🤖 Predicción Machine Learning - {ticker}

### Probabilidades
- **📈 Subida en {prediction['prediction_days']} días:** {prediction['probability_up']*100:.1f}%
- **📉 Bajada en {prediction['prediction_days']} días:** {prediction['probability_down']*100:.1f}%

### Recomendación
{emoji} **{prediction['recommendation']}**

### Confianza del Modelo
- **Nivel:** {prediction['confidence_level']}
- **Score:** {prediction['confidence']*100:.1f}%
- **Accuracy del modelo:** {prediction['model_accuracy']*100:.1f}%

### Interpretación
"""
    
    if prediction['probability_up'] > 0.65:
        output += "✅ El modelo tiene **alta confianza** en una subida. Las condiciones técnicas favorecen posiciones largas."
    elif prediction['probability_up'] > 0.55:
        output += "↗️ El modelo sugiere **leve sesgo alcista**. Considerar entrada con stops ajustados."
    elif prediction['probability_up'] < 0.35:
        output += "❌ El modelo tiene **alta confianza** en una bajada. Evitar posiciones largas."
    elif prediction['probability_up'] < 0.45:
        output += "↘️ El modelo sugiere **leve sesgo bajista**. Considerar reducir exposición."
    else:
        output += "↔️ El modelo es **neutral**. Esperar señal más clara antes de operar."
    
    if prediction['confidence_level'] == "BAJA":
        output += "\n\n⚠️ **Nota:** Confianza baja. Combinar con análisis técnico tradicional."
    
    return output
