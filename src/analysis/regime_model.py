"""Regime Detection (Layer 1 & 2) for Advanced Trading System.

Handles feature engineering and probabilistic regime classification using GMM.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """Detects and classifies hidden market regimes using statistical modeling."""

    def __init__(self, n_regimes: int = 4, lookback: int = 252*2):
        """Initialize the regime detector.
        
        Args:
            n_regimes: Number of regimes to detect (e.g., 3 or 4)
            lookback: Lookback period for training the model (in days)
        """
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.model = GaussianMixture(
            n_components=n_regimes,
            covariance_type='full',
            random_state=42,
            n_init=10
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.regime_map = {} # Maps GMM labels to human names

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """COMPUTE LAYER 1: DATA + FEATURE ENGINE."""
        if len(df) < 50:
            return pd.DataFrame()

        # 1. Returns
        features = pd.DataFrame(index=df.index)
        features['ret_1d'] = df['Close'].pct_change(1)
        features['ret_5d'] = df['Close'].pct_change(5)
        features['ret_20d'] = df['Close'].pct_change(20)

        # 2. Rolling Volatility
        features['vol_20d'] = features['ret_1d'].rolling(20).std()
        
        # ATR (normalized by price)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr_ratio'] = tr.rolling(14).mean() / df['Close']

        # 3. Volume Anomaly
        features['vol_anomaly'] = df['Volume'] / df['Volume'].rolling(20).mean()

        # 4. Trend Strength (Slope)
        log_price = np.log(df['Close'])
        features['slope_20d'] = log_price.diff(20) / 20 # Simple proxy for slope
        
        # 5. Volatility of Volatility
        features['vol_of_vol'] = features['vol_20d'].rolling(20).std()

        return features.dropna()

    def train(self, benchmarks: List[pd.DataFrame]):
        """Train the GMM on benchmark data (e.g., Nifty 50, Bank Nifty)."""
        logger.info(f"Training regime model with {len(benchmarks)} indices...")
        
        all_features = []
        for df in benchmarks:
            feat = self.extract_features(df)
            # Take last self.lookback days
            all_features.append(feat.tail(self.lookback))
            
        if not all_features:
            logger.error("No valid benchmark features for training")
            return
            
        X = pd.concat(all_features).values
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled)
        self.is_trained = True
        
        # Auto-label regimes based on volatility and returns
        labels = self.model.predict(X_scaled)
        self._assign_labels(X, labels)
        
        logger.info("Regime model training complete.")

    def _assign_labels(self, X: np.ndarray, labels: np.ndarray):
        """Interprets GMM clusters and maps them to descriptive names."""
        unique_labels = np.unique(labels)
        label_stats = []
        
        for label in unique_labels:
            mask = (labels == label)
            cluster_data = X[mask]
            # [ret_1d, ret_5d, ret_20d, vol_20d, atr_ratio, vol_anomaly, slope_20d, vol_of_vol]
            avg_vol = cluster_data[:, 3].mean()
            avg_sh_ret = cluster_data[:, 0].mean()
            avg_slope = cluster_data[:, 6].mean()
            
            label_stats.append({
                'label': label,
                'vol': avg_vol,
                'ret': avg_sh_ret,
                'slope': avg_slope
            })
            
        # Sort by volatility (Crises usually have highest vol)
        sorted_by_vol = sorted(label_stats, key=lambda x: x['vol'])
        
        # Simple heuristic mapping for 4 regimes
        # 0: Low Vol, Positive Trend -> Bull Trending
        # 1: Mid Vol, Low Trend -> Sideways / Mean Reverting
        # 2: High Vol, Negative Trend -> Bear / Distressed
        # 3: Extreme Vol -> Crisis / Panic
        
        # Sort lowest vol to highest
        self.regime_map[sorted_by_vol[0]['label']] = "Bullish Trending"
        self.regime_map[sorted_by_vol[1]['label']] = "Sideways / Mean Reverting"
        self.regime_map[sorted_by_vol[2]['label']] = "Volatile / Defensive"
        self.regime_map[sorted_by_vol[3]['label']] = "Crisis / High Vol"

    def predict(self, df: pd.DataFrame) -> Dict:
        """COMPUTE LAYER 2 OUTPUT: Regime probabilities."""
        if not self.is_trained:
            return {'error': 'Model not trained'}
            
        feat = self.extract_features(df)
        if feat.empty:
            return {'error': 'Insufficient data for features'}
            
        last_X = feat.iloc[-1].values.reshape(1, -1)
        X_scaled = self.scaler.transform(last_X)
        
        probs = self.model.predict_proba(X_scaled)[0]
        pred_label = self.model.predict(X_scaled)[0]
        
        regime_name = self.regime_map.get(pred_label, "Unknown")
        
        # Build probability distribution
        prob_dist = {self.regime_map[i]: float(probs[i]) for i in range(len(probs))}
        
        return {
            'regime': regime_name,
            'probabilities': prob_dist,
            'timestamp': feat.index[-1].strftime('%Y-%m-%d'),
            'features': feat.iloc[-1].to_dict()
        }

    def save_state(self, path: str):
        """Save trained model state."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'regime_map': self.regime_map,
                'is_trained': self.is_trained
            }, f)

    @classmethod
    def load_state(cls, path: str):
        """Load trained model state."""
        import pickle
        detector = cls()
        with open(path, 'rb') as f:
            state = pickle.load(f)
            detector.model = state['model']
            detector.scaler = state['scaler']
            detector.regime_map = state['regime_map']
            detector.is_trained = state['is_trained']
        return detector
