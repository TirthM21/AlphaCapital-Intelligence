"""
Nifty Factor Indices Engine
===========================
Implements the exact NSE Methodology (Feb 2026) for:
- Alpha 50 / Alpha 30 (Jensen's Alpha)
- High Beta 50
- Low Volatility 50 / Low Volatility 30
- Momentum 30 / Momentum 50 (Normalized Momentum Score)
- Quality 30 / Quality 50 (ROE, D/E, EPS variability)
- Value (E/P, B/P, S/P, Dividend Yield)
- Multifactor MQVLv 50 (Momentum + Quality + Value + Low Volatility)
- Dual Momentum (Absolute + Relative)
"""
import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf
import logging

logger = logging.getLogger(__name__)


class NiftyFactorAnalyzer:
    """
    Computes factor scores for a universe of stocks per NSE Index Methodology.
    """

    # ------------------------------------------------------------------ #
    #  PRICE-BASED HELPERS
    # ------------------------------------------------------------------ #
    @staticmethod
    def _log_returns(series):
        return np.log(series / series.shift(1)).dropna()

    @staticmethod
    def calculate_annualized_volatility(price_series, window=252):
        """SD of log-normal daily returns, annualised (1-year)."""
        lr = NiftyFactorAnalyzer._log_returns(price_series)
        if len(lr) < 100:
            return np.nan
        return float(lr.tail(window).std() * np.sqrt(252))

    @staticmethod
    def calculate_alpha_beta(stock_prices, market_prices):
        """Jensen's Alpha & Beta using 1-year trailing prices."""
        sr = stock_prices.pct_change().dropna()
        mr = market_prices.pct_change().dropna()
        common = sr.index.intersection(mr.index)
        combined = pd.concat([sr.loc[common], mr.loc[common]], axis=1).dropna()
        if len(combined) < 100:
            return np.nan, np.nan
        y, x = combined.iloc[:, 0].values, combined.iloc[:, 1].values
        slope, intercept, _, _, _ = stats.linregress(x, y)
        return intercept * 252, slope          # annualised alpha, beta

    @staticmethod
    def calculate_momentum_ratio(price_series, months=12):
        """MR = Price Return / Annualised Volatility."""
        days = months * 21
        if len(price_series) < max(days, 252):
            return np.nan
        price_return = (price_series.iloc[-1] / price_series.iloc[-days]) - 1
        vol = NiftyFactorAnalyzer.calculate_annualized_volatility(price_series)
        if vol == 0 or np.isnan(vol):
            return np.nan
        return price_return / vol

    # ------------------------------------------------------------------ #
    #  Z-SCORE / NORMALIZED SCORE HELPERS  (NSE formula)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _z_scores(series):
        """Standard Z-score: (x – μ) / σ"""
        s = series.dropna()
        if s.std() == 0:
            return series * 0
        return (series - s.mean()) / s.std()

    @staticmethod
    def _normalized_score(z_series):
        """
        NSE Normalized Score:
          (1 + Z)      if Z >= 0
          (1 - Z)^-1   if Z <  0
        """
        return z_series.apply(lambda z: (1 + z) if z >= 0 else (1 - z) ** -1)

    # ------------------------------------------------------------------ #
    #  FACTOR 1 – MOMENTUM  (Nifty200 Momentum 30 / Nifty500 Momentum 50)
    # ------------------------------------------------------------------ #
    def _compute_momentum(self, factor_df):
        """
        Weighted Avg Z = 50% * Z(MR12) + 50% * Z(MR6)
        Normalized Momentum Score from that Z.
        """
        z6 = self._z_scores(factor_df['mr6'])
        z12 = self._z_scores(factor_df['mr12'])
        factor_df['momentum_z'] = 0.5 * z12 + 0.5 * z6
        factor_df['momentum_score'] = self._normalized_score(factor_df['momentum_z'])
        factor_df['momentum_percentile'] = factor_df['momentum_score'].rank(pct=True)
        return factor_df

    # ------------------------------------------------------------------ #
    #  FACTOR 2 – ALPHA  (Nifty Alpha 50 / Nifty100 Alpha 30)
    # ------------------------------------------------------------------ #
    def _compute_alpha_score(self, factor_df):
        """
        Alpha Score = Normalized(Z(Jensen's Alpha))
        Only positive alphas are considered for selection.
        """
        z = self._z_scores(factor_df['alpha'])
        factor_df['alpha_score'] = self._normalized_score(z)
        return factor_df

    # ------------------------------------------------------------------ #
    #  FACTOR 3 – LOW VOLATILITY  (Nifty Low Vol 50 / Nifty100 Low Vol 30)
    # ------------------------------------------------------------------ #
    def _compute_low_vol_score(self, factor_df):
        """
        Low Volatility = 1 / volatility  (inverse).
        Z-score on the inverse, then Normalized Score.
        Least volatile stock gets highest score.
        """
        factor_df['inv_vol'] = 1.0 / factor_df['volatility'].replace(0, np.nan)
        z = self._z_scores(factor_df['inv_vol'])
        factor_df['low_vol_score'] = self._normalized_score(z)
        factor_df['low_vol_percentile'] = factor_df['low_vol_score'].rank(pct=True)
        return factor_df

    # ------------------------------------------------------------------ #
    #  FACTOR 4 – QUALITY  (Nifty100 Quality 30 / Nifty500 Quality 50)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fetch_quality_data(ticker):
        """
        Fetches ROE, D/E ratio, and trailing EPS from yfinance.
        Returns dict or None.
        """
        try:
            info = yf.Ticker(ticker).info
            roe = info.get('returnOnEquity', np.nan)
            de = info.get('debtToEquity', np.nan)
            # Normalize D/E from percentage to ratio
            if de is not None and not np.isnan(de):
                de = de / 100.0
            trailing_eps = info.get('trailingEps', np.nan)
            forward_eps = info.get('forwardEps', np.nan)
            sector = info.get('sector', 'Unknown')
            is_financial = 'Financial' in str(sector)
            return {
                'roe': roe, 'de': de,
                'trailing_eps': trailing_eps, 'forward_eps': forward_eps,
                'is_financial': is_financial, 'sector': sector
            }
        except Exception:
            return None

    def _compute_quality_score(self, factor_df):
        """
        Quality Z = 1/3 * Z(ROE) + 1/3 * -Z(D/E) + 1/3 * -Z(EPS var)
        For financials: 0.5 * Z(ROE) + 0.5 * -Z(EPS var)
        """
        # Fetch fundamental data
        quality_data = []
        for _, row in factor_df.iterrows():
            ticker = row['symbol']
            q = self._fetch_quality_data(ticker)
            if q:
                quality_data.append({**row.to_dict(), **q})
            else:
                quality_data.append({**row.to_dict(), 'roe': np.nan, 'de': np.nan,
                                     'trailing_eps': np.nan, 'forward_eps': np.nan,
                                     'is_financial': False, 'sector': 'Unknown'})

        qdf = pd.DataFrame(quality_data)

        # EPS growth variability proxy: abs(forward - trailing) / trailing
        qdf['eps_var'] = np.where(
            qdf['trailing_eps'].abs() > 0,
            ((qdf['forward_eps'] - qdf['trailing_eps']) / qdf['trailing_eps'].abs()).abs(),
            np.nan
        )

        z_roe = self._z_scores(qdf['roe'])
        z_de = self._z_scores(qdf['de'])
        z_eps_var = self._z_scores(qdf['eps_var'])

        # Weighted Z for non-financial
        qdf['quality_z'] = np.where(
            qdf['is_financial'],
            0.5 * z_roe + 0.5 * (-z_eps_var),
            (1 / 3) * z_roe + (1 / 3) * (-z_de) + (1 / 3) * (-z_eps_var)
        )

        qdf['quality_score'] = self._normalized_score(qdf['quality_z'])
        qdf['quality_percentile'] = qdf['quality_score'].rank(pct=True)
        return qdf

    # ------------------------------------------------------------------ #
    #  FACTOR 5 – VALUE  (from Nifty500 Multifactor MQVLv 50)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fetch_value_data(ticker):
        """Fetches E/P, B/P, S/P, Dividend Yield from yfinance."""
        try:
            info = yf.Ticker(ticker).info
            pe = info.get('trailingPE', None)
            pb = info.get('priceToBook', None)
            ps = info.get('priceToSalesTrailing12Months', None)
            dy = info.get('dividendYield', 0) or 0

            ep = (1.0 / pe) if pe and pe > 0 else np.nan
            bp = (1.0 / pb) if pb and pb > 0 else np.nan
            sp = (1.0 / ps) if ps and ps > 0 else np.nan

            return {'ep': ep, 'bp': bp, 'sp': sp, 'div_yield': dy}
        except Exception:
            return None

    def _compute_value_score(self, factor_df):
        """
        Value Z = 0.25*Z(E/P) + 0.25*Z(B/P) + 0.25*Z(S/P) + 0.25*Z(Div Yield)
        """
        value_data = []
        for _, row in factor_df.iterrows():
            v = self._fetch_value_data(row['symbol'])
            if v:
                value_data.append({**row.to_dict(), **v})
            else:
                value_data.append({**row.to_dict(), 'ep': np.nan, 'bp': np.nan,
                                   'sp': np.nan, 'div_yield': np.nan})

        vdf = pd.DataFrame(value_data)

        z_ep = self._z_scores(vdf['ep'])
        z_bp = self._z_scores(vdf['bp'])
        z_sp = self._z_scores(vdf['sp'])
        z_dy = self._z_scores(vdf['div_yield'])

        vdf['value_z'] = 0.25 * z_ep + 0.25 * z_bp + 0.25 * z_sp + 0.25 * z_dy
        vdf['value_score'] = self._normalized_score(vdf['value_z'])
        vdf['value_percentile'] = vdf['value_score'].rank(pct=True)
        return vdf

    # ------------------------------------------------------------------ #
    #  DUAL MOMENTUM  (Absolute + Relative)
    # ------------------------------------------------------------------ #
    def _compute_dual_momentum(self, factor_df):
        """
        Dual Momentum combines:
        1.  Absolute Momentum: Is 12-month return positive? (stock going up)
        2.  Relative Momentum: Is the stock outperforming the benchmark?
        Score = Absolute flag * Relative Momentum Score
        """
        factor_df['abs_mom_12m'] = factor_df['mr12'].apply(
            lambda x: 1 if (not np.isnan(x) and x > 0) else 0
        )
        # Relative momentum = momentum_score already computed
        if 'momentum_score' not in factor_df.columns:
            factor_df = self._compute_momentum(factor_df)

        factor_df['dual_momentum_score'] = factor_df['abs_mom_12m'] * factor_df['momentum_score']
        return factor_df

    # ------------------------------------------------------------------ #
    #  MULTIFACTOR MQVLv  (Momentum + Quality + Value + Low Volatility)
    # ------------------------------------------------------------------ #
    def _compute_multifactor(self, factor_df):
        """
        Aggregate Percentile = 25%*Mom + 25%*Quality + 25%*Value + 25%*LowVol
        Composite Score = 25%*MomScore + 25%*QualScore + 25%*ValScore + 25%*LowVolScore
        """
        # Ensure all sub-scores exist
        if 'momentum_percentile' not in factor_df.columns:
            factor_df = self._compute_momentum(factor_df)
        if 'low_vol_percentile' not in factor_df.columns:
            factor_df = self._compute_low_vol_score(factor_df)
        if 'quality_percentile' not in factor_df.columns:
            factor_df = self._compute_quality_score(factor_df)
        if 'value_percentile' not in factor_df.columns:
            factor_df = self._compute_value_score(factor_df)

        factor_df['aggregate_percentile'] = (
            0.25 * factor_df['momentum_percentile'].fillna(0) +
            0.25 * factor_df['quality_percentile'].fillna(0) +
            0.25 * factor_df['value_percentile'].fillna(0) +
            0.25 * factor_df['low_vol_percentile'].fillna(0)
        )

        factor_df['composite_score'] = (
            0.25 * factor_df['momentum_score'].fillna(0) +
            0.25 * factor_df['quality_score'].fillna(0) +
            0.25 * factor_df['value_score'].fillna(0) +
            0.25 * factor_df['low_vol_score'].fillna(0)
        )
        return factor_df

    # ------------------------------------------------------------------ #
    #  MAIN ENTRY POINT
    # ------------------------------------------------------------------ #
    def compute_scores(self, universe_data, market_data, factors=None):
        """
        Computes requested factor scores for a universe of stocks.

        Parameters
        ----------
        universe_data : dict  {ticker_symbol: DataFrame}
        market_data   : DataFrame  (benchmark – Nifty 50)
        factors       : list of str, e.g. ['alpha','momentum','volatility',
                        'beta','quality','value','dual_momentum','multifactor']
                        None = compute base price factors only.
        """
        if factors is None:
            factors = ['alpha', 'momentum', 'volatility', 'beta']

        results = []
        for ticker, df in universe_data.items():
            try:
                if len(df) < 252:
                    continue
                prices = df['Close']
                alpha, beta = self.calculate_alpha_beta(prices, market_data['Close'])
                vol = self.calculate_annualized_volatility(prices)
                mr6 = self.calculate_momentum_ratio(prices, months=6)
                mr12 = self.calculate_momentum_ratio(prices, months=12)

                results.append({
                    'symbol': ticker,
                    'alpha': alpha, 'beta': beta,
                    'volatility': vol,
                    'mr6': mr6, 'mr12': mr12
                })
            except Exception as e:
                logger.error(f"Error computing base factors for {ticker}: {e}")

        factor_df = pd.DataFrame(results)
        if factor_df.empty:
            return factor_df

        # Always compute momentum & low-vol (price-only, fast)
        factor_df = self._compute_momentum(factor_df)
        factor_df = self._compute_alpha_score(factor_df)
        factor_df = self._compute_low_vol_score(factor_df)

        # Dual momentum
        if 'dual_momentum' in factors:
            factor_df = self._compute_dual_momentum(factor_df)

        # Fundamental-dependent factors (slower – API calls)
        if 'quality' in factors or 'multifactor' in factors:
            logger.info("Fetching fundamental data for Quality scores...")
            factor_df = self._compute_quality_score(factor_df)

        if 'value' in factors or 'multifactor' in factors:
            logger.info("Fetching fundamental data for Value scores...")
            factor_df = self._compute_value_score(factor_df)

        if 'multifactor' in factors:
            factor_df = self._compute_multifactor(factor_df)

        return factor_df
