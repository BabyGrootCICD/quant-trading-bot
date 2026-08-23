"""Single source of truth for the model feature set.

Both LogisticModel and XGBoostModel train on this list. Keeping it in one
place stops the two models drifting apart (xgboost was still on the original
11-column list while the feature engineer produced 26).

Every name here must exist as a column on the `features` table -- see
migrations/003_add_enhanced_features.sql.
"""

FEATURE_COLS = [
    "log_return_1h", "log_return_2h", "log_return_4h",
    "log_return_8h", "log_return_24h",
    "rsi_14", "rsi_7",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "bb_position",
    "volume_ratio", "volume_ratio_48",
    "atr_14", "atr_14_pct",
    "williams_r",
    "stoch_k", "stoch_d",
    "ha_trend",
    "close_pct_ma20", "close_pct_ma50",
    "vol_20", "skew_20",
]
