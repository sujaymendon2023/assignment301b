"""Configuration constants for the trading strategy analyzer."""

# Time window for analysis
SLA_INTERVAL_MINUTES = 15

# Predictions query
PREDICTIONS_QUERY = """
SELECT 
 TO_CHAR(
        TO_TIMESTAMP(date__ || LPAD(time__::text, 4, '0'), 'YYYYMMDDHH24MI')
        + INTERVAL '15 minutes',
        'YYYYMMDD'
    ) AS date__,
    TO_CHAR(
        TO_TIMESTAMP(date__ || LPAD(time__::text, 4, '0'), 'YYYYMMDDHH24MI')
        + INTERVAL '15 minutes',
        'HH24MI'
    ) AS time__,
    predictions,
    "Insert_date_time"
FROM predictions
ORDER BY date__, time__;
"""

# Actuals query - NIFTY BANK data (loads all dates, filtered in Python)
ACTUALS_QUERY = """
SELECT 
    DATE(s.exchange_timestamp) AS date__,
    TO_CHAR(s.exchange_timestamp::timestamp,'HH24MI') AS time__,
    AVG(s.last_price) AS actual_value
FROM 
    stock_data s
INNER JOIN 
    instruments i 
ON 
    s.instrument_token = i.instrument_token
    AND DATE(s.exchange_timestamp) = DATE(i.snapshot_date)
WHERE 
    exchange = 'NSE' 
    AND segment = 'INDICES' 
    AND name = 'NIFTY BANK'
GROUP BY date__, time__
ORDER BY date__, time__;
"""

# Strategy configurations
STRATEGIES = {
    'strategy_1': {
        'name': 'Min/Max Entry-Exit',
        'description': 'Buy/Sell at min time, exit at max time'
    },
    'strategy_2_1min': {
        'name': 'Fixed Exit 1min',
        'description': 'Buy/Sell at min time, exit after 1 minute'
    },
    'strategy_2_2min': {
        'name': 'Fixed Exit 2min',
        'description': 'Buy/Sell at min time, exit after 2 minutes'
    },
    'strategy_2_3min': {
        'name': 'Fixed Exit 3min',
        'description': 'Buy/Sell at min time, exit after 3 minutes'
    },
    'strategy_2_4min': {
        'name': 'Fixed Exit 4min',
        'description': 'Buy/Sell at min time, exit after 4 minutes'
    },
    'strategy_2_5min': {
        'name': 'Fixed Exit 5min',
        'description': 'Buy/Sell at min time, exit after 5 minutes'
    },
    'strategy_2_6min': {
        'name': 'Fixed Exit 6min',
        'description': 'Buy/Sell at min time, exit after 6 minutes'
    },
    'strategy_3_2pt': {
        'name': 'Target Exit 2pts',
        'description': 'Exit when prediction moves 2 points'
    },
    'strategy_3_3pt': {
        'name': 'Target Exit 3pts',
        'description': 'Exit when prediction moves 3 points'
    },
    'strategy_norm_touch': {
        'name': 'Norm Touch',
        'description': 'Actual touched pred extreme first → trade with +30pt exit'
    },
    'strategy_norm_touch_v2': {
        'name': 'Norm Touch v2',
        'description': 'Exit at predicted min/max time (opposite extreme)'
    }
}

# Normalized Touch Strategy Parameters
NORM_TOUCH_TOLERANCE = 0.05  # Tolerance for normalized comparison (5%)
NORM_TOUCH_EXIT_TOLERANCE = 0.05  # Exit tolerance for v2 (5%)
NORM_ACTUAL_LOOKBACK_MINUTES = 2  # Lookback window for actuals before prediction window (to detect signal)
NORM_TOUCH_TARGET_POINTS = 30  # Target profit points for Norm Touch

# Trading parameters
TRANSACTION_COST = 0.05  # Transaction cost per trade
SLIPPAGE = 0.1  # Slippage per trade
STOP_LOSS_POINTS = 10  # Global stop loss for all strategies (in points)
