"""Helpers for loading predictions/actuals from Postgres and aligning timestamps."""

from datetime import datetime, timedelta
from typing import Optional, Tuple, List

import pandas as pd
from sqlalchemy import create_engine

import config

# Database connection string


def normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a series using min-max normalization.
    
    Returns:
        Series where 0 = local min, 1 = local max
    """
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)
DATABASE_URL = "postgresql+psycopg2://postgres:1234@localhost:5432/postgres"


def get_engine():
    """Create and return database engine."""
    return create_engine(DATABASE_URL)


def _combine_date_time(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    """Normalize date__ and time__ columns into datetime objects."""
    # Convert date to string format YYYYMMDD
    date_str = pd.to_datetime(date_series).dt.strftime('%Y%m%d')
    # Pad time to 4 digits
    time_padded = time_series.astype(str).str.zfill(4)
    combined = date_str + time_padded
    return pd.to_datetime(combined, format="%Y%m%d%H%M", errors="coerce")


class MarketData:
    """Handles loading and querying market data from the database."""
    
    def __init__(self):
        self.engine = get_engine()
        self.all_predictions = self._load_predictions()
        self.all_actuals = self._load_actuals()
        self.current_date = None
        self.predictions = self.all_predictions
        self.actuals = self.all_actuals
        
        # Set default date to the first available date
        available_dates = self.get_available_dates()
        if available_dates:
            self.set_date(available_dates[0])
    
    def get_available_dates(self) -> List[str]:
        """Get list of unique dates from predictions."""
        if self.all_predictions.empty:
            return []
        
        dates = self.all_predictions['timestamp'].dt.date.unique()
        return sorted([d.strftime('%Y-%m-%d') for d in dates])
    
    def set_date(self, date_str: str):
        """Filter data for a specific date."""
        self.current_date = date_str
        
        # Filter predictions for the selected date
        mask_pred = self.all_predictions['timestamp'].dt.strftime('%Y-%m-%d') == date_str
        self.predictions = self.all_predictions[mask_pred].copy().reset_index(drop=True)
        
        # Filter actuals for the selected date
        mask_actual = self.all_actuals['timestamp'].dt.strftime('%Y-%m-%d') == date_str
        self.actuals = self.all_actuals[mask_actual].copy().reset_index(drop=True)

    def _load_predictions(self) -> pd.DataFrame:
        """Load predictions from database."""
        with self.engine.connect() as connection:
            df = pd.read_sql(config.PREDICTIONS_QUERY, connection)

        df["timestamp"] = _combine_date_time(df["date__"], df["time__"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _load_actuals(self) -> pd.DataFrame:
        """Load actual values from database."""
        with self.engine.connect() as connection:
            df = pd.read_sql(config.ACTUALS_QUERY, connection)

        df["timestamp"] = _combine_date_time(df["date__"], df["time__"])
        df = df.rename(columns={"actual_value": "actual_value"})
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df[["timestamp", "actual_value"]]
        return df

    def get_time_windows(self, window_minutes: int = 15, increment_minutes: int = 15) -> List[Tuple[datetime, datetime]]:
        """Generate time windows for the slider.
        
        Args:
            window_minutes: Size of each window (fixed at 15 minutes)
            increment_minutes: How much to move forward for each new window (5, 10, or 15 minutes)
        """
        if self.predictions.empty:
            return []
        
        start_time = self.predictions['timestamp'].min()
        end_time = self.predictions['timestamp'].max()
        
        windows = []
        current = start_time
        while current + timedelta(minutes=window_minutes) <= end_time + timedelta(minutes=window_minutes):
            window_end = current + timedelta(minutes=window_minutes)
            windows.append((current, window_end))
            current = current + timedelta(minutes=increment_minutes)
            
            # Stop if we've gone past the data
            if current > end_time:
                break
        
        return windows
    
    def window(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Get predictions within a time window."""
        mask = (self.predictions["timestamp"] >= start) & (self.predictions["timestamp"] <= end)
        return self.predictions.loc[mask].copy()

    def get_actual_at_time(self, target: datetime) -> Tuple[Optional[datetime], Optional[float]]:
        """Get actual value at or nearest to target time."""
        if self.actuals.empty:
            return None, None
        
        # Find closest timestamp
        idx = (self.actuals['timestamp'] - target).abs().idxmin()
        row = self.actuals.iloc[idx]
        return row["timestamp"], row["actual_value"]

    def get_actual_after_minutes(self, start_time: datetime, minutes: int) -> Tuple[Optional[datetime], Optional[float]]:
        """Get actual value after specified minutes from start time."""
        target_time = start_time + timedelta(minutes=minutes)
        return self.get_actual_at_time(target_time)
    
    def get_actuals_before_time(self, end_time: datetime, lookback_minutes: int = 15) -> pd.DataFrame:
        """Get actual values in the window before a given time.
        
        Args:
            end_time: The end time (exclusive) - typically the start of prediction window
            lookback_minutes: How many minutes to look back
            
        Returns:
            DataFrame with actuals in the lookback window
        """
        start_time = end_time - timedelta(minutes=lookback_minutes)
        mask = (self.actuals['timestamp'] >= start_time) & (self.actuals['timestamp'] < end_time)
        return self.actuals.loc[mask].copy()


class TradingStrategy:
    """Calculate profit/loss for different trading strategies."""
    
    def __init__(self, market_data: MarketData):
        self.market_data = market_data
    
    def analyze_window(self, start: datetime, end: datetime) -> dict:
        """Analyze a time window and return signal and strategy results."""
        window_df = self.market_data.window(start, end)
        
        if window_df.empty:
            return None
        
        # Find min and max predictions
        min_idx = window_df['predictions'].idxmin()
        max_idx = window_df['predictions'].idxmax()
        
        min_row = window_df.loc[min_idx]
        max_row = window_df.loc[max_idx]
        
        # Determine signal based on time order
        # If min occurred first, it's a BUY (buy low, sell high)
        # If max occurred first, it's a SELL (sell high, buy low)
        min_occurred_first = min_row['timestamp'] < max_row['timestamp']
        signal = 'BUY' if min_occurred_first else 'SELL'
        
        # Alternative signal based on value comparison
        signal_value = 'BUY' if min_row['predictions'] < max_row['predictions'] else 'SELL'
        
        # Get actual values at min and max times
        min_actual_time, min_actual_value = self.market_data.get_actual_at_time(min_row['timestamp'])
        max_actual_time, max_actual_value = self.market_data.get_actual_at_time(max_row['timestamp'])
        
        # Calculate strategies
        strategies = {}
        
        # Determine entry point (first occurring time)
        if min_occurred_first:
            entry_time = min_row['timestamp']
            entry_actual_value = min_actual_value
            exit_time = max_row['timestamp']
            exit_actual_value = max_actual_value
        else:
            entry_time = max_row['timestamp']
            entry_actual_value = max_actual_value
            exit_time = min_row['timestamp']
            exit_actual_value = min_actual_value
        
        # Strategy 1: Buy/Sell at first occurrence, exit at second occurrence
        if entry_actual_value is not None and exit_actual_value is not None:
            if signal == 'BUY':
                pnl = exit_actual_value - entry_actual_value - config.TRANSACTION_COST - config.SLIPPAGE
            else:
                pnl = entry_actual_value - exit_actual_value - config.TRANSACTION_COST - config.SLIPPAGE
            strategies['strategy_1'] = {
                'entry_time': entry_time,
                'entry_value': entry_actual_value,
                'exit_time': exit_time,
                'exit_value': exit_actual_value,
                'pnl': pnl
            }
        
        # Strategy 2: Fixed time exits (1, 2, 3, 4, 5, 6 minutes)
        for minutes in [1, 2, 3, 4, 5, 6]:
            exit_time_fixed, exit_value_fixed = self.market_data.get_actual_after_minutes(entry_time, minutes)
            if exit_value_fixed is not None and entry_actual_value is not None:
                if signal == 'BUY':
                    pnl = exit_value_fixed - entry_actual_value - config.TRANSACTION_COST - config.SLIPPAGE
                else:
                    pnl = entry_actual_value - exit_value_fixed - config.TRANSACTION_COST - config.SLIPPAGE
                strategies[f'strategy_2_{minutes}min'] = {
                    'entry_time': entry_time,
                    'entry_value': entry_actual_value,
                    'exit_time': exit_time_fixed,
                    'exit_value': exit_value_fixed,
                    'pnl': pnl
                }
        
        # Strategy 3: Exit after N points movement in PREDICTION value (2pt and 3pt)
        # Entry prediction value at entry time
        if min_occurred_first:
            entry_prediction_value = float(min_row['predictions'])
        else:
            entry_prediction_value = float(max_row['predictions'])
        
        # Get predictions after entry time (for target strategies)
        predictions_after = self.market_data.predictions[
            self.market_data.predictions['timestamp'] >= entry_time
        ]
        
        # Calculate for each target point (2pt and 3pt)
        for target_points in [2, 3]:
            if signal == 'BUY':
                # For BUY: exit when prediction goes UP by N points
                target_prediction = entry_prediction_value + target_points
            else:
                # For SELL: exit when prediction goes DOWN by N points
                target_prediction = entry_prediction_value - target_points
            
            # Find when target is hit in PREDICTIONS
            exit_found = False
            exit_time_target = None
            for _, row in predictions_after.iterrows():
                pred_value = float(row['predictions'])
                if signal == 'BUY' and pred_value >= target_prediction:
                    exit_time_target = row['timestamp']
                    exit_found = True
                    break
                elif signal == 'SELL' and pred_value <= target_prediction:
                    exit_time_target = row['timestamp']
                    exit_found = True
                    break
            
            # If prediction target hit, get ACTUAL values at entry and exit times for P&L
            if exit_found and entry_actual_value is not None:
                _, exit_actual_value_target = self.market_data.get_actual_at_time(exit_time_target)
                
                if exit_actual_value_target is not None:
                    if signal == 'BUY':
                        pnl = exit_actual_value_target - entry_actual_value - config.TRANSACTION_COST - config.SLIPPAGE
                    else:
                        pnl = entry_actual_value - exit_actual_value_target - config.TRANSACTION_COST - config.SLIPPAGE
                    strategies[f'strategy_3_{target_points}pt'] = {
                        'entry_time': entry_time,
                        'entry_value': entry_actual_value,
                        'exit_time': exit_time_target,
                        'exit_value': exit_actual_value_target,
                        'pnl': pnl
                    }
        
        # Strategy 4: Normalized Touch - compare relative positions
        # Get actuals BEFORE the prediction window
        actuals_before = self.market_data.get_actuals_before_time(
            start, 
            config.NORM_ACTUAL_LOOKBACK_MINUTES
        )
        
        norm_touch_result = {
            'touched_max': False,
            'touched_min': False,
            'actual_normalized_pos': None,
            'pred_max_normalized': None,
            'pred_min_normalized': None,
            'signal': None
        }
        
        if not actuals_before.empty and not window_df.empty:
            # Normalize actuals (before window)
            actual_values = actuals_before['actual_value']
            actual_norm = normalize_series(actual_values)
            
            # Position of last actual (just before window starts)
            last_actual_pos = actual_norm.iloc[-1] if len(actual_norm) > 0 else None
            last_actual_time = actuals_before['timestamp'].iloc[-1] if len(actuals_before) > 0 else None
            last_actual_value = actual_values.iloc[-1] if len(actual_values) > 0 else None
            
            # Normalize predictions (the future window)
            pred_values = window_df['predictions']
            pred_norm = normalize_series(pred_values)
            
            # Max and min positions in normalized prediction space
            pred_max_normalized = pred_norm.max()  # Always ≈ 1.0
            pred_min_normalized = pred_norm.min()  # Always ≈ 0.0
            
            norm_touch_result['actual_normalized_pos'] = float(last_actual_pos) if last_actual_pos is not None else None
            norm_touch_result['pred_max_normalized'] = float(pred_max_normalized)
            norm_touch_result['pred_min_normalized'] = float(pred_min_normalized)
            
            tolerance = config.NORM_TOUCH_TOLERANCE
            
            # Check if actual touched future predicted MAX (short signal)
            if last_actual_pos is not None:
                touched_max = abs(last_actual_pos - pred_max_normalized) <= tolerance
                touched_min = abs(last_actual_pos - pred_min_normalized) <= tolerance
                
                norm_touch_result['touched_max'] = touched_max
                norm_touch_result['touched_min'] = touched_min
                
                # Generate trade signal and calculate P&L
                if touched_max and last_actual_value is not None:
                    # Actual at high of range, predicted max → SELL (fade)
                    norm_touch_result['signal'] = 'SELL'
                    
                    # Entry at last actual before window, exit at min prediction time
                    _, exit_value = self.market_data.get_actual_at_time(min_row['timestamp'])
                    if exit_value is not None:
                        pnl = last_actual_value - exit_value - config.TRANSACTION_COST - config.SLIPPAGE
                        strategies['strategy_norm_touch_max'] = {
                            'entry_time': last_actual_time,
                            'entry_value': last_actual_value,
                            'exit_time': min_row['timestamp'],
                            'exit_value': exit_value,
                            'pnl': pnl,
                            'actual_norm_pos': float(last_actual_pos),
                            'touched': True
                        }
                
                if touched_min and last_actual_value is not None:
                    # Actual at low of range, predicted min → BUY (fade)
                    norm_touch_result['signal'] = 'BUY'
                    
                    # Entry at last actual before window, exit at max prediction time
                    _, exit_value = self.market_data.get_actual_at_time(max_row['timestamp'])
                    if exit_value is not None:
                        pnl = exit_value - last_actual_value - config.TRANSACTION_COST - config.SLIPPAGE
                        strategies['strategy_norm_touch_min'] = {
                            'entry_time': last_actual_time,
                            'entry_value': last_actual_value,
                            'exit_time': max_row['timestamp'],
                            'exit_value': exit_value,
                            'pnl': pnl,
                            'actual_norm_pos': float(last_actual_pos),
                            'touched': True
                        }
        
        # Order min/max by time (which occurred first)
        if min_occurred_first:
            first_event = {'type': 'MIN', 'time': min_row['timestamp'], 'prediction': float(min_row['predictions']), 'actual': min_actual_value}
            second_event = {'type': 'MAX', 'time': max_row['timestamp'], 'prediction': float(max_row['predictions']), 'actual': max_actual_value}
        else:
            first_event = {'type': 'MAX', 'time': max_row['timestamp'], 'prediction': float(max_row['predictions']), 'actual': max_actual_value}
            second_event = {'type': 'MIN', 'time': min_row['timestamp'], 'prediction': float(min_row['predictions']), 'actual': min_actual_value}
        
        # Get actuals for the last 5 minutes before the prediction window (for chart display)
        actuals_before_5min = self.market_data.get_actuals_before_time(start, lookback_minutes=5)
        actuals_before_list = []
        for _, row in actuals_before_5min.iterrows():
            actuals_before_list.append({
                'timestamp': row['timestamp'],
                'actual_value': float(row['actual_value'])
            })
        
        return {
            'window_start': start,
            'window_end': end,
            'predictions': window_df.to_dict('records'),
            'actuals_before': actuals_before_list,
            'actuals_before_count': len(actuals_before_list),
            'min_prediction': {
                'time': min_row['timestamp'],
                'value': float(min_row['predictions']),
                'actual_time': min_actual_time,
                'actual_value': float(min_actual_value) if min_actual_value is not None else None
            },
            'max_prediction': {
                'time': max_row['timestamp'],
                'value': float(max_row['predictions']),
                'actual_time': max_actual_time,
                'actual_value': float(max_actual_value) if max_actual_value is not None else None
            },
            'first_event': first_event,
            'second_event': second_event,
            'signal_time_based': signal,
            'signal_value_based': signal_value,
            'difference': float(max_row['predictions'] - min_row['predictions']),
            'norm_touch': norm_touch_result,
            'strategies': strategies
        }
