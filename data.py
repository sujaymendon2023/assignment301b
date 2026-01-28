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
        
        # Helper function to check for stop loss between entry and planned exit
        def check_stop_loss(entry_t, entry_v, planned_exit_t, planned_exit_v, sig):
            """Check if stop loss is hit before planned exit. Returns (exit_time, exit_value, exit_reason)"""
            stop_loss = config.STOP_LOSS_POINTS
            
            # Get actuals between entry and planned exit
            actuals_between = self.market_data.actuals[
                (self.market_data.actuals['timestamp'] > entry_t) & 
                (self.market_data.actuals['timestamp'] <= planned_exit_t)
            ]
            
            for _, act_row in actuals_between.iterrows():
                actual_price = float(act_row['actual_value'])
                if sig == 'BUY':
                    # Stop loss for BUY: price drops below entry - stop_loss
                    if actual_price <= entry_v - stop_loss:
                        return act_row['timestamp'], actual_price, 'STOP_LOSS'
                else:
                    # Stop loss for SELL: price rises above entry + stop_loss
                    if actual_price >= entry_v + stop_loss:
                        return act_row['timestamp'], actual_price, 'STOP_LOSS'
            
            # No stop loss hit, use planned exit
            return planned_exit_t, planned_exit_v, 'TARGET'
        
        # Strategy 1: Buy/Sell at first occurrence, exit at second occurrence (with stop loss)
        if entry_actual_value is not None and exit_actual_value is not None:
            final_exit_time, final_exit_value, exit_reason = check_stop_loss(
                entry_time, entry_actual_value, exit_time, exit_actual_value, signal
            )
            
            if signal == 'BUY':
                pnl = final_exit_value - entry_actual_value - config.TRANSACTION_COST - config.SLIPPAGE
            else:
                pnl = entry_actual_value - final_exit_value - config.TRANSACTION_COST - config.SLIPPAGE
            strategies['strategy_1'] = {
                'entry_time': entry_time,
                'entry_value': entry_actual_value,
                'exit_time': final_exit_time,
                'exit_value': final_exit_value,
                'pnl': pnl,
                'signal': signal,
                'exit_reason': exit_reason
            }
        
        # Strategy 2: Fixed time exits (1, 2, 3, 4, 5, 6 minutes) with stop loss
        for minutes in [1, 2, 3, 4, 5, 6]:
            exit_time_fixed, exit_value_fixed = self.market_data.get_actual_after_minutes(entry_time, minutes)
            if exit_value_fixed is not None and entry_actual_value is not None:
                final_exit_time, final_exit_value, exit_reason = check_stop_loss(
                    entry_time, entry_actual_value, exit_time_fixed, exit_value_fixed, signal
                )
                
                if signal == 'BUY':
                    pnl = final_exit_value - entry_actual_value - config.TRANSACTION_COST - config.SLIPPAGE
                else:
                    pnl = entry_actual_value - final_exit_value - config.TRANSACTION_COST - config.SLIPPAGE
                strategies[f'strategy_2_{minutes}min'] = {
                    'entry_time': entry_time,
                    'entry_value': entry_actual_value,
                    'exit_time': final_exit_time,
                    'exit_value': final_exit_value,
                    'pnl': pnl,
                    'signal': signal,
                    'exit_reason': exit_reason
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
            
            # If prediction target hit, get ACTUAL values at entry and exit times for P&L (with stop loss)
            if exit_found and entry_actual_value is not None:
                _, exit_actual_value_target = self.market_data.get_actual_at_time(exit_time_target)
                
                if exit_actual_value_target is not None:
                    final_exit_time, final_exit_value, exit_reason = check_stop_loss(
                        entry_time, entry_actual_value, exit_time_target, exit_actual_value_target, signal
                    )
                    
                    if signal == 'BUY':
                        pnl = final_exit_value - entry_actual_value - config.TRANSACTION_COST - config.SLIPPAGE
                    else:
                        pnl = entry_actual_value - final_exit_value - config.TRANSACTION_COST - config.SLIPPAGE
                    strategies[f'strategy_3_{target_points}pt'] = {
                        'entry_time': entry_time,
                        'entry_value': entry_actual_value,
                        'exit_time': final_exit_time,
                        'exit_value': final_exit_value,
                        'pnl': pnl,
                        'signal': signal,
                        'exit_reason': exit_reason
                    }
        
        # Strategy 4: Normalized Touch - check if actuals touched Pred Max or Pred Min
        # Get actuals BEFORE the prediction window (5 mins)
        actuals_before = self.market_data.get_actuals_before_time(
            start, 
            config.NORM_ACTUAL_LOOKBACK_MINUTES
        )
        
        norm_touch_result = {
            'touched_max': False,
            'touched_min': False,
            'first_touch': None,  # 'MAX' or 'MIN' - which was touched first
            'touch_time': None,
            'actual_normalized_pos': None,
            'pred_max_normalized': None,
            'pred_min_normalized': None,
            'signal': None,
            'actuals_count': len(actuals_before)
        }
        
        if not actuals_before.empty and not window_df.empty:
            # Normalize actuals (before window)
            actual_values = actuals_before['actual_value']
            actual_norm = normalize_series(actual_values)
            
            # Normalize predictions (the future window)
            pred_values = window_df['predictions']
            pred_norm = normalize_series(pred_values)
            
            # Max and min positions in normalized prediction space
            pred_max_normalized = pred_norm.max()  # Always ≈ 1.0
            pred_min_normalized = pred_norm.min()  # Always ≈ 0.0
            
            tolerance = config.NORM_TOUCH_TOLERANCE
            
            # Check ALL actuals to find which touched first (max or min)
            first_touch = None
            first_touch_time = None
            first_touch_value = None
            first_touch_norm_pos = None
            
            for i, (idx, row) in enumerate(actuals_before.iterrows()):
                norm_pos = actual_norm.iloc[i]
                touched_max_at_point = abs(norm_pos - pred_max_normalized) <= tolerance
                touched_min_at_point = abs(norm_pos - pred_min_normalized) <= tolerance
                
                if touched_max_at_point and first_touch is None:
                    first_touch = 'MAX'
                    first_touch_time = row['timestamp']
                    first_touch_value = row['actual_value']
                    first_touch_norm_pos = norm_pos
                    norm_touch_result['touched_max'] = True
                    break
                elif touched_min_at_point and first_touch is None:
                    first_touch = 'MIN'
                    first_touch_time = row['timestamp']
                    first_touch_value = row['actual_value']
                    first_touch_norm_pos = norm_pos
                    norm_touch_result['touched_min'] = True
                    break
            
            # Also check if any touched (for display purposes)
            for i in range(len(actual_norm)):
                norm_pos = actual_norm.iloc[i]
                if abs(norm_pos - pred_max_normalized) <= tolerance:
                    norm_touch_result['touched_max'] = True
                if abs(norm_pos - pred_min_normalized) <= tolerance:
                    norm_touch_result['touched_min'] = True
            
            # Position of last actual (for display)
            last_actual_pos = actual_norm.iloc[-1] if len(actual_norm) > 0 else None
            norm_touch_result['actual_normalized_pos'] = float(last_actual_pos) if last_actual_pos is not None else None
            norm_touch_result['pred_max_normalized'] = float(pred_max_normalized)
            norm_touch_result['pred_min_normalized'] = float(pred_min_normalized)
            norm_touch_result['first_touch'] = first_touch
            
            # Generate trade signal based on which was touched first
            if first_touch == 'MAX' and first_touch_value is not None:
                # Touched MAX first → SELL signal (fade the high)
                norm_touch_result['signal'] = 'SELL'
                norm_touch_result['touch_time'] = first_touch_time
                
                # Entry at touch point
                entry_value = float(first_touch_value)
                target_price = entry_value - config.NORM_TOUCH_TARGET_POINTS  # Target profit
                stop_loss_price = entry_value + config.STOP_LOSS_POINTS  # Stop loss
                
                # Find when actual reaches target OR stop loss (whichever first)
                all_actuals_after = self.market_data.actuals[
                    self.market_data.actuals['timestamp'] >= first_touch_time
                ]
                
                exit_found = False
                exit_time = None
                exit_value = None
                exit_reason = None
                for _, act_row in all_actuals_after.iterrows():
                    actual_price = float(act_row['actual_value'])
                    # Check stop loss first (price went up for SELL)
                    if actual_price >= stop_loss_price:
                        exit_time = act_row['timestamp']
                        exit_value = actual_price
                        exit_found = True
                        exit_reason = 'STOP_LOSS'
                        break
                    # Check target (price went down for SELL)
                    if actual_price <= target_price:
                        exit_time = act_row['timestamp']
                        exit_value = actual_price
                        exit_found = True
                        exit_reason = 'TARGET'
                        break
                
                if exit_found:
                    pnl = entry_value - exit_value - config.TRANSACTION_COST - config.SLIPPAGE
                    strategies['strategy_norm_touch'] = {
                        'entry_time': first_touch_time,
                        'entry_value': entry_value,
                        'exit_time': exit_time,
                        'exit_value': exit_value,
                        'pnl': pnl,
                        'touch_type': 'MAX',
                        'signal': 'SELL',
                        'exit_reason': exit_reason
                    }
            
            elif first_touch == 'MIN' and first_touch_value is not None:
                # Touched MIN first → BUY signal (fade the low)
                norm_touch_result['signal'] = 'BUY'
                norm_touch_result['touch_time'] = first_touch_time
                
                # Entry at touch point
                entry_value = float(first_touch_value)
                target_price = entry_value + config.NORM_TOUCH_TARGET_POINTS  # Target profit
                stop_loss_price = entry_value - config.STOP_LOSS_POINTS  # Stop loss
                
                # Find when actual reaches target OR stop loss (whichever first)
                all_actuals_after = self.market_data.actuals[
                    self.market_data.actuals['timestamp'] >= first_touch_time
                ]
                
                exit_found = False
                exit_time = None
                exit_value = None
                exit_reason = None
                for _, act_row in all_actuals_after.iterrows():
                    actual_price = float(act_row['actual_value'])
                    # Check stop loss first (price went down for BUY)
                    if actual_price <= stop_loss_price:
                        exit_time = act_row['timestamp']
                        exit_value = actual_price
                        exit_found = True
                        exit_reason = 'STOP_LOSS'
                        break
                    # Check target (price went up for BUY)
                    if actual_price >= target_price:
                        exit_time = act_row['timestamp']
                        exit_value = actual_price
                        exit_found = True
                        exit_reason = 'TARGET'
                        break
                
                if exit_found:
                    pnl = exit_value - entry_value - config.TRANSACTION_COST - config.SLIPPAGE
                    strategies['strategy_norm_touch'] = {
                        'entry_time': first_touch_time,
                        'entry_value': entry_value,
                        'exit_time': exit_time,
                        'exit_value': exit_value,
                        'pnl': pnl,
                        'touch_type': 'MIN',
                        'signal': 'BUY',
                        'exit_reason': exit_reason
                    }
            
            # Strategy: Norm Touch v2 - exit at predicted min/max time (with stop loss)
            # Entry on MAX → exit at MIN time, Entry on MIN → exit at MAX time
            if first_touch is not None and first_touch_value is not None:
                entry_value_v2 = float(first_touch_value)
                
                if first_touch == 'MAX':
                    # Entered on MAX (SELL), exit at predicted MIN time
                    exit_time_v2 = min_row['timestamp']
                    _, exit_value_v2 = self.market_data.get_actual_at_time(exit_time_v2)
                    
                    if exit_value_v2 is not None:
                        final_exit_time_v2, final_exit_value_v2, exit_reason_v2 = check_stop_loss(
                            first_touch_time, entry_value_v2, exit_time_v2, float(exit_value_v2), 'SELL'
                        )
                        pnl_v2 = entry_value_v2 - final_exit_value_v2 - config.TRANSACTION_COST - config.SLIPPAGE
                        strategies['strategy_norm_touch_v2'] = {
                            'entry_time': first_touch_time,
                            'entry_value': entry_value_v2,
                            'exit_time': final_exit_time_v2,
                            'exit_value': float(final_exit_value_v2),
                            'pnl': pnl_v2,
                            'touch_type': 'MAX',
                            'exit_at': 'MIN_TIME',
                            'signal': 'SELL',
                            'exit_reason': exit_reason_v2
                        }
                
                elif first_touch == 'MIN':
                    # Entered on MIN (BUY), exit at predicted MAX time
                    exit_time_v2 = max_row['timestamp']
                    _, exit_value_v2 = self.market_data.get_actual_at_time(exit_time_v2)
                    
                    if exit_value_v2 is not None:
                        final_exit_time_v2, final_exit_value_v2, exit_reason_v2 = check_stop_loss(
                            first_touch_time, entry_value_v2, exit_time_v2, float(exit_value_v2), 'BUY'
                        )
                        pnl_v2 = final_exit_value_v2 - entry_value_v2 - config.TRANSACTION_COST - config.SLIPPAGE
                        strategies['strategy_norm_touch_v2'] = {
                            'entry_time': first_touch_time,
                            'entry_value': entry_value_v2,
                            'exit_time': final_exit_time_v2,
                            'exit_value': float(final_exit_value_v2),
                            'pnl': pnl_v2,
                            'touch_type': 'MIN',
                            'exit_at': 'MAX_TIME',
                            'signal': 'BUY',
                            'exit_reason': exit_reason_v2
                        }
        
        # Order min/max by time (which occurred first)
        if min_occurred_first:
            first_event = {'type': 'MIN', 'time': min_row['timestamp'], 'prediction': float(min_row['predictions']), 'actual': min_actual_value}
            second_event = {'type': 'MAX', 'time': max_row['timestamp'], 'prediction': float(max_row['predictions']), 'actual': max_actual_value}
        else:
            first_event = {'type': 'MAX', 'time': max_row['timestamp'], 'prediction': float(max_row['predictions']), 'actual': max_actual_value}
            second_event = {'type': 'MIN', 'time': min_row['timestamp'], 'prediction': float(min_row['predictions']), 'actual': min_actual_value}
        
        # Get actuals for the entire chart period (5 min before + during prediction window)
        lookback_start = start - timedelta(minutes=5)
        actuals_full_period = self.market_data.actuals[
            (self.market_data.actuals['timestamp'] >= lookback_start) & 
            (self.market_data.actuals['timestamp'] <= end)
        ]
        actuals_chart_list = []
        for _, row in actuals_full_period.iterrows():
            actuals_chart_list.append({
                'timestamp': row['timestamp'],
                'actual_value': float(row['actual_value'])
            })
        
        # Also keep actuals before for normalized touch calculation
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
            'actuals_chart': actuals_chart_list,
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
