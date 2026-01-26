"""Flask application for trading strategy analysis and P&L visualization."""

from datetime import datetime
from flask import Flask, render_template, jsonify, request

from data import MarketData, TradingStrategy
import config

app = Flask(__name__)

# Global data cache
market_data = None
trading_strategy = None


def init_data():
    """Initialize market data and trading strategy."""
    global market_data, trading_strategy
    if market_data is None:
        market_data = MarketData()
        trading_strategy = TradingStrategy(market_data)


@app.route('/')
def index():
    """Main page with the trading dashboard."""
    init_data()
    return render_template('index.html')


@app.route('/api/dates')
def get_dates():
    """Get available dates from predictions."""
    init_data()
    dates = market_data.get_available_dates()
    current_date = market_data.current_date
    return jsonify({
        'dates': dates,
        'current_date': current_date
    })


@app.route('/api/windows')
def get_windows():
    """Get available time windows."""
    init_data()
    
    # Set date if provided
    date = request.args.get('date', type=str)
    if date:
        market_data.set_date(date)
    
    increment = request.args.get('increment', type=int, default=15)
    windows = market_data.get_time_windows(window_minutes=15, increment_minutes=increment)
    
    return jsonify({
        'windows': [
            {
                'index': idx,
                'start': start.isoformat(),
                'end': end.isoformat(),
                'label': f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')}"
            }
            for idx, (start, end) in enumerate(windows)
        ]
    })


@app.route('/api/analyze')
def analyze():
    """Analyze a specific time window."""
    init_data()
    
    # Set date if provided
    date = request.args.get('date', type=str)
    if date:
        market_data.set_date(date)
    
    window_idx = request.args.get('window', type=int, default=0)
    increment = request.args.get('increment', type=int, default=15)
    windows = market_data.get_time_windows(window_minutes=15, increment_minutes=increment)
    
    if window_idx < 0 or window_idx >= len(windows):
        return jsonify({'error': 'Invalid window index'}), 400
    
    start, end = windows[window_idx]
    analysis = trading_strategy.analyze_window(start, end)
    
    if analysis is None:
        return jsonify({'error': 'No data for this window'}), 404
    
    # Serialize the analysis result
    result = {
        'window_start': analysis['window_start'].isoformat(),
        'window_end': analysis['window_end'].isoformat(),
        'predictions': [
            {
                'time': p['timestamp'].isoformat() if isinstance(p['timestamp'], datetime) else p['timestamp'],
                'value': float(p['predictions'])
            }
            for p in analysis['predictions']
        ],
        'actuals_before': [
            {
                'time': a['timestamp'].isoformat() if isinstance(a['timestamp'], datetime) else a['timestamp'],
                'value': float(a['actual_value'])
            }
            for a in analysis.get('actuals_before', [])
        ],
        'actuals_before_count': analysis.get('actuals_before_count', 0),
        'min_prediction': {
            'time': analysis['min_prediction']['time'].isoformat(),
            'value': analysis['min_prediction']['value'],
            'actual_time': analysis['min_prediction']['actual_time'].isoformat() if analysis['min_prediction']['actual_time'] else None,
            'actual_value': analysis['min_prediction']['actual_value']
        },
        'max_prediction': {
            'time': analysis['max_prediction']['time'].isoformat(),
            'value': analysis['max_prediction']['value'],
            'actual_time': analysis['max_prediction']['actual_time'].isoformat() if analysis['max_prediction']['actual_time'] else None,
            'actual_value': analysis['max_prediction']['actual_value']
        },
        'first_event': {
            'type': analysis['first_event']['type'],
            'time': analysis['first_event']['time'].isoformat(),
            'prediction': analysis['first_event']['prediction'],
            'actual': analysis['first_event']['actual']
        },
        'second_event': {
            'type': analysis['second_event']['type'],
            'time': analysis['second_event']['time'].isoformat(),
            'prediction': analysis['second_event']['prediction'],
            'actual': analysis['second_event']['actual']
        },
        'signal_time_based': analysis['signal_time_based'],
        'signal_value_based': analysis['signal_value_based'],
        'difference': analysis['difference'],
        'norm_touch': {
            'touched_max': bool(analysis.get('norm_touch', {}).get('touched_max', False)),
            'touched_min': bool(analysis.get('norm_touch', {}).get('touched_min', False)),
            'first_touch': analysis.get('norm_touch', {}).get('first_touch'),
            'actual_normalized_pos': float(analysis.get('norm_touch', {}).get('actual_normalized_pos')) if analysis.get('norm_touch', {}).get('actual_normalized_pos') is not None else None,
            'pred_max_normalized': float(analysis.get('norm_touch', {}).get('pred_max_normalized')) if analysis.get('norm_touch', {}).get('pred_max_normalized') is not None else None,
            'pred_min_normalized': float(analysis.get('norm_touch', {}).get('pred_min_normalized')) if analysis.get('norm_touch', {}).get('pred_min_normalized') is not None else None,
            'signal': analysis.get('norm_touch', {}).get('signal'),
            'actuals_count': analysis.get('norm_touch', {}).get('actuals_count', 0)
        },
        'strategies': {}
    }
    
    # Serialize strategies
    for strategy_name, strategy_data in analysis['strategies'].items():
        result['strategies'][strategy_name] = {
            'entry_time': strategy_data['entry_time'].isoformat(),
            'entry_value': float(strategy_data['entry_value']),
            'exit_time': strategy_data['exit_time'].isoformat(),
            'exit_value': float(strategy_data['exit_value']),
            'pnl': float(strategy_data['pnl'])
        }
    
    return jsonify(result)


@app.route('/api/strategies')
def get_strategies():
    """Get strategy definitions."""
    return jsonify(config.STRATEGIES)


@app.route('/api/summary')
def get_summary():
    """Get P&L summary for all time windows and all strategies."""
    init_data()
    
    # Set date if provided
    date = request.args.get('date', type=str)
    if date:
        market_data.set_date(date)
    
    increment = request.args.get('increment', type=int, default=15)
    windows = market_data.get_time_windows(window_minutes=15, increment_minutes=increment)
    
    # Strategy keys in order
    strategy_keys = ['strategy_1', 'strategy_2_1min', 'strategy_2_2min', 'strategy_2_3min', 'strategy_2_4min', 'strategy_2_5min', 'strategy_2_6min', 'strategy_3_2pt', 'strategy_3_3pt', 'strategy_norm_touch', 'strategy_norm_touch_v2']
    
    rows = []
    totals = {key: 0.0 for key in strategy_keys}
    counts = {key: 0 for key in strategy_keys}
    
    for idx, (start, end) in enumerate(windows):
        analysis = trading_strategy.analyze_window(start, end)
        
        row = {
            'index': idx,
            'time_window': f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')}",
            'signal': analysis['signal_time_based'] if analysis else None,
            'difference': analysis['difference'] if analysis else None,
            'strategies': {}
        }
        
        if analysis:
            for key in strategy_keys:
                if key in analysis['strategies']:
                    pnl = float(analysis['strategies'][key]['pnl'])
                    row['strategies'][key] = pnl
                    totals[key] += pnl
                    counts[key] += 1
                else:
                    row['strategies'][key] = None
        else:
            for key in strategy_keys:
                row['strategies'][key] = None
        
        rows.append(row)
    
    return jsonify({
        'rows': rows,
        'totals': totals,
        'counts': counts,
        'strategy_keys': strategy_keys
    })


@app.route('/api/refresh')
def refresh_data():
    """Force refresh of market data from database."""
    global market_data, trading_strategy
    market_data = None
    trading_strategy = None
    init_data()
    
    # Return updated dates list
    dates = market_data.get_available_dates()
    return jsonify({
        'status': 'success', 
        'message': 'Data refreshed',
        'dates': dates,
        'current_date': market_data.current_date
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

