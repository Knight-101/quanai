import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List
import queue
import threading
from dataclasses import dataclass
import json
import requests

logger = logging.getLogger(__name__)

@dataclass
class AlertConfig:
    level: str
    threshold: float
    message_template: str
    webhook_url: str = None
    email: str = None

class TradingDashboard:
    def __init__(
        self,
        update_interval: int = 5,
        max_history: int = 1000,
        alert_configs: List[AlertConfig] = None
    ):
        self.update_interval = update_interval
        self.max_history = max_history
        self.alert_configs = alert_configs or self._default_alert_configs()
        
        # Initialize data storage
        self.metrics_history = {
            'timestamp': [],
            'equity': [],
            'pnl': [],
            'drawdown': [],
            'leverage': [],
            'var': [],
            'sharpe': [],
            'positions': [],
            'liquidation_risk': []
        }
        
        # Message queue for real-time updates
        self.message_queue = queue.Queue()
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
    def _default_alert_configs(self) -> List[AlertConfig]:
        """Default alert configurations"""
        return [
            AlertConfig(
                level="info",
                threshold=0.5,
                message_template="Position size exceeds {threshold}% of portfolio"
            ),
            AlertConfig(
                level="warning",
                threshold=0.7,
                message_template="Leverage utilization at {value}%, approaching limit"
            ),
            AlertConfig(
                level="critical",
                threshold=0.9,
                message_template="Drawdown at {value}%, near maximum allowed"
            )
        ]
        
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Quantitative Trading Dashboard"),
                html.Div(id='last-update-time')
            ], className='header'),
            
            # Main metrics
            html.Div([
                html.Div([
                    html.H3("Portfolio Overview"),
                    dcc.Graph(id='equity-chart'),
                    html.Div([
                        html.Div(id='current-equity'),
                        html.Div(id='daily-pnl'),
                        html.Div(id='sharpe-ratio')
                    ], className='metrics-row')
                ], className='chart-container'),
                
                html.Div([
                    html.H3("Risk Metrics"),
                    dcc.Graph(id='risk-metrics-chart'),
                    html.Div([
                        html.Div(id='current-drawdown'),
                        html.Div(id='var-metric'),
                        html.Div(id='leverage-ratio')
                    ], className='metrics-row')
                ], className='chart-container')
            ], className='main-row'),
            
            # Positions and alerts
            html.Div([
                html.Div([
                    html.H3("Active Positions"),
                    html.Div(id='positions-table')
                ], className='positions-container'),
                
                html.Div([
                    html.H3("Alerts"),
                    html.Div(id='alerts-panel')
                ], className='alerts-container')
            ], className='bottom-row'),
            
            # Hidden divs for storing intermediate data
            html.Div(id='intermediate-value', style={'display': 'none'}),
            
            # Update interval
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval * 1000,  # milliseconds
                n_intervals=0
            )
        ])
        
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('equity-chart', 'figure'),
             Output('risk-metrics-chart', 'figure'),
             Output('current-equity', 'children'),
             Output('daily-pnl', 'children'),
             Output('sharpe-ratio', 'children'),
             Output('current-drawdown', 'children'),
             Output('var-metric', 'children'),
             Output('leverage-ratio', 'children'),
             Output('positions-table', 'children'),
             Output('alerts-panel', 'children'),
             Output('last-update-time', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_metrics(_):
            # Get latest data
            latest_data = self._get_latest_data()
            
            # Create figures
            equity_fig = self._create_equity_chart()
            risk_fig = self._create_risk_chart()
            
            # Format metrics
            current_equity = f"Equity: ${latest_data['equity']:,.2f}"
            daily_pnl = f"Daily P&L: {latest_data['pnl']:+,.2f}"
            sharpe = f"Sharpe Ratio: {latest_data['sharpe']:.2f}"
            drawdown = f"Drawdown: {latest_data['drawdown']:.2%}"
            var = f"Value at Risk: ${latest_data['var']:,.2f}"
            leverage = f"Leverage: {latest_data['leverage']:.2f}x"
            
            # Create positions table
            positions_table = self._create_positions_table(latest_data['positions'])
            
            # Check alerts
            alerts_panel = self._create_alerts_panel()
            
            # Update time
            update_time = f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return (equity_fig, risk_fig, current_equity, daily_pnl, sharpe,
                    drawdown, var, leverage, positions_table, alerts_panel, update_time)
                    
    def _create_equity_chart(self) -> go.Figure:
        """Create equity chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.metrics_history['timestamp'],
            y=self.metrics_history['equity'],
            name='Portfolio Equity',
            line=dict(color='#2ecc71')
        ))
        
        fig.update_layout(
            title='Portfolio Equity Curve',
            xaxis_title='Time',
            yaxis_title='Equity ($)',
            template='plotly_dark',
            height=400
        )
        
        return fig
        
    def _create_risk_chart(self) -> go.Figure:
        """Create risk metrics chart"""
        fig = go.Figure()
        
        # Add drawdown
        fig.add_trace(go.Scatter(
            x=self.metrics_history['timestamp'],
            y=self.metrics_history['drawdown'],
            name='Drawdown',
            line=dict(color='#e74c3c')
        ))
        
        # Add VaR
        fig.add_trace(go.Scatter(
            x=self.metrics_history['timestamp'],
            y=self.metrics_history['var'],
            name='VaR',
            line=dict(color='#f1c40f')
        ))
        
        # Add leverage
        fig.add_trace(go.Scatter(
            x=self.metrics_history['timestamp'],
            y=self.metrics_history['leverage'],
            name='Leverage',
            line=dict(color='#3498db')
        ))
        
        fig.update_layout(
            title='Risk Metrics',
            xaxis_title='Time',
            yaxis_title='Value',
            template='plotly_dark',
            height=400
        )
        
        return fig
        
    def _create_positions_table(self, positions: List[Dict]) -> html.Table:
        """Create positions table"""
        headers = ['Asset', 'Size', 'Entry Price', 'Current Price', 'PnL', 'Funding']
        
        return html.Table([
            html.Thead(html.Tr([html.Th(col) for col in headers])),
            html.Tbody([
                html.Tr([
                    html.Td(pos['asset']),
                    html.Td(f"{pos['size']:.4f}"),
                    html.Td(f"${pos['entry_price']:,.2f}"),
                    html.Td(f"${pos['current_price']:,.2f}"),
                    html.Td(f"{pos['pnl']:+,.2f}"),
                    html.Td(f"{pos['funding']:+,.2f}")
                ]) for pos in positions
            ])
        ])
        
    def _create_alerts_panel(self) -> html.Div:
        """Create alerts panel"""
        alerts = []
        for alert in self._check_alerts():
            alerts.append(html.Div(
                alert['message'],
                className=f"alert alert-{alert['level']}"
            ))
        return html.Div(alerts)
        
    def _check_alerts(self) -> List[Dict]:
        """Check for alert conditions"""
        alerts = []
        latest_data = self._get_latest_data()
        
        for config in self.alert_configs:
            if self._check_alert_condition(latest_data, config):
                alert = {
                    'level': config.level,
                    'message': config.message_template.format(
                        value=latest_data.get(config.level, 0),
                        threshold=config.threshold
                    )
                }
                alerts.append(alert)
                self._send_alert(alert, config)
                
        return alerts
        
    def _check_alert_condition(self, data: Dict, config: AlertConfig) -> bool:
        """Check if alert condition is met"""
        value = data.get(config.level, 0)
        return value > config.threshold
        
    def _send_alert(self, alert: Dict, config: AlertConfig):
        """Send alert to configured endpoints"""
        if config.webhook_url:
            try:
                requests.post(
                    config.webhook_url,
                    json={
                        'text': alert['message'],
                        'level': alert['level']
                    }
                )
            except Exception as e:
                logger.error(f"Failed to send webhook alert: {str(e)}")
                
        if config.email:
            # Implement email alerting here
            pass
            
    def _get_latest_data(self) -> Dict:
        """Get latest metrics data"""
        if not self.metrics_history['timestamp']:
            return {
                'equity': 0,
                'pnl': 0,
                'drawdown': 0,
                'leverage': 0,
                'var': 0,
                'sharpe': 0,
                'positions': [],
                'liquidation_risk': 0
            }
            
        return {
            'equity': self.metrics_history['equity'][-1],
            'pnl': self.metrics_history['pnl'][-1],
            'drawdown': self.metrics_history['drawdown'][-1],
            'leverage': self.metrics_history['leverage'][-1],
            'var': self.metrics_history['var'][-1],
            'sharpe': self.metrics_history['sharpe'][-1],
            'positions': self.metrics_history['positions'][-1],
            'liquidation_risk': self.metrics_history['liquidation_risk'][-1]
        }
        
    def _update_loop(self):
        """Background update loop"""
        while True:
            try:
                # Get new data from queue
                new_data = self.message_queue.get(timeout=self.update_interval)
                
                # Update history
                timestamp = datetime.now()
                self.metrics_history['timestamp'].append(timestamp)
                
                for key, value in new_data.items():
                    if key in self.metrics_history:
                        self.metrics_history[key].append(value)
                        
                        # Maintain history size
                        if len(self.metrics_history[key]) > self.max_history:
                            self.metrics_history[key].pop(0)
                            
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in update loop: {str(e)}")
                
    def update_metrics(self, metrics: Dict):
        """Update dashboard with new metrics"""
        try:
            self.message_queue.put(metrics)
        except Exception as e:
            logger.error(f"Failed to update metrics: {str(e)}")
            
    def run(self, host='localhost', port=8050, debug=False):
        """Run the dashboard"""
        self.app.run_server(host=host, port=port, debug=debug)
        
    def shutdown(self):
        """Cleanup resources"""
        # Stop update thread
        if self.update_thread.is_alive():
            self.update_thread.join(timeout=1)
            
if __name__ == '__main__':
    # Example usage
    dashboard = TradingDashboard()
    dashboard.run()