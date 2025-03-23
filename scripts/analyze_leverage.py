#!/usr/bin/env python
"""
Leverage Analysis Tool for Trading System

This script analyzes leverage patterns from training logs and evaluation results,
providing visualizations and statistics to help understand leverage utilization.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
import json
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze leverage metrics from training runs')
    parser.add_argument('--log-dir', type=str, default='logs/incremental',
                        help='Directory containing training logs')
    parser.add_argument('--output-dir', type=str, default='analysis/leverage',
                        help='Directory to save analysis results')
    parser.add_argument('--phases', type=int, nargs='+', 
                        help='Specific training phases to analyze')
    parser.add_argument('--plot-style', type=str, default='dark_background',
                        choices=['default', 'dark_background', 'ggplot', 'seaborn'],
                        help='Matplotlib style for plots')
    return parser.parse_args()

def extract_leverage_from_logs(log_file):
    """Extract leverage metrics from a log file"""
    leverage_data = []
    step_pattern = re.compile(r'Step (\d+)')
    leverage_pattern = re.compile(r'\[Leverage Monitor\] Step (\d+): ([\d\.]+)x')
    gross_pattern = re.compile(r'Gross leverage: ([\d\.]+)x')
    net_pattern = re.compile(r'Net leverage: ([\d\.]+)x')
    
    current_step = None
    current_record = {}
    
    with open(log_file, 'r') as f:
        for line in f:
            # Check for leverage monitor entries
            leverage_match = leverage_pattern.search(line)
            if leverage_match:
                if current_record and current_step is not None:
                    leverage_data.append(current_record)
                
                current_step = int(leverage_match.group(1))
                current_record = {
                    'step': current_step,
                    'leverage': float(leverage_match.group(2)),
                    'timestamp': None
                }
                
                # Try to extract timestamp if present
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if timestamp_match:
                    current_record['timestamp'] = timestamp_match.group(1)
                
                continue
            
            # Check for gross leverage
            if current_record and current_step is not None:
                gross_match = gross_pattern.search(line)
                if gross_match:
                    current_record['gross_leverage'] = float(gross_match.group(1))
                    continue
                
                # Check for net leverage
                net_match = net_pattern.search(line)
                if net_match:
                    current_record['net_leverage'] = float(net_match.group(1))
                    # Add the completed record to the data
                    leverage_data.append(current_record)
                    current_record = {}
                    current_step = None
    
    # Add the last record if it exists
    if current_record and current_step is not None:
        leverage_data.append(current_record)
    
    return pd.DataFrame(leverage_data)

def load_evaluation_results(eval_dir):
    """Load leverage metrics from evaluation results"""
    result_files = glob.glob(os.path.join(eval_dir, "eval_*.csv"))
    
    if not result_files:
        return None
    
    # Use the latest result file
    latest_file = max(result_files, key=os.path.getctime)
    eval_df = pd.read_csv(latest_file)
    
    # Check if leverage columns exist
    if 'gross_leverage' not in eval_df.columns or 'net_leverage' not in eval_df.columns:
        return None
    
    return eval_df

def plot_leverage_over_time(df, output_path, title="Leverage Over Time"):
    """Plot leverage metrics over time"""
    plt.figure(figsize=(12, 7))
    
    if 'timestamp' in df.columns and df['timestamp'].notna().any():
        df['datetime'] = pd.to_datetime(df['timestamp'])
        x_column = 'datetime'
        plt.xlabel('Time')
    else:
        x_column = 'step'
        plt.xlabel('Training Step')
    
    # Plot overall leverage
    if 'leverage' in df.columns:
        plt.plot(df[x_column], df['leverage'], 
                 label='Overall Leverage', 
                 color='yellow', linewidth=2)
    
    # Plot gross leverage
    if 'gross_leverage' in df.columns:
        plt.plot(df[x_column], df['gross_leverage'], 
                 label='Gross Leverage', 
                 color='red', linestyle='-')
    
    # Plot net leverage
    if 'net_leverage' in df.columns:
        plt.plot(df[x_column], df['net_leverage'], 
                 label='Net Leverage', 
                 color='green', linestyle='-')
    
    plt.title(title)
    plt.ylabel('Leverage (x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add horizontal lines for reference
    plt.axhline(y=1.0, color='white', linestyle='--', alpha=0.5)
    plt.axhline(y=5.0, color='orange', linestyle='--', alpha=0.5)
    plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_leverage_distribution(df, output_path, title="Leverage Distribution"):
    """Plot the distribution of leverage values"""
    plt.figure(figsize=(12, 7))
    
    # Create a multi-panel plot for distributions
    if 'leverage' in df.columns and 'gross_leverage' in df.columns and 'net_leverage' in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Overall leverage
        sns.histplot(df['leverage'], kde=True, ax=axes[0], color='yellow')
        axes[0].set_title('Overall Leverage')
        axes[0].set_xlabel('Leverage (x)')
        
        # Gross leverage
        sns.histplot(df['gross_leverage'], kde=True, ax=axes[1], color='red')
        axes[1].set_title('Gross Leverage')
        axes[1].set_xlabel('Leverage (x)')
        
        # Net leverage
        sns.histplot(df['net_leverage'], kde=True, ax=axes[2], color='green')
        axes[2].set_title('Net Leverage')
        axes[2].set_xlabel('Leverage (x)')
        
    else:
        # Single plot for whatever leverage metrics are available
        for col, color in [('leverage', 'yellow'), 
                          ('gross_leverage', 'red'), 
                          ('net_leverage', 'green')]:
            if col in df.columns:
                sns.histplot(df[col], kde=True, color=color, label=col.replace('_', ' ').title())
        
        plt.xlabel('Leverage (x)')
        plt.ylabel('Frequency')
        plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_leverage_by_phase(log_dir, output_dir, phases=None, plot_style='dark_background'):
    """Analyze leverage metrics across different training phases"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use(plot_style)
    
    # Determine which phases to analyze
    available_phases = []
    for dir_name in os.listdir(log_dir):
        if dir_name.startswith('phase'):
            try:
                phase_num = int(dir_name[5:])
                available_phases.append(phase_num)
            except ValueError:
                continue
    
    available_phases.sort()
    
    if phases:
        phases_to_analyze = [p for p in phases if p in available_phases]
    else:
        phases_to_analyze = available_phases
    
    if not phases_to_analyze:
        print(f"No valid phases found to analyze in {log_dir}")
        return
    
    print(f"Analyzing phases: {phases_to_analyze}")
    
    # Prepare summary data
    summary_data = []
    all_leverage_data = pd.DataFrame()
    
    # Process each phase
    for phase in phases_to_analyze:
        phase_dir = os.path.join(log_dir, f"phase{phase}")
        eval_dir = os.path.join(log_dir, f"eval_phase{phase}")
        
        phase_summary = {'phase': phase}
        
        # Check for training logs
        log_files = glob.glob(os.path.join(phase_dir, "*.log"))
        if log_files:
            latest_log = max(log_files, key=os.path.getctime)
            print(f"Processing log file: {latest_log}")
            
            # Extract leverage data from logs
            leverage_df = extract_leverage_from_logs(latest_log)
            
            if not leverage_df.empty:
                # Add phase column
                leverage_df['phase'] = phase
                
                # Add to all leverage data
                all_leverage_data = pd.concat([all_leverage_data, leverage_df])
                
                # Calculate summary statistics
                phase_summary.update({
                    'avg_leverage': leverage_df['leverage'].mean() if 'leverage' in leverage_df else None,
                    'max_leverage': leverage_df['leverage'].max() if 'leverage' in leverage_df else None,
                    'min_leverage': leverage_df['leverage'].min() if 'leverage' in leverage_df else None,
                    'std_leverage': leverage_df['leverage'].std() if 'leverage' in leverage_df else None,
                })
                
                # Create phase-specific plots
                phase_output_dir = os.path.join(output_dir, f"phase{phase}")
                os.makedirs(phase_output_dir, exist_ok=True)
                
                # Time series plot
                plot_leverage_over_time(
                    leverage_df, 
                    os.path.join(phase_output_dir, 'leverage_time_series.png'),
                    title=f"Phase {phase} - Leverage Over Time"
                )
                
                # Distribution plot
                plot_leverage_distribution(
                    leverage_df,
                    os.path.join(phase_output_dir, 'leverage_distribution.png'),
                    title=f"Phase {phase} - Leverage Distribution"
                )
        
        # Check for evaluation results
        if os.path.exists(eval_dir):
            eval_df = load_evaluation_results(eval_dir)
            if eval_df is not None:
                phase_summary.update({
                    'eval_avg_gross_leverage': eval_df['gross_leverage'].mean(),
                    'eval_avg_net_leverage': eval_df['net_leverage'].mean(),
                    'eval_max_gross_leverage': eval_df['gross_leverage'].max(),
                    'eval_min_net_leverage': eval_df['net_leverage'].min(),
                })
        
        # Add to summary data
        summary_data.append(phase_summary)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'leverage_summary.csv'), index=False)
    
    # Create combined plots if we have data from multiple phases
    if len(phases_to_analyze) > 1 and not all_leverage_data.empty:
        # Plot leverage by phase
        plt.figure(figsize=(12, 7))
        sns.boxplot(x='phase', y='leverage', data=all_leverage_data)
        plt.title('Leverage Distribution by Training Phase')
        plt.xlabel('Training Phase')
        plt.ylabel('Leverage (x)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'leverage_by_phase.png'))
        plt.close()
        
        # Create a heatmap of summary statistics
        if not summary_df.empty:
            plt.figure(figsize=(14, 8))
            summary_for_heatmap = summary_df.set_index('phase')
            
            # Select only numeric columns
            numeric_cols = summary_for_heatmap.select_dtypes(include=[np.number]).columns
            
            # Create heatmap
            sns.heatmap(summary_for_heatmap[numeric_cols], annot=True, cmap='viridis', 
                        linewidths=0.5, fmt=".2f")
            plt.title('Leverage Metrics Across Training Phases')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'leverage_metrics_heatmap.png'))
            plt.close()
    
    print(f"Analysis complete. Results saved to {output_dir}")
    
    return summary_df

if __name__ == "__main__":
    args = parse_args()
    analyze_leverage_by_phase(args.log_dir, args.output_dir, args.phases, args.plot_style) 