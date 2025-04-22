import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import os
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def format_currency(value: Union[float, int]) -> str:
    """
    Format a value as GBP currency.
    
    Args:
        value: Numeric value to format
        
    Returns:
        Formatted currency string
    """
    return f"£{value:,.2f}"

def format_percent(value: float) -> str:
    """
    Format a value as a percentage.
    
    Args:
        value: Numeric value to format (0.1 = 10%)
        
    Returns:
        Formatted percentage string
    """
    return f"{value:.1%}"

def format_int(value: Union[float, int]) -> str:
    """
    Format a value as an integer.
    
    Args:
        value: Numeric value to format
        
    Returns:
        Formatted integer string
    """
    return f"{int(value):,}"

def calculate_date_ranges() -> Dict[str, Tuple[str, str]]:
    """
    Calculate common date ranges for filtering.
    
    Returns:
        Dictionary with period names as keys and (start_date, end_date) tuples as values
    """
    today = datetime.now()
    
    # Last 7 days
    last_7_days_end = today
    last_7_days_start = today - timedelta(days=7)
    
    # Last 30 days
    last_30_days_end = today
    last_30_days_start = today - timedelta(days=30)
    
    # Last 90 days
    last_90_days_end = today
    last_90_days_start = today - timedelta(days=90)

    # Current month
    current_month_start = today.replace(day=1)
    current_month_end = today
    
    # Previous month
    if today.month == 1:
        previous_month_start = today.replace(year=today.year-1, month=12, day=1)
    else:
        previous_month_start = today.replace(month=today.month-1, day=1)
    
    previous_month_end = current_month_start - timedelta(days=1)
    
    # Current quarter
    current_quarter = (today.month - 1) // 3 + 1
    current_quarter_start = datetime(today.year, 3 * current_quarter - 2, 1)
    current_quarter_end = today
    
    # Previous quarter
    if current_quarter == 1:
        previous_quarter_start = datetime(today.year - 1, 10, 1)
        previous_quarter_end = datetime(today.year, 1, 1) - timedelta(days=1)
    else:
        previous_quarter_start = datetime(today.year, 3 * (current_quarter - 1) - 2, 1)
        previous_quarter_end = current_quarter_start - timedelta(days=1)
    
    # Format dates as strings
    date_format = '%Y-%m-%d'
    
    return {
        'last_7_days': (
            last_7_days_start.strftime(date_format),
            last_7_days_end.strftime(date_format)
        ),
        'last_30_days': (
            last_30_days_start.strftime(date_format),
            last_30_days_end.strftime(date_format)
        ),
        'last_90_days': (
            last_90_days_start.strftime(date_format),
            last_90_days_end.strftime(date_format)
        ),
        'current_month': (
            current_month_start.strftime(date_format),
            current_month_end.strftime(date_format)
        ),
        'previous_month': (
            previous_month_start.strftime(date_format),
            previous_month_end.strftime(date_format)
        ),
        'current_quarter': (
            current_quarter_start.strftime(date_format),
            current_quarter_end.strftime(date_format)
        ),
        'previous_quarter': (
            previous_quarter_start.strftime(date_format),
            previous_quarter_end.strftime(date_format)
        )
    }

def get_period_name(start_date: str, end_date: str) -> str:
    """
    Get a friendly name for a date period if it matches common ranges.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Friendly period name or custom range description
    """
    date_ranges = calculate_date_ranges()
    
    for period_name, (period_start, period_end) in date_ranges.items():
        if start_date == period_start and end_date == period_end:
            # Convert to friendly name
            return period_name.replace('_', ' ').title()
    
    # Custom range
    return f"Custom Range: {start_date} to {end_date}"

def detect_excel_format(df: pd.DataFrame) -> str:
    """
    Detect the format of an Excel file based on its columns.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Format description string
    """
    columns = set(df.columns.str.lower())
    
    # Check for standard format (office, channel, spend, instructions, leads, roi)
    standard_cols = {'office', 'channel', 'spend', 'instructions', 'leads', 'roi'}
    if standard_cols.issubset(columns):
        return "Standard Format"
    
    # Check for office performance format
    office_cols = {'office', 'division', 'instructions', 'leads'}
    if office_cols.issubset(columns) and 'channel' not in columns:
        return "Office Performance Format"
    
    # Check for channel performance format
    channel_cols = {'channel', 'spend', 'roi'}
    if channel_cols.issubset(columns) and 'office' not in columns:
        return "Channel Performance Format"
    
    # Check for time series format
    if 'date' in columns:
        return "Time Series Format"
    
    # Default
    return "Unknown Format"

def safe_divide(numerator: Union[float, int], 
              denominator: Union[float, int]) -> float:
    """
    Safely divide two numbers, returning 0 for division by zero.
    
    Args:
        numerator: The number to divide
        denominator: The number to divide by
        
    Returns:
        The result of division, or 0 if denominator is 0
    """
    if denominator == 0:
        return 0
    return numerator / denominator

