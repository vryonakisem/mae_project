"""
Configuration settings for the MAE (Marketing Analyst Expert) application.
Contains paths, constants, and settings used throughout the application.

This file defines all configuration variables used across the application,
including database paths, API keys, and various constants.
"""

import os
from pathlib import Path

# Application information
APP_TITLE = "MAE - Marketing Analyst Expert"
COMPANY_NAME = "London Real Estate"

# Paths and directories
DATABASE_PATH = Path(__file__).parent / 'data' / 'marketing_db.sqlite'
UPLOAD_DIR = Path(__file__).parent / 'uploads'

# Date format for consistency across the application
DATE_FORMAT = '%Y-%m-%d'

# OpenAI API configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
OPENAI_MODEL = 'gpt-4'  # or another model like 'gpt-3.5-turbo'

# Anomaly detection settings
DEFAULT_ANOMALY_THRESHOLD = 0.25  # 25% change threshold for anomaly detection

# Metrics for comparison and analysis
COMPARISON_METRICS = [
    'spend', 
    'roi', 
    'instructions', 
    'leads', 
    'cost_per_instruction', 
    'cost_per_lead', 
    'conversion_rate'
]

# Standard marketing channels
MARKETING_CHANNELS = [
    'Direct Mail',
    'Aggregators',
    'Website',
    'Social Media',
    'Email',
    'Google Ads',
    'Print Media',
    'Outdoor',
    'TV/Radio',
    'Events'
]

# Company divisions
DIVISIONS = [
    'Sales',
    'Lettings',
    'New Homes',
    'Commercial',
    'Property Management'
]

# Visualization settings
THEME_COLOR = "#1f77b4"  # Default theme color for visualizations
COLOR_SEQUENCE = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf"   # Teal
]