import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import io
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Import local modules
from config import (
    APP_TITLE, 
    COMPANY_NAME, 
    THEME_COLOR, 
    OPENAI_API_KEY,
    COMPARISON_METRICS,
    UPLOAD_DIR
)
from database.db_manager import DatabaseManager
from models.data_processor import DataProcessor
from models.anomaly_detector import AnomalyDetector
from models.ai_analyst import AIAnalyst
from utils.visualization import Visualizer
from utils.helpers import (
    format_currency, 
    format_percent, 
    format_int,
    calculate_date_ranges,
    get_period_name
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize database and modules
db_manager = DatabaseManager()
data_processor = DataProcessor(db_manager)
anomaly_detector = AnomalyDetector(db_manager, data_processor)
ai_analyst = AIAnalyst(db_manager)
visualizer = Visualizer(THEME_COLOR)

# Set page config
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title(APP_TITLE)
st.markdown(f"*Your intelligent marketing analyst for {COMPANY_NAME}*")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Data Upload", "Analysis", "Insights", "Reports", "Chat with MAE"]
)

# Check if database is empty
def is_database_empty() -> bool:
    """Check if the database has any marketing data."""
    upload_history = db_manager.get_upload_history(limit=1)
    return len(upload_history) == 0

# Date filter in sidebar (only for pages that need it)
if page in ["Dashboard", "Analysis", "Insights", "Reports"]:
    st.sidebar.header("Date Range")
    
    # Get available date ranges
    date_ranges = calculate_date_ranges()
    
    # Default to last 30 days
    default_range = "last_30_days"
    
    # Date range selector
    selected_range = st.sidebar.selectbox(
        "Select time period",
        list(date_ranges.keys()),
        format_func=lambda x: x.replace('_', ' ').title(),
        index=list(date_ranges.keys()).index(default_range)
    )
    
    # Get start and end dates from selection
    start_date, end_date = date_ranges[selected_range]
    
    # Option for custom date range
    use_custom_range = st.sidebar.checkbox("Use custom date range")
    
    if use_custom_range:
        custom_start_date = st.sidebar.date_input(
            "Start date",
            datetime.strptime(start_date, '%Y-%m-%d')
        )
        custom_end_date = st.sidebar.date_input(
            "End date",
            datetime.strptime(end_date, '%Y-%m-%d')
        )
        
        # Update start and end dates
        start_date = custom_start_date.strftime('%Y-%m-%d')
        end_date = custom_end_date.strftime('%Y-%m-%d')

# Additional filters in sidebar
if page in ["Dashboard", "Analysis"]:
    st.sidebar.header("Filters")
    
    # Get available options from database
    all_offices = db_manager.get_all_offices()
    all_channels = db_manager.get_all_channels()
    all_divisions = db_manager.get_all_divisions()
    
    # Only show filters if we have data
    if all_offices or all_channels or all_divisions:
        # Office filter
        if all_offices:
            selected_office = st.sidebar.selectbox(
                "Office",
                ["All Offices"] + all_offices
            )
            office_filter = selected_office if selected_office != "All Offices" else None
        else:
            office_filter = None
        
        # Channel filter
        if all_channels:
            selected_channel = st.sidebar.selectbox(
                "Marketing Channel",
                ["All Channels"] + all_channels
            )
            channel_filter = selected_channel if selected_channel != "All Channels" else None
        else:
            channel_filter = None
        
        # Division filter
        if all_divisions:
            selected_division = st.sidebar.selectbox(
                "Division",
                ["All Divisions"] + all_divisions
            )
            division_filter = selected_division if selected_division != "All Divisions" else None
        else:
            division_filter = None
    else:
        office_filter = None
        channel_filter = None
        division_filter = None

# Function to load filtered data
def load_filtered_data():
    """Load data based on current filters."""
    return db_manager.get_marketing_data(
        start_date=start_date,
        end_date=end_date,
        office=office_filter,
        division=division_filter,
        channel=channel_filter
    )

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to handle empty database state
def show_empty_state():
    st.info("Welcome to MAE! Please upload your marketing data to get started.")
    st.markdown("""
    MAE works with Excel exports from Power BI containing your marketing performance data.
    
    Your data should include these columns:
    - **date**: The date of the marketing activity
    - **office**: The office name
    - **division**: The division (e.g., Residential Sales)
    - **channel**: The marketing channel (Direct Mail, Aggregators, Website)
    - **spend**: Marketing spend in GBP
    - **roi**: Return on investment
    - **instructions**: Number of instructions received
    - **leads**: Number of leads generated
    
    Go to the **Data Upload** page to import your first dataset.
    """)

# Dashboard page
if page == "Dashboard":
    st.header("Marketing Performance Dashboard")
    
    if is_database_empty():
        show_empty_state()
    else:
        # Load filtered data
        df = load_filtered_data()
        
        if df.empty:
            st.warning(f"No data available for the selected period: {start_date} to {end_date}")
        else:
            # Show overview metrics
            st.subheader("Key Metrics")
            
            total_spend = df['spend'].sum() if 'spend' in df.columns else 0
            total_instructions = df['instructions'].sum() if 'instructions' in df.columns else 0
            total_leads = df['leads'].sum() if 'leads' in df.columns else 0
            avg_roi = df['roi'].mean() if 'roi' in df.columns else 0
            
            # Create metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Spend",
                    value=format_currency(total_spend)
                )
            
            with col2:
                st.metric(
                    label="Total Instructions",
                    value=format_int(total_instructions)
                )
            
            with col3:
                st.metric(
                    label="Total Leads",
                    value=format_int(total_leads)
                )
            
            with col4:
                st.metric(
                    label="Average ROI",
                    value=format_percent(avg_roi)
                )
            
            # Calculate derived metrics
            if 'spend' in df.columns and 'instructions' in df.columns and total_instructions > 0:
                cost_per_instruction = total_spend / total_instructions
            else:
                cost_per_instruction = 0
            
            if 'spend' in df.columns and 'leads' in df.columns and total_leads > 0:
                cost_per_lead = total_spend / total_leads
            else:
                cost_per_lead = 0
            
            if 'leads' in df.columns and 'instructions' in df.columns and total_leads > 0:
                conversion_rate = total_instructions / total_leads
            else:
                conversion_rate = 0
            
            # Create second metrics row
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Cost per Instruction",
                    value=format_currency(cost_per_instruction)
                )
            
            with col2:
                st.metric(
                    label="Cost per Lead",
                    value=format_currency(cost_per_lead)
                )
            
            with col3:
                st.metric(
                    label="Conversion Rate",
                    value=format_percent(conversion_rate)
                )
            
            # Create charts
            st.markdown("---")
            st.subheader("Performance by Channel")
            
            channel_chart = visualizer.create_channel_performance_chart(df)
            st.plotly_chart(channel_chart, use_container_width=True)
            
            # Time series chart
            st.markdown("---")
            st.subheader("Trends Over Time")
            
            metric_options = [m.title() for m in COMPARISON_METRICS if m.lower() in df.columns]
            selected_metrics = st.multiselect(
                "Select metrics to display",
                metric_options,
                default=metric_options[:2]  # Default to first two metrics
            )
            
            if selected_metrics:
                metrics_to_plot = [m.lower() for m in selected_metrics]
                time_chart = visualizer.create_time_series_chart(df, metrics=metrics_to_plot)
                st.plotly_chart(time_chart, use_container_width=True)
            
            # Office performance
            if 'office' in df.columns:
                st.markdown("---")
                st.subheader("Office Performance")
                
                metric_for_office = st.selectbox(
                    "Select metric for office comparison",
                    metric_options,
                    index=1 if len(metric_options) > 1 else 0
                )
                
                office_chart = visualizer.create_office_performance_chart(
                    df, 
                    metric=metric_for_office.lower(),
                    top_n=10
                )
                st.plotly_chart(office_chart, use_container_width=True)
            
            # Recent insights
            st.markdown("---")
            st.subheader("Recent Insights")
            
            recent_insights = db_manager.get_insights(
                start_date=start_date,
                end_date=end_date,
                limit=5
            )
            
            if recent_insights:
                for insight in recent_insights:
                    with st.expander(insight['description']):
                        st.write(f"**Type:** {insight['type'].title()}")
                        st.write(f"**Severity:** {insight['severity'].title() if insight['severity'] else 'Medium'}")
                        
                        # Show affected dimensions
                        dimensions = []
                        if insight.get('office'):
                            dimensions.append(f"Office: {insight['office']}")
                        if insight.get('channel'):
                            dimensions.append(f"Channel: {insight['channel']}")
                        if insight.get('division'):
                            dimensions.append(f"Division: {insight['division']}")
                        
                        if dimensions:
                            st.write("**Affected:** " + ", ".join(dimensions))
                        
                        if insight.get('recommendation'):
                            st.markdown(f"**Recommendation:** {insight['recommendation']}")
            else:
                st.info("No insights available for the selected period.")

# Data Upload page
elif page == "Data Upload":
    st.header("Upload Marketing Data")
    
    st.markdown("""
    Upload your Excel exports from Power BI containing marketing performance data.
    
    The data should include: date, office, division, channel, spend, ROI, instructions, leads.
    """)
    
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        file_path = Path(UPLOAD_DIR) / uploaded_file.name
        
        # Create upload directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the file
        with st.spinner("Processing data..."):
            try:
                df = data_processor.process_excel(file_path)
                
                # Show success message
                st.success(f"Successfully processed {len(df)} rows of data!")
                
                # Preview the data
                st.subheader("Data Preview")
                st.dataframe(df.head(10))
                
                # Summary of the data
                st.subheader("Data Summary")
                
                # Get basic stats
                if 'date' in df.columns:
                    min_date = df['date'].min()
                    max_date = df['date'].max()
                    st.write(f"Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                
                # Count of records by key dimensions
                if 'office' in df.columns:
                    st.write(f"Number of offices: {df['office'].nunique()}")
                
                if 'channel' in df.columns:
                    st.write(f"Marketing channels: {', '.join(df['channel'].unique())}")
                
                # Generate insights from the data
                st.subheader("Initial Insights")
                
                # Run anomaly detection on the uploaded data
                with st.spinner("Generating insights..."):
                    anomalies = anomaly_detector.detect_performance_anomalies(df)
                    
                    if anomalies:
                        st.write(f"Found {len(anomalies)} potential insights in your data:")
                        
                        for anomaly in anomalies[:5]:  # Show top 5
                            st.markdown(f"- **{anomaly['description']}**")
                            if 'recommendation' in anomaly:
                                st.markdown(f"  *Recommendation: {anomaly['recommendation']}*")
                    else:
                        st.write("No significant anomalies detected in the uploaded data.")
                
                # Option to run AI analysis
                if OPENAI_API_KEY:
                    st.subheader("AI Analysis")
                    if st.button("Generate AI Analysis"):
                        with st.spinner("Generating AI analysis..."):
                            ai_summary = ai_analyst.generate_data_summary(df)
                            st.markdown(ai_summary)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                logger.error(f"Error processing uploaded file: {str(e)}")
    
    # Show upload history
    st.markdown("---")
    st.subheader("Upload History")
    
    upload_history = db_manager.get_upload_history()
    
    if upload_history:
        history_df = pd.DataFrame(upload_history)
        history_df = history_df[['filename', 'upload_date', 'data_start_date', 'data_end_date', 'record_count']]
        history_df.columns = ['Filename', 'Upload Date', 'Data Start', 'Data End', 'Records']
        st.dataframe(history_df)
    else:
        st.info("No previous uploads found.")

# Analysis page
elif page == "Analysis":
    st.header("Marketing Performance Analysis")
    
    if is_database_empty():
        show_empty_state()
    else:
        # Load filtered data
        df = load_filtered_data()
        
        if df.empty:
            st.warning(f"No data available for the selected period: {start_date} to {end_date}")
        else:
            # Analysis type selector
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Channel Performance", "Office Performance", "Time Trends", "Period Comparison", "Custom Analysis"]
            )
            
            if analysis_type == "Channel Performance":
                st.subheader("Channel Performance Analysis")
                
                # Charts for channel performance
                channel_metrics_chart = visualizer.create_channel_performance_chart(df)
                st.plotly_chart(channel_metrics_chart, use_container_width=True)
                
                # Channel comparison table
                if 'channel' in df.columns:
                    st.subheader("Channel Comparison")
                    
                    channel_data = df.groupby('channel').agg({
                        'spend': 'sum',
                        'instructions': 'sum',
                        'leads': 'sum',
                        'roi': 'mean'
                    }).reset_index()
                    
                    # Calculate derived metrics
                    channel_data['cost_per_instruction'] = channel_data.apply(
                        lambda row: row['spend'] / row['instructions'] if row['instructions'] > 0 else np.nan,
                        axis=1
                    )
                    
                    channel_data['cost_per_lead'] = channel_data.apply(
                        lambda row: row['spend'] / row['leads'] if row['leads'] > 0 else np.nan,
                        axis=1
                    )
                    
                    channel_data['conversion_rate'] = channel_data.apply(
                        lambda row: row['instructions'] / row['leads'] if row['leads'] > 0 else np.nan,
                        axis=1
                    )
                    
                    # Format the table
                    formatted_channel_data = channel_data.copy()
                    formatted_channel_data['spend'] = formatted_channel_data['spend'].apply(format_currency)
                    formatted_channel_data['roi'] = formatted_channel_data['roi'].apply(format_percent)
                    formatted_channel_data['cost_per_instruction'] = formatted_channel_data['cost_per_instruction'].apply(
                        lambda x: format_currency(x) if pd.notna(x) else 'N/A'
                    )
                    formatted_channel_data['cost_per_lead'] = formatted_channel_data['cost_per_lead'].apply(
                        lambda x: format_currency(x) if pd.notna(x) else 'N/A'
                    )
                    formatted_channel_data['conversion_rate'] = formatted_channel_data['conversion_rate'].apply(
                        lambda x: format_percent(x) if pd.notna(x) else 'N/A'
                    )
                    
                    # Rename columns for display
                    formatted_channel_data.columns = [
                        'Channel', 'Spend', 'Instructions', 'Leads', 'ROI', 
                        'Cost per Instruction', 'Cost per Lead', 'Conversion Rate'
                    ]
                    
                    st.dataframe(formatted_channel_data, use_container_width=True)
                    
                    # AI analysis of channel performance
                    if OPENAI_API_KEY:
                        st.subheader("AI Analysis")
                        if st.button("Generate Channel Analysis"):
                            with st.spinner("Analyzing channel performance..."):
                                # Prepare a prompt focused on channel performance
                                question = f"Analyze the performance of different marketing channels between {start_date} and {end_date}. Which channel is most effective and why?"
                                analysis = ai_analyst.answer_question(
                                    question=question,
                                    context_start_date=start_date,
                                    context_end_date=end_date
                                )
                                st.markdown(analysis)
            
            elif analysis_type == "Office Performance":
                st.subheader("Office Performance Analysis")
                
                if 'office' not in df.columns:
                    st.warning("Office data not available in the selected dataset.")
                else:
                    # Metric selector
                    metric_options = [m.title() for m in COMPARISON_METRICS if m.lower() in df.columns]
                    selected_metric = st.selectbox(
                        "Select metric for analysis",
                        metric_options,
                        index=1 if len(metric_options) > 1 else 0  # Default to instructions if available
                    )
                    
                    metric = selected_metric.lower()
                    
                    # Office performance chart
                    office_chart = visualizer.create_office_performance_chart(
                        df, 
                        metric=metric,
                        top_n=15
                    )
                    st.plotly_chart(office_chart, use_container_width=True)
                    
                    # Office heatmap if we have channel data too
                    if 'channel' in df.columns:
                        st.subheader(f"{selected_metric} by Office and Channel")
                        
                        heatmap = visualizer.create_heatmap(
                            df,
                            x_col='office',
                            y_col='channel',
                            value_col=metric
                        )
                        st.plotly_chart(heatmap, use_container_width=True)
                    
                    # Office comparison table
                    st.subheader("Office Comparison")
                    
                    office_data = df.groupby('office').agg({
                        'spend': 'sum',
                        'instructions': 'sum',
                        'leads': 'sum',
                        'roi': 'mean'
                    }).reset_index()
                    
                    # Calculate derived metrics
                    office_data['cost_per_instruction'] = office_data.apply(
                        lambda row: row['spend'] / row['instructions'] if row['instructions'] > 0 else np.nan,
                        axis=1
                    )
                    
                    office_data['cost_per_lead'] = office_data.apply(
                        lambda row: row['spend'] / row['leads'] if row['leads'] > 0 else np.nan,
                        axis=1
                    )
                    
                    # Sort by the selected metric
                    office_data = office_data.sort_values(metric, ascending=False)
                    
                    # Format the table
                    formatted_office_data = office_data.copy()
                    formatted_office_data['spend'] = formatted_office_data['spend'].apply(format_currency)
                    formatted_office_data['roi'] = formatted_office_data['roi'].apply(format_percent)
                    formatted_office_data['cost_per_instruction'] = formatted_office_data['cost_per_instruction'].apply(
                        lambda x: format_currency(x) if pd.notna(x) else 'N/A'
                    )
                    formatted_office_data['cost_per_lead'] = formatted_office_data['cost_per_lead'].apply(
                        lambda x: format_currency(x) if pd.notna(x) else 'N/A'
                    )
                    
                    # Rename columns for display
                    formatted_office_data.columns = [
                        'Office', 'Spend', 'Instructions', 'Leads', 'ROI', 
                        'Cost per Instruction', 'Cost per Lead'
                    ]
                    
                    st.dataframe(formatted_office_data, use_container_width=True)
                    
                    # AI analysis of office performance
                    if OPENAI_API_KEY:
                        st.subheader("AI Analysis")
                        if st.button("Generate Office Analysis"):
                            with st.spinner("Analyzing office performance..."):
                                # Prepare a prompt focused on office performance
                                question = f"Analyze the performance of different offices between {start_date} and {end_date}. Which offices are performing best and worst? What patterns do you see?"
                                analysis = ai_analyst.answer_question(
                                    question=question,
                                    context_start_date=start_date,
                                    context_end_date=end_date
                                )
                                st.markdown(analysis)
            
            elif analysis_type == "Time Trends":
                st.subheader("Time Trend Analysis")
                
                if 'date' not in df.columns:
                    st.warning("Date information not available in the selected dataset.")
                else:
                    # Group by options
                    group_options = []
                    if 'channel' in df.columns:
                        group_options.append('channel')
                    if 'office' in df.columns:
                        group_options.append('office')
                    if 'division' in df.columns:
                        group_options.append('division')
                    
                    group_by = None
                    if group_options:
                        group_by = st.selectbox(
                            "Group by",
                            ["None"] + group_options
                        )
                        
                        if group_by == "None":
                            group_by = None
                    
                    # Metric selector
                    metric_options = [m.title() for m in COMPARISON_METRICS if m.lower() in df.columns]
                    selected_metrics = st.multiselect(
                        "Select metrics to display",
                        metric_options,
                        default=metric_options[:2]  # Default to first two metrics
                    )
                    
                    if selected_metrics:
                        metrics_to_plot = [m.lower() for m in selected_metrics]
                        
                        # Time aggregation
                        time_agg = st.selectbox(
                            "Time aggregation",
                            ["Daily", "Weekly", "Monthly"]
                        )
                        
                        # Aggregate data by time
                        time_period = time_agg.lower()
                        agg_df = data_processor.get_aggregated_data(
                            df=df,
                            time_period=time_period,
                            group_by=[group_by] if group_by else []
                        )
                        
                        if not agg_df.empty:
                            # Create time series chart
                            time_chart = visualizer.create_time_series_chart(
                                agg_df, 
                                metrics=metrics_to_plot,
                                group_by=group_by
                            )
                            st.plotly_chart(time_chart, use_container_width=True)
                        else:
                            st.warning("Not enough data for the selected time aggregation.")
                    
                    # AI analysis of time trends
                    if OPENAI_API_KEY:
                        st.subheader("AI Analysis")
                        if st.button("Generate Trend Analysis"):
                            with st.spinner("Analyzing time trends..."):
                                # Prepare a prompt focused on time trends
                                question = f"Analyze the trends in marketing performance between {start_date} and {end_date}. What patterns do you see over time? Are there any seasonal factors or notable changes?"
                                analysis = ai_analyst.answer_question(
                                    question=question,
                                    context_start_date=start_date,
                                    context_end_date=end_date
                                )
                                st.markdown(analysis)
            
            elif analysis_type == "Period Comparison":
                st.subheader("Period Comparison Analysis")
                
                # Get current period details
                current_period_name = get_period_name(start_date, end_date)
                st.write(f"Current period: {current_period_name} ({start_date} to {end_date})")
                
                # Select comparison period
                date_ranges = calculate_date_ranges()
                comparison_options = list(date_ranges.keys())
                
                # Remove current period from options
                if selected_range in comparison_options:
                    comparison_options.remove(selected_range)
                
                if comparison_options:
                    comparison_range = st.selectbox(
                        "Compare with",
                        comparison_options,
                        format_func=lambda x: x.replace('_', ' ').title()
                    )
                    
                    # Get comparison period dates
                    prev_start_date, prev_end_date = date_ranges[comparison_range]
                    
                    st.write(f"Comparison period: {comparison_range.replace('_', ' ').title()} ({prev_start_date} to {prev_end_date})")
                    
                    # Option to use custom comparison period
                    use_custom_comparison = st.checkbox("Use custom comparison period")
                    
                    if use_custom_comparison:
                        custom_prev_start = st.date_input(
                            "Comparison start date",
                            datetime.strptime(prev_start_date, '%Y-%m-%d')
                        )
                        custom_prev_end = st.date_input(
                            "Comparison end date",
                            datetime.strptime(prev_end_date, '%Y-%m-%d')
                        )
                        
                        # Update comparison dates
                        prev_start_date = custom_prev_start.strftime('%Y-%m-%d')
                        prev_end_date = custom_prev_end.strftime('%Y-%m-%d')
                    
                    # Select comparison dimensions
                    comparison_dimensions = []
                    if 'channel' in df.columns:
                        if st.checkbox("Compare by channel", value=True):
                            comparison_dimensions.append('channel')
                    
                    if 'office' in df.columns:
                        if st.checkbox("Compare by office"):
                            comparison_dimensions.append('office')
                    
                    # Select metric to compare
                    metric_options = [m.title() for m in COMPARISON_METRICS if m.lower() in df.columns]
                    selected_metric = st.selectbox(
                        "Select metric for comparison",
                        metric_options,
                        index=1 if len(metric_options) > 1 else 0  # Default to instructions if available
                    )
                    
                    metric = selected_metric.lower()
                    
                    # Generate comparison
                    if st.button("Generate Comparison"):
                        with st.spinner("Generating period comparison..."):
                            # Get comparison data
                            comparison_df = data_processor.compare_periods(
                                current_period_start=start_date,
                                current_period_end=end_date,
                                previous_period_start=prev_start_date,
                                previous_period_end=prev_end_date,
                                comparison_dimensions=comparison_dimensions
                            )
                            
                            if comparison_df.empty:
                                st.warning("No data available for comparison.")
                            else:
                                # Create comparison chart
                                comparison_chart = visualizer.create_comparison_chart(
                                    comparison_df,
                                    metric=metric,
                                    comparison_col=comparison_dimensions[0] if comparison_dimensions else None
                                )
                                st.plotly_chart(comparison_chart, use_container_width=True)
                                
                                # Show comparison table
                                st.subheader("Detailed Comparison")
                                
                                # Select columns for display
                                display_cols = comparison_dimensions.copy()
                                
                                metric_cols = [
                                    f"{metric}_current",
                                    f"{metric}_previous",
                                    f"{metric}_change",
                                    f"{metric}_pct_change"
                                ]
                                
                                # Format the comparison data for display
                                display_df = comparison_df[display_cols + metric_cols].copy()
                                
                                # Format numeric columns
                                for col in metric_cols:
                                    if col in display_df.columns:
                                        if 'spend' in col and 'pct' not in col:
                                            display_df[col] = display_df[col].apply(
                                                lambda x: format_currency(x) if pd.notna(x) else 'N/A'
                                            )
                                        elif 'roi' in col and 'pct' not in col:
                                            display_df[col] = display_df[col].apply(
                                                lambda x: format_percent(x) if pd.notna(x) else 'N/A'
                                            )
                                        elif 'pct_change' in col:
                                            display_df[col] = display_df[col].apply(
                                                lambda x: f"{x:+.1f}%" if pd.notna(x) and x != float('inf') else 'N/A'
                                            )
                                        elif 'change' in col:
                                            if 'spend' in col:
                                                display_df[col] = display_df[col].apply(
                                                    lambda x: f"{format_currency(x)}" if pd.notna(x) else 'N/A'
                                            )
                                        else:
                                              display_df[col] = display_df[col].apply(
                                                    lambda x: f"{x:+,.0f}" if pd.notna(x) else 'N/A'
                                            )

                                    
                                
                                # Rename columns for display
                                rename_dict = {}
                                for col in display_df.columns:
                                    if col in comparison_dimensions:
                                        rename_dict[col] = col.title()
                                    elif col == f"{metric}_current":
                                        rename_dict[col] = f"Current {metric.title()}"
                                    elif col == f"{metric}_previous":
                                        rename_dict[col] = f"Previous {metric.title()}"
                                    elif col == f"{metric}_change":
                                        rename_dict[col] = f"Absolute Change"
                                    elif col == f"{metric}_pct_change":
                                        rename_dict[col] = f"% Change"
                                
                                display_df = display_df.rename(columns=rename_dict)
                                
                                st.dataframe(display_df, use_container_width=True)
                                
                                # AI analysis of comparison
                                if OPENAI_API_KEY:
                                    st.subheader("AI Analysis")
                                    if st.button("Analyze Comparison"):
                                        with st.spinner("Analyzing period comparison..."):
                                            # Prepare a prompt focused on period comparison
                                            dimensions_text = " and ".join([d.title() for d in comparison_dimensions])
                                            question = f"Compare marketing performance between {start_date} to {end_date} and {prev_start_date} to {prev_end_date}. Focus on {selected_metric} by {dimensions_text}. What are the most significant changes and potential reasons?"
                                            analysis = ai_analyst.answer_question(
                                                question=question,
                                                context_start_date=min(start_date, prev_start_date),
                                                context_end_date=max(end_date, prev_end_date)
                                            )
                                            st.markdown(analysis)
                else:
                    st.warning("No comparison periods available.")
            
            elif analysis_type == "Custom Analysis":
                st.subheader("Custom Analysis")
                
                st.markdown("""
                Enter your specific analysis question, and MAE will provide insights based on your data.
                
                Examples:
                - Which marketing channel has the best ROI?
                - How has our spend efficiency changed over the last month?
                - Which offices have the highest cost per instruction?
                - Compare website performance across different divisions.
                """)
                
                # Custom analysis question
                analysis_question = st.text_area(
                    "What would you like to analyze?",
                    height=100,
                    placeholder="Enter your analysis question here..."
                )
                
                if analysis_question and OPENAI_API_KEY:
                    if st.button("Analyze"):
                        with st.spinner("Analyzing your question..."):
                            analysis = ai_analyst.answer_question(
                                question=analysis_question,
                                context_start_date=start_date,
                                context_end_date=end_date
                            )
                            st.markdown(analysis)
                elif analysis_question and not OPENAI_API_KEY:
                    st.warning("To use custom analysis, please configure your OpenAI API key.")

# Insights page
elif page == "Insights":
    st.header("Marketing Insights")
    
    if is_database_empty():
        show_empty_state()
    else:
        # Options to generate new insights
        st.subheader("Generate New Insights")
        
        col1, col2 = st.columns(2)

        # Make sure filters are defined
        if 'office_filter' not in locals():
            office_filter = None
        if 'channel_filter' not in locals():
            channel_filter = None
        if 'division_filter' not in locals():
            division_filter = None

        with col1:
            if st.button("Run Anomaly Detection"):
                with st.spinner("Detecting anomalies..."):
                    # Get filtered data
                    df = load_filtered_data()
                    
                    if df.empty:
                        st.warning(f"No data available for the period: {start_date} to {end_date}")
                    else:
                        # Run anomaly detection
                        anomalies = anomaly_detector.detect_performance_anomalies(df)
                        
                        if anomalies:
                            st.success(f"Found {len(anomalies)} potential anomalies!")
                            
                            # Store anomalies as insights
                            for anomaly in anomalies:
                                # Format the anomaly as an insight
                                insight_data = {
                                    'date_generated': datetime.now().strftime('%Y-%m-%d'),
                                    'type': anomaly['type'],
                                    'description': anomaly['description'],
                                    'metric': anomaly.get('metric'),
                                    'office': anomaly.get('office'),
                                    'channel': anomaly.get('channel'),
                                    'division': anomaly.get('division'),
                                    'recommendation': anomaly.get('recommendation'),
                                    'severity': anomaly.get('severity', 'medium'),
                                    'status': 'new',
                                    'period_start': start_date,
                                    'period_end': end_date
                                }
                                
                                # Store the insight
                                db_manager.store_insight(insight_data)
                        else:
                            st.info("No anomalies detected in the current data.")
        with col2:
            if st.button("Run Trend Analysis"):
                with st.spinner("Analyzing trends..."):
            # Make sure filters are defined
                    if 'office_filter' not in locals():
                        office_filter = None
                    if 'channel_filter' not in locals():
                        channel_filter = None
                    if 'division_filter' not in locals():
                        division_filter = None
    
                    # Run trend detection
                    trend_anomalies = anomaly_detector.detect_trend_changes(
                        start_date=start_date,
                        end_date=end_date,
                        group_by=['channel', 'office']
                    )
                    
                    if trend_anomalies:
                        st.success(f"Found {len(trend_anomalies)} trend changes!")
                        
                        # Store trend anomalies as insights
                        for anomaly in trend_anomalies:
                            # Format the anomaly as an insight
                            insight_data = {
                                'date_generated': datetime.now().strftime('%Y-%m-%d'),
                                'type': 'trend_change',
                                'description': anomaly['description'],
                                'metric': anomaly.get('metric'),
                                'office': anomaly.get('office'),
                                'channel': anomaly.get('channel'),
                                'recommendation': anomaly.get('recommendation'),
                                'severity': anomaly.get('severity', 'medium'),
                                'status': 'new',
                                'period_start': anomaly.get('period'),
                                'period_end': anomaly.get('period')
                            }
                            
                            # Store the insight
                            db_manager.store_insight(insight_data)
                    else:
                        st.info("No significant trend changes detected.")
        
        # Display insights
        st.markdown("---")
        st.subheader("Current Insights")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            insight_type_filter = st.selectbox(
                "Filter by type",
                ["All Types", "outlier", "trend_change", "comparative"]
            )
            insight_type = None if insight_type_filter == "All Types" else insight_type_filter
        
        with col2:
            severity_filter = st.selectbox(
                "Filter by severity",
                ["All Severities", "high", "medium", "low"]
            )
            severity = None if severity_filter == "All Severities" else severity_filter
        
        with col3:
            status_filter = st.selectbox(
                "Filter by status",
                ["All Statuses", "new", "acknowledged", "resolved"]
            )
            status = None if status_filter == "All Statuses" else status_filter
        
        # Get insights with filters
        insights = db_manager.get_insights(
            insight_type=insight_type,
            start_date=start_date,
            end_date=end_date,
            status=status,
            office=office_filter,
            channel=channel_filter
        )
        
        # Further filter by severity if needed
        if severity:
            insights = [insight for insight in insights if insight.get('severity') == severity]
        
        # Display insights
        if insights:
            for insight in insights:
                # Create a unique key for the expander
                expander_key = f"insight_{insight['id']}"
                
                # Add severity badge
                severity_badge = ""
                if insight.get('severity') == 'high':
                    severity_badge = "🔴 HIGH: "
                elif insight.get('severity') == 'medium':
                    severity_badge = "🟠 MEDIUM: "
                elif insight.get('severity') == 'low':
                    severity_badge = "🟢 LOW: "
                
                # Add status badge
                status_badge = ""
                if insight.get('status') == 'new':
                    status_badge = "🆕 "
                elif insight.get('status') == 'acknowledged':
                    status_badge = "👁️ "
                elif insight.get('status') == 'resolved':
                    status_badge = "✅ "

                with st.expander(f"{status_badge}{severity_badge}{insight['description']}"):
                    st.write(f"**Type:** {insight['type'].replace('_', ' ').title()}")
                    st.write(f"**Generated:** {insight['date_generated']}")
                    
                    # Show period if available
                    if insight.get('period_start') and insight.get('period_end'):
                        st.write(f"**Period:** {insight['period_start']} to {insight['period_end']}")
                    
                    # Show affected dimensions
                    dimensions = []
                    if insight.get('office'):
                        dimensions.append(f"Office: {insight['office']}")
                    if insight.get('channel'):
                        dimensions.append(f"Channel: {insight['channel']}")
                    if insight.get('division'):
                        dimensions.append(f"Division: {insight['division']}")
                    if insight.get('metric'):
                        dimensions.append(f"Metric: {insight['metric'].title()}")
                    
                    if dimensions:
                        st.write("**Affects:** " + ", ".join(dimensions))
                    
                    if insight.get('recommendation'):
                        st.markdown(f"**Recommendation:** {insight['recommendation']}")
                    
                    # Status update buttons
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if insight.get('status') != 'acknowledged':
                            if st.button("Acknowledge", key=f"ack_{insight['id']}"):
                                db_manager.update_insight_status(insight['id'], 'acknowledged')
                                st.success("Status updated!")
                                st.rerun()

                    
                    with col2:
                        if insight.get('status') != 'resolved':
                            if st.button("Mark Resolved", key=f"res_{insight['id']}"):
                                db_manager.update_insight_status(insight['id'], 'resolved')
                                st.success("Status updated!")
                                sst.rerun()

                    
                    with col3:
                        if OPENAI_API_KEY:
                            if st.button("Analyze Further", key=f"ai_{insight['id']}"):
                                with st.spinner("Generating detailed analysis..."):
                                    analysis = ai_analyst.analyze_insight(insight)
                                    st.markdown("## Detailed Analysis")
                                    st.markdown(analysis)
        else:
            st.info("No insights found matching the current filters.")

# Reports page
elif page == "Reports":
    st.header("Marketing Reports")
    
    if is_database_empty():
        show_empty_state()
    else:
        # Report type selector
        report_type = st.selectbox(
            "Select Report Type",
            ["Weekly Performance", "Channel Analysis", "Office Performance", "Custom Report"]
        )
        
        if report_type == "Weekly Performance":
            st.subheader("Weekly Marketing Performance Report")
            
            # Date selector for the report end date
            report_end_date = st.date_input(
                "Report End Date",
                datetime.strptime(end_date, '%Y-%m-%d')
            ).strftime('%Y-%m-%d')
            
            if st.button("Generate Weekly Report"):
                with st.spinner("Generating weekly performance report..."):
                    report_text = ai_analyst.generate_weekly_report(
                        end_date=report_end_date,
                        days_back=7
                    )
                    
                    st.markdown(report_text)
                    
                    # Option to export
                    report_bytes = report_text.encode()
                    st.download_button(
                        label="Download Report",
                        data=report_bytes,
                        file_name=f"weekly_report_{report_end_date}.md",
                        mime="text/markdown"
                    )
        
        elif report_type == "Channel Analysis":
            st.subheader("Channel Analysis Report")
            
            # Filter to specific channels if desired
            channels = db_manager.get_all_channels()
            selected_channels = st.multiselect(
                "Select channels to include (leave empty for all)",
                channels
            )
            
            channel_filter = selected_channels if selected_channels else None
            
            if st.button("Generate Channel Report"):
                with st.spinner("Generating channel analysis report..."):
                    # Load data for the selected period and channels
                    df = db_manager.get_marketing_data(
                        start_date=start_date,
                        end_date=end_date,
                        channel=channel_filter
                    )
                    
                    if df.empty:
                        st.warning(f"No data available for the period: {start_date} to {end_date}")
                    else:
                        # Create visualizations
                        channel_perf_chart = visualizer.create_channel_performance_chart(df)
                        st.plotly_chart(channel_perf_chart, use_container_width=True)
                        
                        # Channel comparison table
                        if 'channel' in df.columns:
                            st.subheader("Channel Performance Metrics")
                            
                            channel_data = df.groupby('channel').agg({
                                'spend': 'sum',
                                'instructions': 'sum',
                                'leads': 'sum',
                                'roi': 'mean'
                            }).reset_index()
                            
                            # Calculate derived metrics
                            channel_data['cost_per_instruction'] = channel_data.apply(
                                lambda row: row['spend'] / row['instructions'] if row['instructions'] > 0 else np.nan,
                                axis=1
                            )
                            
                            channel_data['cost_per_lead'] = channel_data.apply(
                                lambda row: row['spend'] / row['leads'] if row['leads'] > 0 else np.nan,
                                axis=1
                            )
                            
                            channel_data['conversion_rate'] = channel_data.apply(
                                lambda row: row['instructions'] / row['leads'] if row['leads'] > 0 else np.nan,
                                axis=1
                            )
                            
                            # Format the table
                            formatted_channel_data = channel_data.copy()
                            formatted_channel_data['spend'] = formatted_channel_data['spend'].apply(format_currency)
                            formatted_channel_data['roi'] = formatted_channel_data['roi'].apply(format_percent)
                            formatted_channel_data['cost_per_instruction'] = formatted_channel_data['cost_per_instruction'].apply(
                                lambda x: format_currency(x) if pd.notna(x) else 'N/A'
                            )
                            formatted_channel_data['cost_per_lead'] = formatted_channel_data['cost_per_lead'].apply(
                                lambda x: format_currency(x) if pd.notna(x) else 'N/A'
                            )
                            formatted_channel_data['conversion_rate'] = formatted_channel_data['conversion_rate'].apply(
                                lambda x: format_percent(x) if pd.notna(x) else 'N/A'
                            )
                            
                            # Rename columns for display
                            formatted_channel_data.columns = [
                                'Channel', 'Spend', 'Instructions', 'Leads', 'ROI', 
                                'Cost per Instruction', 'Cost per Lead', 'Conversion Rate'
                            ]
                            
                            st.dataframe(formatted_channel_data, use_container_width=True)
                        
                        # AI analysis of channel performance
                        if OPENAI_API_KEY:
                            with st.spinner("Generating AI analysis..."):
                                question = f"Provide a comprehensive analysis of marketing channel performance between {start_date} and {end_date}. Compare the channels on all key metrics, and provide specific recommendations for each channel."
                                analysis = ai_analyst.answer_question(
                                    question=question,
                                    context_start_date=start_date,
                                    context_end_date=end_date
                                )
                                st.markdown("## Channel Analysis")
                                st.markdown(analysis)


                        elif report_type == "Office Performance":
                            st.subheader("Office Performance Report")
            
            # Filter to specific offices if desired
            offices = db_manager.get_all_offices()
            selected_offices = st.multiselect(
                "Select offices to include (leave empty for all)",
                offices
            )
            
            office_filter = selected_offices if selected_offices else None
            
            if st.button("Generate Office Report"):
                with st.spinner("Generating office performance report..."):
                    # Load data for the selected period and offices
                    df = db_manager.get_marketing_data(
                        start_date=start_date,
                        end_date=end_date,
                        office=office_filter
                    )
                    
                    if df.empty:
                        st.warning(f"No data available for the period: {start_date} to {end_date}")
                    else:
                        # Create office performance chart
                        office_chart = visualizer.create_office_performance_chart(
                            df, 
                            metric='instructions',
                            top_n=15
                        )
                        st.plotly_chart(office_chart, use_container_width=True)
                        
                        # Office comparison table
                        if 'office' in df.columns:
                            st.subheader("Office Performance Metrics")
                            
                            office_data = df.groupby('office').agg({
                                'spend': 'sum',
                                'instructions': 'sum',
                                'leads': 'sum',
                                'roi': 'mean'
                            }).reset_index()
                            
                            # Calculate derived metrics
                            office_data['cost_per_instruction'] = office_data.apply(
                                lambda row: row['spend'] / row['instructions'] if row['instructions'] > 0 else np.nan,
                                axis=1
                            )
                            
                            office_data['cost_per_lead'] = office_data.apply(
                                lambda row: row['spend'] / row['leads'] if row['leads'] > 0 else np.nan,
                                axis=1
                            )
                            
                            # Sort by instructions
                            office_data = office_data.sort_values('instructions', ascending=False)
                            
                            # Format the table
                            formatted_office_data = office_data.copy()
                            formatted_office_data['spend'] = formatted_office_data['spend'].apply(format_currency)
                            formatted_office_data['roi'] = formatted_office_data['roi'].apply(format_percent)
                            formatted_office_data['cost_per_instruction'] = formatted_office_data['cost_per_instruction'].apply(
                                lambda x: format_currency(x) if pd.notna(x) else 'N/A'
                            )
                            formatted_office_data['cost_per_lead'] = formatted_office_data['cost_per_lead'].apply(
                                lambda x: format_currency(x) if pd.notna(x) else 'N/A'
                            )
                            
                            # Rename columns for display
                            formatted_office_data.columns = [
                                'Office', 'Spend', 'Instructions', 'Leads', 'ROI', 
                                'Cost per Instruction', 'Cost per Lead'
                            ]
                            
                            st.dataframe(formatted_office_data, use_container_width=True)
                        
                        # AI analysis of office performance
                        if OPENAI_API_KEY:
                            with st.spinner("Generating AI analysis..."):
                                question = f"Provide a comprehensive analysis of office performance between {start_date} and {end_date}. Identify top and bottom performing offices, explain potential reasons for performance differences, and provide specific recommendations."
                                analysis = ai_analyst.answer_question(
                                    question=question,
                                    context_start_date=start_date,
                                    context_end_date=end_date
                                )
                                st.markdown("## Office Performance Analysis")
                                st.markdown(analysis)
        
        elif report_type == "Custom Report":
            st.subheader("Custom Marketing Report")
            
            st.markdown("""
            Enter the specific details and questions you would like addressed in your custom report.
            MAE will analyze your marketing data and generate a comprehensive report based on your requirements.
            """)
            
            report_title = st.text_input("Report Title", "Marketing Performance Analysis")
            
            report_requirements = st.text_area(
                "Report Requirements",
                height=150,
                placeholder="Describe what you want to see in this report. For example:\n- Analyze the performance of Direct Mail campaigns\n- Compare ROI across all channels\n- Identify underperforming offices\n- Provide recommendations for budget allocation"
            )
            
            if report_requirements and OPENAI_API_KEY:
                if st.button("Generate Custom Report"):
                    with st.spinner("Generating custom report..."):
                        # Prepare the report prompt
                        prompt = f"""
                        Create a comprehensive marketing report titled "{report_title}" covering the period from {start_date} to {end_date}.
                        
                        Requirements:
                        {report_requirements}
                        
                        Please include data-driven insights, visualizations commentary, and specific recommendations.
                        """
                        
                        report = ai_analyst.answer_question(
                            question=prompt,
                            context_start_date=start_date,
                            context_end_date=end_date
                        )
                        
                        st.markdown(f"# {report_title}")
                        st.markdown(f"**Period:** {start_date} to {end_date}")
                        st.markdown("---")
                        st.markdown(report)
                        
                        # Option to export
                        full_report = f"# {report_title}\n\n**Period:** {start_date} to {end_date}\n\n---\n\n{report}"
                        report_bytes = full_report.encode()
                        st.download_button(
                            label="Download Report",
                            data=report_bytes,
                            file_name=f"{report_title.lower().replace(' ', '_')}_{start_date}_to_{end_date}.md",
                            mime="text/markdown"
                        )
            elif not OPENAI_API_KEY:
                st.warning("To generate custom reports, please configure your OpenAI API key.")

# Chat with MAE page
elif page == "Chat with MAE":
    st.header("Chat with MAE")
    
    if is_database_empty():
        show_empty_state()
    else:
        st.markdown("""
        Ask MAE anything about your marketing performance data. 
        MAE can answer questions, provide insights, and help you understand your marketing data better.
        """)
        
                # Chat input
        user_input = st.text_input("Ask MAE", key="user_query")

        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**MAE:** {message['content']}")

        # Process input only if it's new
        if user_input and st.session_state.get("last_user_input") != user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.last_user_input = user_input

            if OPENAI_API_KEY:
                with st.spinner("MAE is thinking..."):
                    try:
                        ai_response = ai_analyst.answer_question(
                            question=user_input,
                            context_start_date=start_date if 'start_date' in locals() else None,
                            context_end_date=end_date if 'end_date' in locals() else None
                        )
                        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                        db_manager.store_interaction(user_input, ai_response)
                    except Exception as e:
                        error_message = f"Error processing question: {str(e)}"
                        logger.error(error_message)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_message})
            else:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "Please configure your OpenAI API key."
                })


        
            # Clear chat button
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.is_processing = False
                st.rerun()

