import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Visualizer:
    """
    Creates various visualizations for marketing data using Plotly.
    Handles chart generation for Streamlit display.
    """
    
    def __init__(self, theme_color: str = "#1f77b4"):
        """
        Initialize the visualizer.
        
        Args:
            theme_color: Primary color for charts
        """
        self.theme_color = theme_color
        self.color_sequence = [
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
    
    def create_channel_performance_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a bar chart showing performance by marketing channel.
        
        Args:
            df: DataFrame with channel performance data
            
        Returns:
            Plotly figure object
        """
        if df.empty or 'channel' not in df.columns:
            logger.warning("Cannot create channel performance chart: missing data or 'channel' column")
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No channel data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Aggregate by channel if not already aggregated
        metrics = [col for col in ['spend', 'instructions', 'leads', 'roi'] if col in df.columns]
        
        if not metrics:
            logger.warning("Cannot create channel performance chart: no metrics available")
            fig = go.Figure()
            fig.add_annotation(
                text="No metric data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Check if we need to aggregate
        if len(df) > 10 and 'date' in df.columns:
            # Data is not aggregated, let's do that
            agg_dict = {metric: 'sum' for metric in metrics if metric != 'roi'}
            if 'roi' in metrics:
                agg_dict['roi'] = 'mean'
                
            channel_df = df.groupby('channel').agg(agg_dict).reset_index()
        else:
            # Data is likely already aggregated
            channel_df = df.copy()
        
        # Create subplots based on available metrics
        subplot_titles = []
        for metric in metrics:
            if metric == 'spend':
                subplot_titles.append("Total Spend by Channel")
            elif metric == 'instructions':
                subplot_titles.append("Total Instructions by Channel")
            elif metric == 'leads':
                subplot_titles.append("Total Leads by Channel")
            elif metric == 'roi':
                subplot_titles.append("Average ROI by Channel")
        
        fig = make_subplots(
            rows=len(metrics), 
            cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1
        )
        
        for i, metric in enumerate(metrics):
            row = i + 1
            
            # Sort data for better visualization
            sorted_df = channel_df.sort_values(metric, ascending=False)
            
            # Create bar trace
            bar = go.Bar(
                x=sorted_df['channel'],
                y=sorted_df[metric],
                name=metric.title(),
                marker_color=self.color_sequence[i % len(self.color_sequence)]
            )
            
            fig.add_trace(bar, row=row, col=1)
            
            # Format y-axis
            if metric == 'spend':
                fig.update_yaxes(title_text="Spend (£)", row=row, col=1)
                # Format spend as currency
                fig.update_traces(
                    hovertemplate="Channel: %{x}<br>Spend: £%{y:,.2f}<extra></extra>",
                    row=row, col=1
                )
            elif metric == 'roi':
                fig.update_yaxes(title_text="ROI", row=row, col=1)
                # Format as decimal
                fig.update_traces(
                    hovertemplate="Channel: %{x}<br>ROI: %{y:.2f}<extra></extra>",
                    row=row, col=1
                )
            else:
                fig.update_yaxes(title_text=metric.title(), row=row, col=1)
                # Format as integer
                fig.update_traces(
                    hovertemplate=f"Channel: %{{x}}<br>{metric.title()}: %{{y:,}}<extra></extra>",
                    row=row, col=1
                )
        
        # Update layout
        fig.update_layout(
            height=300 * len(metrics),
            width=800,
            showlegend=False,
            title_text="Channel Performance",
            title_x=0.5
        )
        
        return fig
    
    def create_time_series_chart(self, 
                               df: pd.DataFrame, 
                               metrics: List[str] = None,
                               group_by: str = None) -> go.Figure:
        """
        Create a time series chart showing trends over time.
        
        Args:
            df: DataFrame with time series data
            metrics: List of metrics to plot
            group_by: Column to group by (e.g., 'channel')
            
        Returns:
            Plotly figure object
        """
        if df.empty or 'date' not in df.columns:
            logger.warning("Cannot create time series chart: missing data or 'date' column")
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No time series data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Ensure date is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Default metrics if not specified
        available_metrics = [col for col in ['spend', 'instructions', 'leads', 'roi'] if col in df.columns]
        
        if not metrics:
            metrics = available_metrics
        else:
            # Filter to only include available metrics
            metrics = [m for m in metrics if m in available_metrics]
        
        if not metrics:
            logger.warning("Cannot create time series chart: no metrics available")
            fig = go.Figure()
            fig.add_annotation(
                text="No metric data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Sort by date
        df = df.sort_values('date')
        
        # Create figure
        fig = go.Figure()
        
        if group_by and group_by in df.columns:
            # Group by the specified column (e.g., channel)
            groups = df[group_by].unique()
            
            for i, group in enumerate(groups):
                group_data = df[df[group_by] == group]
                
                for j, metric in enumerate(metrics):
                    fig.add_trace(
                        go.Scatter(
                            x=group_data['date'],
                            y=group_data[metric],
                            mode='lines+markers',
                            name=f"{group} - {metric.title()}",
                            line=dict(color=self.color_sequence[(i * len(metrics) + j) % len(self.color_sequence)]),
                            marker=dict(size=6)
                        )
                    )
            
            # Add group_by to title
            title = f"Trends by {group_by.title()}"
        else:
            # Plot each metric as a separate line
            for i, metric in enumerate(metrics):
                fig.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df[metric],
                        mode='lines+markers',
                        name=metric.title(),
                        line=dict(color=self.color_sequence[i % len(self.color_sequence)]),
                        marker=dict(size=6)
                    )
                )
            
            title = "Trends Over Time"
        
        # Format hover text based on metrics
        for i, trace in enumerate(fig.data):
            metric = metrics[i % len(metrics)] if i < len(fig.data) else "value"
            if metric == 'spend':
                trace.hovertemplate = "Date: %{x|%Y-%m-%d}<br>Spend: £%{y:,.2f}<extra></extra>"
            elif metric == 'roi':
                trace.hovertemplate = "Date: %{x|%Y-%m-%d}<br>ROI: %{y:.2f}<extra></extra>"
            else:
                trace.hovertemplate = f"Date: %{{x|%Y-%m-%d}}<br>{metric.title()}: %{{y:,}}<extra></extra>"
        
        # Update layout
        fig.update_layout(
            height=500,
            width=800,
            title_text=title,
            title_x=0.5,
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title=group_by.title() if group_by else "Metric",
            hovermode="x unified"
        )
        
        # Update axes
        fig.update_xaxes(
            tickformat="%b %Y",
            tickangle=-45,
            tickmode="auto",
            nticks=10
        )
        
        return fig
    
    def create_office_performance_chart(self, 
                                      df: pd.DataFrame, 
                                      metric: str = 'instructions',
                                      top_n: int = 10) -> go.Figure:
        """
        Create a horizontal bar chart showing office performance.
        
        Args:
            df: DataFrame with office performance data
            metric: Metric to plot
            top_n: Number of top offices to show
            
        Returns:
            Plotly figure object
        """
        if df.empty or 'office' not in df.columns or metric not in df.columns:
            logger.warning(f"Cannot create office performance chart: missing data, 'office' column, or '{metric}' column")
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No office performance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Aggregate by office if not already aggregated
        if len(df) > 100 and 'date' in df.columns:
            # Data is not aggregated, let's do that
            agg_func = 'mean' if metric == 'roi' else 'sum'
            office_df = df.groupby('office').agg({metric: agg_func}).reset_index()
        else:
            # Data is likely already aggregated
            office_df = df.copy()
        
        # Sort and get top N offices
        sorted_df = office_df.sort_values(metric, ascending=False).head(top_n)
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                y=sorted_df['office'],
                x=sorted_df[metric],
                orientation='h',
                marker_color=self.theme_color
            )
        )
        
        # Format hover text based on metric
        if metric == 'spend':
            fig.update_traces(
                hovertemplate="Office: %{y}<br>Spend: £%{x:,.2f}<extra></extra>"
            )
            x_title = "Spend (£)"
        elif metric == 'roi':
            fig.update_traces(
                hovertemplate="Office: %{y}<br>ROI: %{x:.2f}<extra></extra>"
            )
            x_title = "ROI"
        else:
            fig.update_traces(
                hovertemplate=f"Office: %{{y}}<br>{metric.title()}: %{{x:,}}<extra></extra>"
            )
            x_title = metric.title()
        
        # Update layout
        fig.update_layout(
            height=500,
            width=800,
            title_text=f"Top {top_n} Offices by {metric.title()}",
            title_x=0.5,
            xaxis_title=x_title,
            yaxis_title="Office",
            yaxis=dict(
                categoryorder='total ascending'
            )
        )
        
        return fig
    
    def create_comparison_chart(self, df: pd.DataFrame, 
                              metric: str = 'instructions',
                              comparison_col: str = 'channel') -> go.Figure:
        """
        Create a comparison chart between two periods.
        
        Args:
            df: DataFrame with comparison data
            metric: Metric to compare
            comparison_col: Column to group by (e.g., 'channel')
            
        Returns:
            Plotly figure object
        """
        if df.empty or comparison_col not in df.columns:
            logger.warning(f"Cannot create comparison chart: missing data or '{comparison_col}' column")
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No comparison data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Check if required columns exist
        current_col = f"{metric}_current"
        previous_col = f"{metric}_previous"
        
        if current_col not in df.columns or previous_col not in df.columns:
            logger.warning(f"Cannot create comparison chart: missing '{current_col}' or '{previous_col}' columns")
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Missing comparison data for {metric}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Sort by current period value
        sorted_df = df.sort_values(current_col, ascending=False)
        
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=sorted_df[comparison_col],
                y=sorted_df[current_col],
                name="Current Period",
                marker_color=self.theme_color
            )
        )
        
        fig.add_trace(
            go.Bar(
                x=sorted_df[comparison_col],
                y=sorted_df[previous_col],
                name="Previous Period",
                marker_color="#A0A0A0"  # Gray for previous period
            )
        )
        
        # Add percentage change as text
        if f"{metric}_pct_change" in df.columns:
            annotations = []
            
            for i, row in sorted_df.iterrows():
                pct_change = row[f"{metric}_pct_change"]
                
                if pd.notna(pct_change) and pct_change != float('inf'):
                    # Determine position (above the higher of the two bars)
                    y_pos = max(row[current_col], row[previous_col])
                    
                    # Determine color based on positive/negative
                    color = "green" if pct_change >= 0 else "red"
                    
                    # Format the text
                    text = f"+{pct_change:.1f}%" if pct_change >= 0 else f"{pct_change:.1f}%"
                    
                    annotations.append(
                        dict(
                            x=row[comparison_col],
                            y=y_pos,
                            text=text,
                            font=dict(color=color),
                            showarrow=False,
                            yshift=10
                        )
                    )
            
            fig.update_layout(annotations=annotations)
        
        # Format hover text based on metric
        if metric == 'spend':
            fig.update_traces(
                hovertemplate=f"{comparison_col.title()}: %{{x}}<br>Spend: £%{{y:,.2f}}<extra></extra>"
            )
            y_title = "Spend (£)"
        elif metric == 'roi':
            fig.update_traces(
                hovertemplate=f"{comparison_col.title()}: %{{x}}<br>ROI: %{{y:.2f}}<extra></extra>"
            )
            y_title = "ROI"
        else:
            fig.update_traces(
                hovertemplate=f"{comparison_col.title()}: %{{x}}<br>{metric.title()}: %{{y:,}}<extra></extra>"
            )
            y_title = metric.title()
        
        # Format chart title based on available date info
        title = f"Comparison of {metric.title()} by {comparison_col.title()}"
        
        if 'current_period_start' in df.columns and 'previous_period_start' in df.columns:
            # Get date ranges from first row
            current_start = df['current_period_start'].iloc[0]
            current_end = df['current_period_end'].iloc[0]
            previous_start = df['previous_period_start'].iloc[0]
            previous_end = df['previous_period_end'].iloc[0]
            
            subtitle = f"Current: {current_start} to {current_end}<br>Previous: {previous_start} to {previous_end}"
        else:
            subtitle = "Current vs Previous Period"
        
        # Update layout
        fig.update_layout(
            height=500,
            width=800,
            title=dict(
                text=title,
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            title_x=0.5,
            xaxis_title=comparison_col.title(),
            yaxis_title=y_title,
            legend_title="Period",
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1
        )
        
        # Add subtitle
        fig.add_annotation(
            text=subtitle,
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            showarrow=False,
            font=dict(size=12),
            xanchor="center",
            yanchor="bottom"
        )
        
        return fig
    
    def create_metrics_dashboard(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a dashboard of key metrics.
        
        Args:
            df: DataFrame with marketing data
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            logger.warning("Cannot create metrics dashboard: missing data")
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for dashboard",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Check available metrics
        metrics = [col for col in ['spend', 'instructions', 'leads', 'roi'] if col in df.columns]
        
        if not metrics:
            logger.warning("Cannot create metrics dashboard: no metrics available")
            fig = go.Figure()
            fig.add_annotation(
                text="No metric data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create subplot grid based on number of metrics
        rows = (len(metrics) + 1) // 2
        cols = min(2, len(metrics))
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[m.title() for m in metrics],
            specs=[[{'type': 'indicator'} for _ in range(cols)] for _ in range(rows)]
        )
        
        # Calculate values
        metric_values = {}
        for metric in metrics:
            if metric == 'roi':
                metric_values[metric] = df[metric].mean()
            else:
                metric_values[metric] = df[metric].sum()
        
        # Add indicators
        for i, metric in enumerate(metrics):
            row = i // cols + 1
            col = i % cols + 1
            
            value = metric_values[metric]
            
            # Format value and determine mode based on metric
            if metric == 'spend':
                number_format = "£%{value:,.2f}"
                title = "Total Spend"
            elif metric == 'roi':
                number_format = "%{value:.2f}"
                title = "Average ROI"
            else:
                number_format = "%{value:,}"
                title = f"Total {metric.title()}"
            
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=value,
                    number={'valueformat': number_format.replace('%{value', '')},
                    title={'text': title},
                    domain={'row': row, 'column': col}
                ),
                row=row, col=col
            )
        
        # Update layout
        fig.update_layout(
            height=250 * rows,
            width=800,
            title_text="Key Metrics Dashboard",
            title_x=0.5
        )
        
        return fig
    
    def create_funnel_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a funnel chart showing the conversion process.
        
        Args:
            df: DataFrame with marketing data
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            logger.warning("Cannot create funnel chart: missing data")
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for funnel chart",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Check required columns
        required_cols = ['leads', 'instructions']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Cannot create funnel chart: missing columns {missing_cols}")
            fig = go.Figure()
            fig.add_annotation(
                text="Missing required data for funnel chart",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Calculate values
        total_leads = df['leads'].sum()
        total_instructions = df['instructions'].sum()
        
        # Calculate conversion rate
        conversion_rate = total_instructions / total_leads if total_leads > 0 else 0
        
        # Create labels and values
        labels = ["Leads", "Instructions"]
        values = [total_leads, total_instructions]
        
        # Create the funnel chart
        fig = go.Figure(go.Funnel(
            y=labels,
            x=values,
            textposition="inside",
            textinfo="value+percent initial",
            marker={"color": [self.theme_color, "#ff7f0e"]},
            connector={"line": {"color": "royalblue", "dash": "dot", "width": 3}}
        ))
        
        # Update layout
        fig.update_layout(
            title_text="Lead to Instruction Funnel",
            title_x=0.5,
            height=500,
            width=800
        )
        
        # Add annotation with conversion rate
        fig.add_annotation(
            text=f"Conversion Rate: {conversion_rate:.2%}",
            xref="paper", yref="paper",
            x=0.5, y=-0.1,
            showarrow=False,
            font=dict(size=14)
        )
        
        return fig
    
    def create_heatmap(self, df: pd.DataFrame, 
                     x_col: str = 'office', 
                     y_col: str = 'channel',
                     value_col: str = 'instructions') -> go.Figure:
        """
        Create a heatmap visualization.
        
        Args:
            df: DataFrame with marketing data
            x_col: Column for x-axis
            y_col: Column for y-axis
            value_col: Column for values
            
        Returns:
            Plotly figure object
        """
        if df.empty or x_col not in df.columns or y_col not in df.columns or value_col not in df.columns:
            logger.warning(f"Cannot create heatmap: missing data or required columns")
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Missing required data for heatmap",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Pivot the data
        pivot_df = df.pivot_table(
            index=y_col,
            columns=x_col,
            values=value_col,
            aggfunc='sum' if value_col != 'roi' else 'mean'
        ).fillna(0)
        
        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='Blues',
            hoverongaps=False
        ))
        
        # Format hover text based on value_col
        if value_col == 'spend':
            hovertemplate = f"{y_col.title()}: %{{y}}<br>{x_col.title()}: %{{x}}<br>Spend: £%{{z:,.2f}}<extra></extra>"
        elif value_col == 'roi':
            hovertemplate = f"{y_col.title()}: %{{y}}<br>{x_col.title()}: %{{x}}<br>ROI: %{{z:.2f}}<extra></extra>"
        else:
            hovertemplate = f"{y_col.title()}: %{{y}}<br>{x_col.title()}: %{{x}}<br>{value_col.title()}: %{{z:,}}<extra></extra>"
            
        fig.update_traces(hovertemplate=hovertemplate)
        
        # Update layout
        fig.update_layout(
            title=f"{value_col.title()} by {x_col.title()} and {y_col.title()}",
            height=600,
            width=900,
            title_x=0.5
        )
        
        return fig
