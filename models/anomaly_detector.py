import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.ensemble import IsolationForest
from scipy import stats

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import DEFAULT_ANOMALY_THRESHOLD, COMPARISON_METRICS
from database.db_manager import DatabaseManager
from models.data_processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Detects anomalies in marketing data using various statistical methods.
    Identifies unusual patterns, outliers, and significant deviations.
    """
    
    def __init__(self, db_manager: DatabaseManager = None, 
                data_processor: DataProcessor = None,
                threshold: float = DEFAULT_ANOMALY_THRESHOLD):
        """
        Initialize the anomaly detector.
        
        Args:
            db_manager: Database manager instance
            data_processor: Data processor instance
            threshold: Threshold for anomaly detection (default from config)
        """
        self.db_manager = db_manager if db_manager else DatabaseManager()
        self.data_processor = data_processor if data_processor else DataProcessor(self.db_manager)
        self.threshold = threshold
    
    def detect_performance_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect performance anomalies in the given DataFrame using statistical methods.
        
        Args:
            df: DataFrame containing marketing data
            
        Returns:
            List of anomalies detected
        """
        anomalies = []

        # Ensure required columns exist
        required_cols = ['date', 'office', 'channel', 'spend', 'roi', 'instructions', 'leads']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns for anomaly detection: {missing_cols}")
            # Continue with available columns
        
        # Ensure date is in datetime format
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Detect outliers for each numeric metric by channel and office
        metrics = [col for col in ['spend', 'roi', 'instructions', 'leads'] if col in df.columns]
        
        for metric in metrics:
            # Group by channel and office
            if 'channel' in df.columns and 'office' in df.columns:
                grouped = df.groupby(['channel', 'office'])
            elif 'channel' in df.columns:
                grouped = df.groupby(['channel'])
            elif 'office' in df.columns:
                grouped = df.groupby(['office'])
            else:
                # If neither channel nor office is available, use the whole dataset
                grouped = [(None, df)]
            
            for group_name, group_df in grouped:
                if len(group_df) < 5:  # Skip groups with insufficient data
                    continue
                
                # Z-score method for outlier detection
                z_scores = np.abs(stats.zscore(group_df[metric], nan_policy='omit'))
                
                # Consider anything above 3 standard deviations as an outlier
                outliers = group_df[z_scores > 3]
                
                for _, row in outliers.iterrows():
                    anomaly = {
                        'type': 'outlier',
                        'metric': metric,
                        'value': row[metric],
                        'date': row['date'].strftime('%Y-%m-%d') if 'date' in row else None,
                        'office': row['office'] if 'office' in row else None,
                        'channel': row['channel'] if 'channel' in row else None,
                        'division': row['division'] if 'division' in row else None,
                        'z_score': float(z_scores[group_df.index.get_loc(row.name)]),
                        'description': f"Unusual {metric} value detected"
                    }
                    
                    # Add contextual information based on metric
                    if metric == 'spend':
                        direction = "high" if row[metric] > group_df[metric].median() else "low"
                        anomaly['description'] = f"Unusually {direction} spend detected"
                        anomaly['severity'] = 'high' if direction == 'high' else 'medium'
                        
                    elif metric == 'roi':
                        direction = "high" if row[metric] > group_df[metric].median() else "low"
                        anomaly['description'] = f"Unusually {direction} ROI detected"
                        anomaly['severity'] = 'high' if direction == 'low' else 'medium'
                        
                    elif metric == 'instructions':
                        direction = "high" if row[metric] > group_df[metric].median() else "low"
                        anomaly['description'] = f"Unusually {direction} number of instructions detected"
                        anomaly['severity'] = 'high' if direction == 'low' else 'medium'
                        
                    elif metric == 'leads':
                        direction = "high" if row[metric] > group_df[metric].median() else "low"
                        anomaly['description'] = f"Unusually {direction} number of leads detected"
                        anomaly['severity'] = 'high' if direction == 'low' else 'medium'
                    
                    # Generate a recommendation
                    if 'channel' in row and 'office' in row:
                        channel_context = f" for {row['channel']} in {row['office']}"
                    elif 'channel' in row:
                        channel_context = f" for {row['channel']}"
                    elif 'office' in row:
                        channel_context = f" in {row['office']}"
                    else:
                        channel_context = ""
                    
                    if direction == "high" and metric == 'spend':
                        anomaly['recommendation'] = f"Investigate the high marketing spend{channel_context}. Check if there were special campaigns or potential errors in spend recording."
                    elif direction == "low" and (metric == 'roi' or metric == 'instructions' or metric == 'leads'):
                        anomaly['recommendation'] = f"Review marketing effectiveness{channel_context}. Consider adjusting strategy or reallocating budget if this trend continues."
                    
                    anomalies.append(anomaly)
        
        return anomalies
    
    def detect_trend_changes(self, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           group_by: List[str] = None,
                           window_size: int = 3) -> List[Dict]:
        """
        Detect significant changes in trends over time.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            group_by: Dimensions to group by (e.g., ['channel', 'office'])
            window_size: Size of the moving window for trend analysis
            
        Returns:
            List of trend anomalies detected
        """
        if not self.db_manager:
            logger.error("Database manager required for trend change detection")
            return []
        
        # Default grouping if none provided
        if group_by is None:
            group_by = ['channel']
        
        # Get data for the period
        df = self.db_manager.get_marketing_data(
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            logger.warning("No data available for trend analysis")
            return []
        
        # Ensure date is in datetime format
        if 'date' not in df.columns:
            logger.error("'date' column required for trend analysis")
            return []
        
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Group data by month and specified dimensions
        time_period = 'monthly'
        metrics = [m.lower() for m in COMPARISON_METRICS if m.lower() in df.columns]
        
        if not metrics:
            logger.warning("No valid metrics found for trend analysis")
            return []
        
        # Group data - debugging the output of get_aggregated_data
        try:
            aggregated = self.data_processor.get_aggregated_data(
                df=df,
                time_period=time_period,
                group_by=group_by
            )
            
            # Log the type and structure of the returned data for debugging
            logger.info(f"Aggregated data type: {type(aggregated)}")
            
            # Convert to DataFrame if it's a list
            if isinstance(aggregated, list):
                # If empty list, return empty result
                if not aggregated:
                    logger.warning("Aggregated data is an empty list")
                    return []
                    
                # If list contains DataFrames, concatenate them
                if all(isinstance(item, pd.DataFrame) for item in aggregated):
                    logger.info("Converting list of DataFrames to single DataFrame")
                    aggregated = pd.concat(aggregated)
                # Otherwise, try to convert list to DataFrame
                else:
                    logger.info("Converting list to DataFrame")
                    aggregated = pd.DataFrame(aggregated)
            
            # Final check to ensure we have a DataFrame
            if not isinstance(aggregated, pd.DataFrame):
                logger.warning(f"Failed to convert aggregated data to DataFrame. Type: {type(aggregated)}")
                return []
                
            if len(aggregated) == 0:
                logger.warning("No aggregated data available for trend analysis")
                return []
                
        except Exception as e:
            logger.error(f"Error processing aggregated data: {str(e)}")
            return []
        
        # Perform trend analysis for each group
        trend_anomalies = []
        
        # Create a pivot for each metric to analyze time series
        for metric in metrics:
            # FIX: Check if metric is in columns before continuing
            if metric not in aggregated.columns:
                logger.warning(f"Metric '{metric}' not found in aggregated data columns")
                continue
            
            # Define dimensions to group by (excluding 'period')
            dims = [col for col in group_by if col in aggregated.columns and col != 'period']
            
            if not dims:
                # If no valid dimensions, analyze the overall trend
                series = aggregated.sort_values('period')[metric]
                
                # Skip if not enough data points
                if len(series) < window_size + 1:
                    continue
                
                # Calculate moving average
                rolling = series.rolling(window=window_size).mean()
                
                # Calculate percentage changes
                pct_changes = series.pct_change()
                
                # Look for significant changes (beyond threshold)
                for i in range(window_size, len(series)):
                    if abs(pct_changes.iloc[i]) > self.threshold:
                        direction = "increase" if pct_changes.iloc[i] > 0 else "decrease"
                        severity = "high" if abs(pct_changes.iloc[i]) > self.threshold * 2 else "medium"
                        
                        anomaly = {
                            'type': 'trend_change',
                            'metric': metric,
                            'period': aggregated['period'].iloc[i].strftime('%Y-%m-%d'),
                            'value': float(series.iloc[i]),
                            'previous_value': float(series.iloc[i-1]),
                            'change_pct': float(pct_changes.iloc[i] * 100),
                            'direction': direction,
                            'severity': severity,
                            'description': f"Significant {direction} in {metric} detected",
                            'recommendation': self._generate_trend_recommendation(metric, direction)
                        }
                        
                        trend_anomalies.append(anomaly)
            else:
                # Group by dimensions
                for name, group in aggregated.groupby(dims):
                    # Convert to list if it's a single value
                    if not isinstance(name, tuple):
                        name = (name,)
                    
                    # Skip if not enough data points
                    if len(group) < window_size + 1:
                        continue
                    
                    # Sort by period
                    group = group.sort_values('period')
                    series = group[metric]
                    
                    # Calculate percentage changes
                    pct_changes = series.pct_change()
                    
                    # Look for significant changes (beyond threshold)
                    for i in range(window_size, len(series)):
                        if abs(pct_changes.iloc[i]) > self.threshold:
                            direction = "increase" if pct_changes.iloc[i] > 0 else "decrease"
                            severity = "high" if abs(pct_changes.iloc[i]) > self.threshold * 2 else "medium"
                            
                            # Create context description
                            context = ""
                            for j, dim in enumerate(dims):
                                context += f"{dim}='{name[j]}', "
                            context = context.rstrip(', ')
                            
                            anomaly = {
                                'type': 'trend_change',
                                'metric': metric,
                                'period': group['period'].iloc[i].strftime('%Y-%m-%d'),
                                'value': float(series.iloc[i]),
                                'previous_value': float(series.iloc[i-1]),
                                'change_pct': float(pct_changes.iloc[i] * 100),
                                'direction': direction,
                                'severity': severity,
                                'description': f"Significant {direction} in {metric} detected for {context}",
                            }
                            
                            # Add dimension values
                            for j, dim in enumerate(dims):
                                anomaly[dim] = name[j]
                            
                            # Add recommendation
                            anomaly['recommendation'] = self._generate_trend_recommendation(
                                metric, direction, context=context
                            )
                            
                            trend_anomalies.append(anomaly)
        
        return trend_anomalies
    
    def detect_comparative_anomalies(self, 
                                  current_period_start: str,
                                  current_period_end: str,
                                  previous_period_start: Optional[str] = None,
                                  previous_period_end: Optional[str] = None,
                                  comparison_dimensions: List[str] = None) -> List[Dict]:
        """
        Detect anomalies by comparing current period with previous period.
        
        Args:
            current_period_start: Start date of current period
            current_period_end: End date of current period
            previous_period_start: Start date of previous period (if None, will calculate)
            previous_period_end: End date of previous period (if None, will calculate)
            comparison_dimensions: Dimensions to include in comparison
            
        Returns:
            List of comparative anomalies detected
        """
        # Get comparison data
        comparison_df = self.data_processor.compare_periods(
            current_period_start=current_period_start,
            current_period_end=current_period_end,
            previous_period_start=previous_period_start,
            previous_period_end=previous_period_end,
            comparison_dimensions=comparison_dimensions
        )
        
        if comparison_df.empty:
            logger.warning("No comparison data available for anomaly detection")
            return []
        
        # Detect anomalies based on percentage changes
        anomalies = []
        metrics = [m.lower() for m in COMPARISON_METRICS]
        
        for metric in metrics:
            pct_change_col = f"{metric}_pct_change"
            
            if pct_change_col not in comparison_df.columns:
                continue
            
            # Filter rows with significant changes
            significant_changes = comparison_df[
                (comparison_df[pct_change_col].abs() > self.threshold * 100) & 
                (comparison_df[pct_change_col] != np.inf)
            ]
            
            for _, row in significant_changes.iterrows():
                direction = "increase" if row[pct_change_col] > 0 else "decrease"
                severity = "high" if abs(row[pct_change_col]) > self.threshold * 200 else "medium"
                
                # Create context description
                context_parts = []
                for dim in comparison_dimensions or []:
                    if dim in row and pd.notna(row[dim]):
                        context_parts.append(f"{dim}='{row[dim]}'")
                
                context = ", ".join(context_parts)
                
                anomaly = {
                    'type': 'comparative',
                    'metric': metric,
                    'period_start': current_period_start,
                    'period_end': current_period_end,
                    'previous_period_start': row['previous_period_start'],
                    'previous_period_end': row['previous_period_end'],
                    'current_value': float(row[f"{metric}_current"]) if pd.notna(row[f"{metric}_current"]) else None,
                    'previous_value': float(row[f"{metric}_previous"]) if pd.notna(row[f"{metric}_previous"]) else None,
                    'change_pct': float(row[pct_change_col]),
                    'direction': direction,
                    'severity': severity,
                    'description': f"Significant {direction} in {metric} compared to previous period for {context}" if context else f"Significant {direction} in {metric} compared to previous period",
                }
                
                # Add dimension values
                for dim in comparison_dimensions or []:
                    if dim in row and pd.notna(row[dim]):
                        anomaly[dim] = row[dim]
                
                # Add recommendation
                anomaly['recommendation'] = self._generate_comparative_recommendation(
                    metric, direction, context=context
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def detect_all_anomalies(self, 
                           days_back: int = 30,
                           store_insights: bool = True) -> List[Dict]:
        """
        Run all anomaly detection methods and return a comprehensive list of insights.
        
        Args:
            days_back: Number of days to look back for data
            store_insights: Whether to store detected anomalies as insights
            
        Returns:
            List of all anomalies detected
        """
        anomalies = []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates as strings
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Get data for the period
        df = self.db_manager.get_marketing_data(
            start_date=start_date_str,
            end_date=end_date_str
        )
        
        if df.empty:
            logger.warning(f"No data available for the last {days_back} days")
            return []
        
        # 1. Detect performance anomalies (outliers)
        performance_anomalies = self.detect_performance_anomalies(df)
        anomalies.extend(performance_anomalies)
        
        # 2. Detect trend changes
        trend_anomalies = self.detect_trend_changes(
            start_date=start_date_str,
            end_date=end_date_str,
            group_by=['channel', 'office']
        )
        anomalies.extend(trend_anomalies)
        
        # 3. Detect comparative anomalies
        # Current period: last 7 days
        current_end = end_date
        current_start = current_end - timedelta(days=7)
        
        # Previous period: 7 days before that
        previous_end = current_start - timedelta(days=1)
        previous_start = previous_end - timedelta(days=7)
        
        comparative_anomalies = self.detect_comparative_anomalies(
            current_period_start=current_start.strftime('%Y-%m-%d'),
            current_period_end=current_end.strftime('%Y-%m-%d'),
            previous_period_start=previous_start.strftime('%Y-%m-%d'),
            previous_period_end=previous_end.strftime('%Y-%m-%d'),
            comparison_dimensions=['channel', 'office']
        )
        anomalies.extend(comparative_anomalies)
        
        # Store insights in the database if requested
        if store_insights and anomalies and self.db_manager:
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
                    'status': 'new'
                }
                
                # Add period information if available
                if 'period' in anomaly:
                    insight_data['period_start'] = anomaly['period']
                    insight_data['period_end'] = anomaly['period']
                elif 'period_start' in anomaly and 'period_end' in anomaly:
                    insight_data['period_start'] = anomaly['period_start']
                    insight_data['period_end'] = anomaly['period_end']
                
                # Store the insight
                self.db_manager.store_insight(insight_data)
        
        return anomalies
    
    def _generate_trend_recommendation(self, metric: str, direction: str, context: str = "") -> str:
        """
        Generate a recommendation based on the trend change.
        
        Args:
            metric: The metric that changed
            direction: 'increase' or 'decrease'
            context: Additional context (e.g., channel, office)
            
        Returns:
            Recommendation string
        """
        context_str = f" for {context}" if context else ""
        
        if metric == 'spend':
            if direction == 'increase':
                return f"Investigate the increased marketing spend{context_str}. Ensure it's authorized and aligned with expected ROI."
            else:
                return f"Note the decreased marketing spend{context_str}. Check if this is intended or if there are budget allocation issues."
                
        elif metric == 'roi':
            if direction == 'increase':
                return f"The improved ROI{context_str} indicates successful marketing strategies. Consider maintaining or expanding these approaches."
            else:
                return f"Address the declining ROI{context_str}. Review marketing tactics, target audience, and messaging to improve effectiveness."
                
        elif metric == 'instructions':
            if direction == 'increase':
                return f"The increase in instructions{context_str} shows positive results. Analyze what's working well and apply to other areas."
            else:
                return f"The decrease in instructions{context_str} requires attention. Evaluate market conditions and marketing strategies to reverse the trend."
                
        elif metric == 'leads':
            if direction == 'increase':
                return f"The growth in leads{context_str} is promising. Ensure the sales team is prepared to handle the increased volume effectively."
            else:
                return f"Address the decline in leads{context_str}. Review lead generation tactics and consider adjusting messaging or targeting."
        
        return f"Monitor the {direction} in {metric}{context_str} and adjust strategies accordingly."
    
    def _generate_comparative_recommendation(self, metric: str, direction: str, context: str = "") -> str:
        """
        Generate a recommendation based on comparative analysis.
        
        Args:
            metric: The metric being compared
            direction: 'increase' or 'decrease'
            context: Additional context (e.g., channel, office)
            
        Returns:
            Recommendation string
        """
        context_str = f" for {context}" if context else ""
        
        if metric == 'spend':
            if direction == 'increase':
                return f"Marketing spend is significantly higher than the previous period{context_str}. Ensure this increased investment is generating proportional returns."
            else:
                return f"Marketing spend has decreased compared to the previous period{context_str}. Verify if this aligns with your budget strategy or if adjustments are needed."
                
        elif metric == 'roi':
            if direction == 'increase':
                return f"ROI has improved compared to the previous period{context_str}. Identify the factors contributing to this success and apply them to other areas."
            else:
                return f"ROI has decreased compared to the previous period{context_str}. Analyze what has changed and take corrective actions to restore performance."
                
        elif metric == 'instructions':
            if direction == 'increase':
                return f"Instructions have increased compared to the previous period{context_str}. Leverage this momentum and ensure resources are in place to maintain quality service."
            else:
                return f"Instructions have decreased compared to the previous period{context_str}. Investigate market conditions and marketing effectiveness to address this decline."
                
        elif metric == 'leads':
            if direction == 'increase':
                return f"Leads have increased compared to the previous period{context_str}. Ensure follow-up processes are optimized to convert these leads effectively."
            else:
                return f"Leads have decreased compared to the previous period{context_str}. Review lead generation strategies and consider refreshing campaigns or messaging."
        
        return f"Analyze the factors behind the {direction} in {metric} compared to the previous period{context_str} and adjust strategies accordingly."