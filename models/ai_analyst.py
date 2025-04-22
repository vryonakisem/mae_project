import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import openai

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import OPENAI_API_KEY, OPENAI_MODEL
from database.db_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIAnalyst:
    """
    Provides AI-powered analysis and recommendations using OpenAI's API.
    Generates natural language insights and responds to user queries.
    """
    
    def __init__(self, db_manager: DatabaseManager = None):
        """
        Initialize the AI analyst.
        
        Args:
            db_manager: Database manager instance for data access
        """
        self.db_manager = db_manager if db_manager else DatabaseManager()
        
        # Initialize OpenAI client if API key is available
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
            self.has_openai = True
            print(f"OpenAI initialized with API key starting with: {api_key[:8]}...")  # Debug message
        else:
            logger.warning("OpenAI API key not found. Running in limited mode.")
            print("OpenAI API key not found!")  # Debug message
            self.has_openai = False
    
    def generate_data_summary(self, df: pd.DataFrame) -> str:
        """
        Generate a summary of the data.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Text summary of the data
        """
        if df.empty:
            return "No data available to summarize."
        
        # If OpenAI is not available, generate a basic summary
        if not self.has_openai:
            return self._generate_basic_summary(df)
        
        try:
            # Prepare a condensed version of the data for the API
            data_sample = df.head(10).to_dict(orient='records')
            data_stats = {
                'row_count': len(df),
                'date_range': {
                    'min': None,
                    'max': None
                },
                'columns': df.columns.tolist(),
                'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
            }
            
            # Add date range with proper conversion
            if 'date' in df.columns:
                try:
                    # Ensure dates are datetime objects
                    if not pd.api.types.is_datetime64_any_dtype(df['date']):
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    
                    # Now safely get min and max dates
                    if not df['date'].isna().all():
                        min_date = df['date'].min()
                        max_date = df['date'].max()
                        
                        if pd.notna(min_date):
                            data_stats['date_range']['min'] = min_date.strftime('%Y-%m-%d')
                        if pd.notna(max_date):
                            data_stats['date_range']['max'] = max_date.strftime('%Y-%m-%d')
                except Exception as e:
                    logger.error(f"Error processing dates in generate_data_summary: {str(e)}")
            
            # Add basic stats for numeric columns
            numeric_stats = {}
            for col in data_stats['numeric_columns']:
                numeric_stats[col] = {
                    'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    'median': float(df[col].median()) if not pd.isna(df[col].median()) else None
                }
            
            data_stats['numeric_stats'] = numeric_stats
            
            # Add value counts for categorical columns (limited to top 5)
            categorical_stats = {}
            for col in data_stats['categorical_columns']:
                if col in df.columns:
                    value_counts = df[col].value_counts().head(5).to_dict()
                    categorical_stats[col] = {str(k): int(v) for k, v in value_counts.items()}
            
            data_stats['categorical_stats'] = categorical_stats
            
            # Create the prompt
            prompt = f"""
            You are MAE (Marketing Analyst Expert), a real estate marketing analyst. 
            Analyze the marketing data for a real estate company with offices across London.
            
            Data statistics: {json.dumps(data_stats)}
            
            Sample data (first 10 rows): {json.dumps(data_sample)}
            
            Provide a concise summary of this marketing data focusing on:
            1. Overview of the time period and data coverage
            2. Key performance metrics (spend, ROI, instructions, leads)
            3. Notable patterns or trends
            4. Performance by marketing channel, if available
            5. Performance by office location, if available
            
            Keep your analysis professional, data-driven, and focused on actionable insights.
            Write in the first person as if you are MAE.
            """
            
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are MAE (Marketing Analyst Expert), a sophisticated AI analyst for a real estate marketing department."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating data summary with OpenAI: {str(e)}")
            return self._generate_basic_summary(df)
    
    def _generate_basic_summary(self, df: pd.DataFrame) -> str:
        """
        Generate a basic summary of the data without using OpenAI.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Basic text summary of the data
        """
        summary = []
        summary.append("# Marketing Data Summary")
        
        # Date range with proper conversion
        if 'date' in df.columns:
            try:
                # Ensure dates are datetime objects
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                
                # Now safely get min and max dates
                if not df['date'].isna().all():
                    min_date = df['date'].min()
                    max_date = df['date'].max()
                    
                    if pd.notna(min_date) and pd.notna(max_date):
                        date_range = f"Data covers the period from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}."
                        summary.append(date_range)
            except Exception as e:
                logger.error(f"Error processing dates in _generate_basic_summary: {str(e)}")
        
        # Basic stats
        summary.append(f"Total records: {len(df)}")
        
        # Metrics
        metrics = ['spend', 'roi', 'instructions', 'leads']
        metrics_summary = []
        
        for metric in metrics:
            if metric in df.columns:
                total = df[metric].sum()
                avg = df[metric].mean()
                
                if metric == 'spend':
                    metrics_summary.append(f"Total spend: £{total:,.2f}, Average spend: £{avg:,.2f}")
                elif metric == 'roi':
                    metrics_summary.append(f"Average ROI: {avg:.2f}")
                elif metric == 'instructions':
                    metrics_summary.append(f"Total instructions: {int(total)}, Average instructions: {avg:.2f}")
                elif metric == 'leads':
                    metrics_summary.append(f"Total leads: {int(total)}, Average leads: {avg:.2f}")
        
        if metrics_summary:
            summary.append("\n## Key Metrics")
            summary.extend(metrics_summary)
        
        # Channels
        if 'channel' in df.columns:
            summary.append("\n## Channel Performance")
            channel_data = df.groupby('channel').agg({
                'spend': 'sum',
                'instructions': 'sum',
                'leads': 'sum'
            }).reset_index()
            
            for _, row in channel_data.iterrows():
                channel_summary = f"- {row['channel']}: "
                metrics_parts = []
                
                if 'spend' in channel_data.columns:
                    metrics_parts.append(f"Spend: £{row['spend']:,.2f}")
                
                if 'instructions' in channel_data.columns:
                    metrics_parts.append(f"Instructions: {int(row['instructions'])}")
                
                if 'leads' in channel_data.columns:
                    metrics_parts.append(f"Leads: {int(row['leads'])}")
                
                channel_summary += ", ".join(metrics_parts)
                summary.append(channel_summary)
        
        # Offices
        if 'office' in df.columns:
            top_offices = df.groupby('office')['instructions'].sum().sort_values(ascending=False).head(5)
            
            if not top_offices.empty:
                summary.append("\n## Top Performing Offices (by Instructions)")
                for office, count in top_offices.items():
                    summary.append(f"- {office}: {int(count)} instructions")
        
        return "\n".join(summary)
    
    def analyze_insight(self, insight: Dict) -> str:
        """
        Generate a detailed analysis of a specific insight.
        
        Args:
            insight: The insight to analyze
            
        Returns:
            Detailed analysis text
        """
        if not self.has_openai:
            return insight.get('description', '') + "\n\n" + insight.get('recommendation', '')
        
        try:
            # Fetch relevant context data
            context_data = self._fetch_context_for_insight(insight)
            
            # Create the prompt
            prompt = f"""
            You are MAE (Marketing Analyst Expert), a real estate marketing analyst.
            Analyze the following marketing insight in detail:

            Insight Type: {insight.get('type')}
            Description: {insight.get('description')}
            Metric: {insight.get('metric')}
            Office: {insight.get('office', 'N/A')}
            Channel: {insight.get('channel', 'N/A')}
            Division: {insight.get('division', 'N/A')}
            Time Period: {insight.get('period_start', 'N/A')} to {insight.get('period_end', 'N/A')}
            Severity: {insight.get('severity', 'medium')}
            
            Additional context data:
            {json.dumps(context_data)}
            
            Provide a detailed analysis of this insight including:
            1. What exactly happened and why it matters
            2. Potential causes based on the data
            3. How this impacts the overall marketing performance
            4. Specific, actionable recommendations
            
            Keep your tone professional but conversational. Write in the first person as if you are MAE.
            Focus on providing practical advice that a marketing manager can implement.
            """
            
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are MAE (Marketing Analyst Expert), a sophisticated AI analyst for a real estate marketing department."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=1200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error analyzing insight with OpenAI: {str(e)}")
            return insight.get('description', '') + "\n\n" + insight.get('recommendation', '')
    
    def _fetch_context_for_insight(self, insight: Dict) -> Dict:
        """
        Fetch relevant context data for an insight.
        
        Args:
            insight: The insight to get context for
            
        Returns:
            Dictionary of context data
        """
        context = {}
        
        if not self.db_manager:
            return context
        
        try:
            # Determine date range for context
            if 'period_start' in insight and insight['period_start']:
                start_date = insight['period_start']
                end_date = insight['period_end'] or start_date
                
                # Expand date range for context
                start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
                expanded_start = (start_date_obj - timedelta(days=30)).strftime('%Y-%m-%d')
                
                # Filter parameters based on insight
                office = insight.get('office')
                channel = insight.get('channel')
                
                # Get historical data
                df = self.db_manager.get_marketing_data(
                    start_date=expanded_start,
                    end_date=end_date,
                    office=office,
                    channel=channel
                )
                
                if not df.empty:
                    # Calculate aggregates by month
                    if 'date' in df.columns:
                        try:
                            # Ensure dates are datetime objects
                            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                            
                            if not df['date'].isna().all():
                                df['month'] = df['date'].dt.strftime('%Y-%m')
                                monthly_data = df.groupby('month').agg({
                                    'spend': 'sum',
                                    'instructions': 'sum',
                                    'leads': 'sum',
                                    'roi': 'mean'
                                }).reset_index()
                                
                                context['monthly_trends'] = monthly_data.to_dict(orient='records')
                        except Exception as e:
                            logger.error(f"Error processing dates in _fetch_context_for_insight: {str(e)}")
                    
                    # Get channel comparison if relevant
                    if 'channel' in df.columns and channel:
                        # Get data for all channels in the same period
                        all_channels_df = self.db_manager.get_marketing_data(
                            start_date=start_date,
                            end_date=end_date,
                            office=office
                        )
                        
                        if not all_channels_df.empty:
                            channel_compare = all_channels_df.groupby('channel').agg({
                                'spend': 'sum',
                                'instructions': 'sum',
                                'leads': 'sum',
                                'roi': 'mean'
                            }).reset_index()
                            
                            context['channel_comparison'] = channel_compare.to_dict(orient='records')
                    
                    # Get office comparison if relevant
                    if 'office' in df.columns and office:
                        # Get data for all offices in the same period
                        all_offices_df = self.db_manager.get_marketing_data(
                            start_date=start_date,
                            end_date=end_date,
                            channel=channel
                        )
                        
                        if not all_offices_df.empty:
                            office_compare = all_offices_df.groupby('office').agg({
                                'spend': 'sum',
                                'instructions': 'sum',
                                'leads': 'sum',
                                'roi': 'mean'
                            }).reset_index()
                            
                            context['office_comparison'] = office_compare.to_dict(orient='records')
            
            # Get related insights
            related_insights = self.db_manager.get_insights(
                office=insight.get('office'),
                channel=insight.get('channel'),
                metric=insight.get('metric'),
                limit=5
            )
            
            if related_insights:
                # Remove the current insight from related insights
                related_insights = [ri for ri in related_insights if ri.get('id') != insight.get('id')]
                if related_insights:
                    context['related_insights'] = related_insights
            
            return context
        
        except Exception as e:
            logger.error(f"Error fetching context for insight: {str(e)}")
            return context
    
    def _fetch_question_context(self, question: str, start_date: str = None, end_date: str = None) -> Dict:
        """
        Fetch context data relevant to answering a user question.
        
        Args:
            question (str): The user's question
            start_date (str, optional): Start date for data context
            end_date (str, optional): End date for data context
            
        Returns:
            Dictionary of context data
        """
        context = {}
        
        if not self.db_manager:
            return context
        
        try:
            # Extract potential entities from the question
            question_lower = question.lower()
            
            # Check for mentions of specific metrics
            metrics_mentioned = []
            for metric in ['spend', 'roi', 'instructions', 'leads', 'cost per instruction', 'cost per lead', 'conversion rate']:
                if metric in question_lower:
                    metrics_mentioned.append(metric.replace(' ', '_'))
            
            # Check for mentions of channels, offices, or divisions
            channel_filter = None
            office_filter = None
            division_filter = None
            
            # Get all possible values to check against
            all_channels = self.db_manager.get_all_channels()
            for channel in all_channels:
                if channel and channel.lower() in question_lower:
                    channel_filter = channel
                    break
                    
            all_offices = self.db_manager.get_all_offices()
            for office in all_offices:
                if office and office.lower() in question_lower:
                    office_filter = office
                    break
                    
            all_divisions = self.db_manager.get_all_divisions()
            for division in all_divisions:
                if division and division.lower() in question_lower:
                    division_filter = division
                    break
            
            # Get data based on filters and date range
            df = self.db_manager.get_marketing_data(
                start_date=start_date,
                end_date=end_date,
                office=office_filter,
                channel=channel_filter,
                division=division_filter
            )
            
            if df.empty:
                context['data_available'] = False
                return context
                
            context['data_available'] = True
            
            # Add basic stats about the data with PROPER DATE HANDLING
            date_min = None
            date_max = None
            
            if 'date' in df.columns and not df.empty:
                try:
                    # Ensure dates are datetime objects
                    if not pd.api.types.is_datetime64_any_dtype(df['date']):
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    
                    # Now safely get min and max dates
                    if not df['date'].isna().all():
                        min_date = df['date'].min()
                        max_date = df['date'].max()
                        
                        if pd.notna(min_date):
                            date_min = min_date.strftime('%Y-%m-%d')
                        if pd.notna(max_date):
                            date_max = max_date.strftime('%Y-%m-%d')
                except Exception as e:
                    logger.error(f"Error processing dates: {str(e)}")
            
            context['data_summary'] = {
                'row_count': len(df),
                'date_range': {
                    'min': date_min,
                    'max': date_max
                },
                'filters_applied': {
                    'office': office_filter,
                    'channel': channel_filter,
                    'division': division_filter
                }
            }
            
            # Calculate overall metrics
            if 'spend' in df.columns:
                context['total_spend'] = float(df['spend'].sum())
                context['avg_spend'] = float(df['spend'].mean())
                
            if 'instructions' in df.columns:
                context['total_instructions'] = int(df['instructions'].sum())
                context['avg_instructions'] = float(df['instructions'].mean())
                
            if 'leads' in df.columns:
                context['total_leads'] = int(df['leads'].sum())
                context['avg_leads'] = float(df['leads'].mean())
                
            if 'roi' in df.columns:
                context['avg_roi'] = float(df['roi'].mean())
            
            # Calculate derived metrics
            if 'spend' in df.columns and 'instructions' in df.columns and df['instructions'].sum() > 0:
                context['cost_per_instruction'] = float(df['spend'].sum() / df['instructions'].sum())
                
            if 'spend' in df.columns and 'leads' in df.columns and df['leads'].sum() > 0:
                context['cost_per_lead'] = float(df['spend'].sum() / df['leads'].sum())
                
            if 'instructions' in df.columns and 'leads' in df.columns and df['leads'].sum() > 0:
                context['conversion_rate'] = float(df['instructions'].sum() / df['leads'].sum())
            
            # Add channel breakdown if available and relevant
            if 'channel' in df.columns and 'channel' in question_lower:
                channel_data = df.groupby('channel').agg({
                    'spend': 'sum',
                    'instructions': 'sum',
                    'leads': 'sum',
                    'roi': 'mean'
                }).reset_index()
                
                # Calculate derived metrics for each channel
                if 'spend' in channel_data.columns and 'instructions' in channel_data.columns:
                    channel_data['cost_per_instruction'] = channel_data.apply(
                        lambda row: float(row['spend'] / row['instructions']) if row['instructions'] > 0 else None, 
                        axis=1
                    )
                    
                if 'spend' in channel_data.columns and 'leads' in channel_data.columns:
                    channel_data['cost_per_lead'] = channel_data.apply(
                        lambda row: float(row['spend'] / row['leads']) if row['leads'] > 0 else None, 
                        axis=1
                    )
                
                context['channel_breakdown'] = channel_data.to_dict(orient='records')
            
            # Add office breakdown if available and relevant
            if 'office' in df.columns and 'office' in question_lower:
                office_data = df.groupby('office').agg({
                    'spend': 'sum',
                    'instructions': 'sum',
                    'leads': 'sum',
                    'roi': 'mean'
                }).reset_index()
                
                # Calculate derived metrics for each office
                if 'spend' in office_data.columns and 'instructions' in office_data.columns:
                    office_data['cost_per_instruction'] = office_data.apply(
                        lambda row: float(row['spend'] / row['instructions']) if row['instructions'] > 0 else None, 
                        axis=1
                    )
                
                context['office_breakdown'] = office_data.to_dict(orient='records')
            
            # Add time series analysis if time-related terms are in the question
            time_related_terms = ['trend', 'over time', 'monthly', 'weekly', 'daily', 'history', 'historical']
            if 'date' in df.columns and any(term in question_lower for term in time_related_terms):
                try:
                    # Ensure dates are datetime objects
                    if not pd.api.types.is_datetime64_any_dtype(df['date']):
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    
                    # Monthly aggregation - only if date conversion was successful
                    if not df['date'].isna().all():
                        df['month'] = df['date'].dt.strftime('%Y-%m')
                        monthly_data = df.groupby('month').agg({
                            'spend': 'sum',
                            'instructions': 'sum',
                            'leads': 'sum',
                            'roi': 'mean'
                        }).reset_index()
                        
                        context['monthly_trends'] = monthly_data.to_dict(orient='records')
                except Exception as e:
                    logger.error(f"Error creating time series: {str(e)}")
            
            # Add insights related to the question
            related_insights = self.db_manager.get_insights(
                start_date=start_date,
                end_date=end_date,
                office=office_filter,
                channel=channel_filter,
                limit=3
            )
            
            if related_insights:
                context['related_insights'] = related_insights
            
            return context
        
        except Exception as e:
            logger.error(f"Error fetching context for question: {str(e)}")
            return {'error': str(e)}
    
    def answer_question(self, question: str, context_start_date: str = None, context_end_date: str = None) -> str:
        """
        Generate a response to a user question about marketing data.
        
        Args:
            question (str): The user's question
            context_start_date (str, optional): Start date for data context
            context_end_date (str, optional): End date for data context
            
        Returns:
            str: AI-generated response to the question
        """
        if not self.has_openai:
            return self._fallback_answer(question, context_start_date, context_end_date)
        
        try:
            # Fetch relevant context data based on the question and date range
            context_data = self._fetch_question_context(
                question=question,
                start_date=context_start_date,
                end_date=context_end_date
            )
            
            # Create the prompt with the question and context
            prompt = f"""
            You are MAE (Marketing Analyst Expert), a real estate marketing analyst.
            
            The user asks: "{question}"
            
            Time period for analysis: {context_start_date or 'Not specified'} to {context_end_date or 'Not specified'}
            
            Context data:
            {json.dumps(context_data)}
            
            Answer the user's question based on the provided context data.
            If you don't have enough information to give a complete answer, acknowledge this and explain what additional data would be helpful.
            
            Keep your tone professional but conversational. Write in the first person as if you are MAE.
            Focus on providing practical, data-driven insights that address the user's question.
            """
            
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are MAE (Marketing Analyst Expert), a sophisticated AI analyst for a real estate marketing department."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except openai.RateLimitError as e:
            logger.warning("OpenAI rate limit or quota exceeded. Using fallback response.")
            return self._fallback_answer(question, context_start_date, context_end_date)
        except Exception as e:
            logger.error(f"Error answering question with OpenAI: {str(e)}")
            return f"I'm sorry, I encountered an error while processing your question: {str(e)}"
    
    def _fallback_answer(self, question: str, start_date: str = None, end_date: str = None) -> str:
        """Generate a fallback response when OpenAI API is unavailable."""
        # Get basic data for the period
        df = self.db_manager.get_marketing_data(
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            return "I don't have data for the specified period. Please try a different date range."
        
        # Basic question matching
        question_lower = question.lower()
        
        # Generate appropriate responses based on keywords in the question
        if any(word in question_lower for word in ["best", "top", "highest"]):
            if "channel" in question_lower:
                if "roi" in question_lower:
                    # Find channel with highest ROI
                    if 'channel' in df.columns and 'roi' in df.columns:
                        best_channel = df.groupby('channel')['roi'].mean().idxmax()
                        roi_value = df.groupby('channel')['roi'].mean().max()
                        return f"The channel with the highest ROI is {best_channel} with an average ROI of {roi_value:.2f}."
                
                if "instructions" in question_lower:
                    # Find channel with most instructions
                    if 'channel' in df.columns and 'instructions' in df.columns:
                        best_channel = df.groupby('channel')['instructions'].sum().idxmax()
                        instructions = df.groupby('channel')['instructions'].sum().max()
                        return f"The channel with the most instructions is {best_channel} with {int(instructions)} instructions."
        
        if "spend" in question_lower:
            total_spend = df['spend'].sum() if 'spend' in df.columns else 0
            return f"The total marketing spend for the selected period is £{total_spend:,.2f}."
        
        if "roi" in question_lower:
            avg_roi = df['roi'].mean() if 'roi' in df.columns else 0
            return f"The average ROI for the selected period is {avg_roi:.2f}."
        
        # Default response with basic stats
        response = [f"Here's a summary of marketing performance from {start_date} to {end_date}:"]
        
        if 'spend' in df.columns:
            total_spend = df['spend'].sum()
            response.append(f"- Total spend: £{total_spend:,.2f}")
        
        if 'instructions' in df.columns:
            total_instructions = df['instructions'].sum()
            response.append(f"- Total instructions: {int(total_instructions):,}")
        
        if 'leads' in df.columns:
            total_leads = df['leads'].sum()
            response.append(f"- Total leads: {int(total_leads):,}")
        
        if 'roi' in df.columns:
            avg_roi = df['roi'].mean

            def _fallback_answer(self, question: str, start_date: str = None, end_date: str = None) -> str:
                """Generate a fallback response when OpenAI API is unavailable."""
         # Get basic data for the period
        df = self.db_manager.get_marketing_data(
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            return "I don't have data for the specified period. Please try a different date range."
        
        # Basic question matching
        question_lower = question.lower()
        
        # Generate appropriate responses based on keywords in the question
        if any(word in question_lower for word in ["best", "top", "highest"]):
            if "channel" in question_lower:
                if "roi" in question_lower:
                    # Find channel with highest ROI
                    if 'channel' in df.columns and 'roi' in df.columns:
                        best_channel = df.groupby('channel')['roi'].mean().idxmax()
                        roi_value = df.groupby('channel')['roi'].mean().max()
                        return f"The channel with the highest ROI is {best_channel} with an average ROI of {roi_value:.2f}."
                
                if "instructions" in question_lower:
                    # Find channel with most instructions
                    if 'channel' in df.columns and 'instructions' in df.columns:
                        best_channel = df.groupby('channel')['instructions'].sum().idxmax()
                        instructions = df.groupby('channel')['instructions'].sum().max()
                        return f"The channel with the most instructions is {best_channel} with {int(instructions)} instructions."
        
        if "spend" in question_lower:
            total_spend = df['spend'].sum() if 'spend' in df.columns else 0
            return f"The total marketing spend for the selected period is £{total_spend:,.2f}."
        
        if "roi" in question_lower:
            avg_roi = df['roi'].mean() if 'roi' in df.columns else 0
            return f"The average ROI for the selected period is {avg_roi:.2f}."
        
        # Default response with basic stats
        response = [f"Here's a summary of marketing performance from {start_date} to {end_date}:"]
        
        if 'spend' in df.columns:
            total_spend = df['spend'].sum()
            response.append(f"- Total spend: £{total_spend:,.2f}")
        
        if 'instructions' in df.columns:
            total_instructions = df['instructions'].sum()
            response.append(f"- Total instructions: {int(total_instructions):,}")
        
        if 'leads' in df.columns:
            total_leads = df['leads'].sum()
            response.append(f"- Total leads: {int(total_leads):,}")
        
        if 'roi' in df.columns:
            avg_roi = df['roi'].mean()
            response.append(f"- Average ROI: {avg_roi:.2f}")
        
        response.append("\nFor more detailed analysis, please check the Dashboard or Analysis pages.")
        
        return "\n".join(response)
    
    def generate_weekly_report(self, end_date: str, days_back: int = 7) -> str:
        """
        Generate a weekly marketing performance report.
        
        Args:
            end_date (str): End date for the report period (YYYY-MM-DD)
            days_back (int): Number of days to look back
            
        Returns:
            str: Generated report text
        """
        if not self.has_openai:
            return "Weekly reports require OpenAI API access. Please configure your API key."
        
        try:
            # Calculate start date
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            start_date_obj = end_date_obj - timedelta(days=days_back)
            start_date = start_date_obj.strftime('%Y-%m-%d')
            
            # Get data for the period
            df = self.db_manager.get_marketing_data(
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                return f"No data available for the period {start_date} to {end_date}."
            
            # Get previous period data for comparison
            prev_end_date_obj = start_date_obj - timedelta(days=1)
            prev_start_date_obj = prev_end_date_obj - timedelta(days=days_back)
            
            prev_start_date = prev_start_date_obj.strftime('%Y-%m-%d')
            prev_end_date = prev_end_date_obj.strftime('%Y-%m-%d')
            
            prev_df = self.db_manager.get_marketing_data(
                start_date=prev_start_date,
                end_date=prev_end_date
            )
            
            # Prepare data for the report
            report_data = self._prepare_weekly_report_data(df, prev_df)
            
            # Create the prompt
            prompt = f"""
            You are MAE (Marketing Analyst Expert), a real estate marketing analyst.
            
            Generate a comprehensive weekly marketing performance report for the period {start_date} to {end_date}.
            
            Report data:
            {json.dumps(report_data)}
            
            The report should include:
            1. Executive Summary - Key metrics and changes from previous period
            2. Channel Performance - Analysis of each marketing channel
            3. Office Performance - Top and bottom performing offices
            4. Key Observations - Important trends or anomalies
            5. Recommendations - Actionable suggestions based on the data
            
            Format the report with Markdown headings and bullet points.
            Write in the first person as if you are MAE.
            Focus on providing actionable insights that can improve marketing performance.
            """
            
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are MAE (Marketing Analyst Expert), a sophisticated AI analyst for a real estate marketing department."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating weekly report with OpenAI: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    def _prepare_weekly_report_data(self, df: pd.DataFrame, prev_df: pd.DataFrame) -> Dict:
        """
        Prepare data for the weekly report.
        
        Args:
            df: Current period data
            prev_df: Previous period data
            
        Returns:
            Dictionary of report data
        """
        report_data = {
            'current_period': {
                'data_available': not df.empty
            },
            'previous_period': {
                'data_available': not prev_df.empty
            },
            'comparisons': {}
        }
        
        # Current period metrics
        if not df.empty:
            # Overall metrics
            report_data['current_period']['metrics'] = {
                'total_spend': float(df['spend'].sum()) if 'spend' in df.columns else 0,
                'total_instructions': int(df['instructions'].sum()) if 'instructions' in df.columns else 0,
                'total_leads': int(df['leads'].sum()) if 'leads' in df.columns else 0,
                'avg_roi': float(df['roi'].mean()) if 'roi' in df.columns else 0
            }
            
            # Calculate derived metrics
            if 'spend' in df.columns and 'instructions' in df.columns and df['instructions'].sum() > 0:
                report_data['current_period']['metrics']['cost_per_instruction'] = float(df['spend'].sum() / df['instructions'].sum())
                
            if 'spend' in df.columns and 'leads' in df.columns and df['leads'].sum() > 0:
                report_data['current_period']['metrics']['cost_per_lead'] = float(df['spend'].sum() / df['leads'].sum())
                
            if 'instructions' in df.columns and 'leads' in df.columns and df['leads'].sum() > 0:
                report_data['current_period']['metrics']['conversion_rate'] = float(df['instructions'].sum() / df['leads'].sum())
            
            # Channel breakdown
            if 'channel' in df.columns:
                channel_data = df.groupby('channel').agg({
                    'spend': 'sum',
                    'instructions': 'sum',
                    'leads': 'sum',
                    'roi': 'mean'
                }).reset_index()
                
                report_data['current_period']['channels'] = channel_data.to_dict(orient='records')
            
            # Office breakdown
            if 'office' in df.columns:
                office_data = df.groupby('office').agg({
                    'spend': 'sum',
                    'instructions': 'sum',
                    'leads': 'sum',
                    'roi': 'mean'
                }).reset_index()
                
                # Sort by instructions (descending)
                office_data = office_data.sort_values('instructions', ascending=False)
                
                report_data['current_period']['offices'] = office_data.to_dict(orient='records')
        
        # Previous period metrics
        if not prev_df.empty:
            # Overall metrics
            report_data['previous_period']['metrics'] = {
                'total_spend': float(prev_df['spend'].sum()) if 'spend' in prev_df.columns else 0,
                'total_instructions': int(prev_df['instructions'].sum()) if 'instructions' in prev_df.columns else 0,
                'total_leads': int(prev_df['leads'].sum()) if 'leads' in prev_df.columns else 0,
                'avg_roi': float(prev_df['roi'].mean()) if 'roi' in prev_df.columns else 0
            }
            
            # Calculate derived metrics
            if 'spend' in prev_df.columns and 'instructions' in prev_df.columns and prev_df['instructions'].sum() > 0:
                report_data['previous_period']['metrics']['cost_per_instruction'] = float(prev_df['spend'].sum() / prev_df['instructions'].sum())
                
            if 'spend' in prev_df.columns and 'leads' in prev_df.columns and prev_df['leads'].sum() > 0:
                report_data['previous_period']['metrics']['cost_per_lead'] = float(prev_df['spend'].sum() / prev_df['leads'].sum())
                
            if 'instructions' in prev_df.columns and 'leads' in prev_df.columns and prev_df['leads'].sum() > 0:
                report_data['previous_period']['metrics']['conversion_rate'] = float(prev_df['instructions'].sum() / prev_df['leads'].sum())
        
        # Generate comparisons
        if report_data['current_period']['data_available'] and report_data['previous_period']['data_available']:
            # Compare overall metrics
            current_metrics = report_data['current_period']['metrics']
            previous_metrics = report_data['previous_period']['metrics']
            
            metric_comparisons = {}
            for metric, value in current_metrics.items():
                if metric in previous_metrics:
                    prev_value = previous_metrics[metric]
                    
                    if prev_value != 0:
                        pct_change = ((value - prev_value) / prev_value) * 100
                    else:
                        pct_change = 0  # Avoid division by zero
                    
                    metric_comparisons[metric] = {
                        'current': value,
                        'previous': prev_value,
                        'change': value - prev_value,
                        'pct_change': pct_change
                    }
            
            report_data['comparisons']['metrics'] = metric_comparisons
            
            # Compare channels if available
            if 'channels' in report_data['current_period'] and 'channel' in df.columns and 'channel' in prev_df.columns:
                # Get channels from both periods
                current_channels = df['channel'].unique()
                prev_channels = prev_df['channel'].unique()
                all_channels = list(set(current_channels) | set(prev_channels))
                
                channel_comparisons = []
                
                for channel in all_channels:
                    current_channel_data = df[df['channel'] == channel] if channel in current_channels else pd.DataFrame()
                    prev_channel_data = prev_df[prev_df['channel'] == channel] if channel in prev_channels else pd.DataFrame()
                    
                    comparison = {'channel': channel}
                    
                    # Instructions comparison
                    if 'instructions' in df.columns and 'instructions' in prev_df.columns:
                        current_instructions = current_channel_data['instructions'].sum() if not current_channel_data.empty else 0
                        prev_instructions = prev_channel_data['instructions'].sum() if not prev_channel_data.empty else 0
                        
                        if prev_instructions != 0:
                            pct_change = ((current_instructions - prev_instructions) / prev_instructions) * 100
                        else:
                            pct_change = 0
                        
                        comparison['instructions'] = {
                            'current': int(current_instructions),
                            'previous': int(prev_instructions),
                            'change': int(current_instructions - prev_instructions),
                            'pct_change': pct_change
                        }
                    
                    # Spend comparison
                    if 'spend' in df.columns and 'spend' in prev_df.columns:
                        current_spend = current_channel_data['spend'].sum() if not current_channel_data.empty else 0
                        prev_spend = prev_channel_data['spend'].sum() if not prev_channel_data.empty else 0
                        
                        if prev_spend != 0:
                            pct_change = ((current_spend - prev_spend) / prev_spend) * 100
                        else:
                            pct_change = 0
                        
                        comparison['spend'] = {
                            'current': float(current_spend),
                            'previous': float(prev_spend),
                            'change': float(current_spend - prev_spend),
                            'pct_change': pct_change
                        }
                    
                    channel_comparisons.append(comparison)
                
                report_data['comparisons']['channels'] = channel_comparisons
        
        return report_data