import json
import pandas as pd
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATABASE_PATH, DATE_FORMAT

class DatabaseManager:
    """
    Manages database operations for the MAE application.
    Handles storage and retrieval of marketing data, insights, and user interactions.
    """
    
    def __init__(self):
        """Initialize the database connection and create tables if they don't exist."""
        self.conn = self._create_connection()
        self._create_tables()
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a database connection to the SQLite database."""
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DATABASE_PATH))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _create_tables(self) -> None:
        """Create the necessary tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Table for storing uploaded data files metadata
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_date TEXT NOT NULL,
            data_start_date TEXT NOT NULL,
            data_end_date TEXT NOT NULL,
            record_count INTEGER NOT NULL,
            metadata TEXT
        )
        ''')
        
        # Table for storing the actual marketing data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS marketing_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            office TEXT NOT NULL,
            division TEXT NOT NULL,
            channel TEXT NOT NULL,
            spend REAL,
            roi REAL,
            instructions INTEGER,
            leads INTEGER,
            other_metrics TEXT,
            FOREIGN KEY (upload_id) REFERENCES uploads (id)
        )
        ''')
        
        # Table for storing insights generated by MAE
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_generated TEXT NOT NULL,
            type TEXT NOT NULL,
            period_start TEXT,
            period_end TEXT,
            office TEXT,
            division TEXT,
            channel TEXT,
            metric TEXT,
            description TEXT NOT NULL,
            recommendation TEXT,
            severity TEXT,
            status TEXT DEFAULT 'new'
        )
        ''')
        
        # Table for storing user interactions with MAE
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_query TEXT NOT NULL,
            response TEXT NOT NULL,
            related_insights TEXT,
            feedback TEXT
        )
        ''')
        
        self.conn.commit()
    
    def store_upload(self, filename: str, df: pd.DataFrame, metadata: Dict = None) -> int:
        """
        Store information about an uploaded file and its data.
        
        Args:
            filename: Name of the uploaded file
            df: The pandas DataFrame containing the uploaded data
            metadata: Additional metadata about the upload
            
        Returns:
            The ID of the inserted upload record
        """
        cursor = self.conn.cursor()
        
        # Extract date range from dataframe
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            start_date = df['date'].min().strftime(DATE_FORMAT)
            end_date = df['date'].max().strftime(DATE_FORMAT)
        else:
            # Default to current date if no date column
            today = datetime.now().strftime(DATE_FORMAT)
            start_date = today
            end_date = today
        
        # Store upload metadata
        cursor.execute('''
        INSERT INTO uploads (filename, upload_date, data_start_date, data_end_date, record_count, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            filename,
            datetime.now().strftime(DATE_FORMAT),
            start_date,
            end_date,
            len(df),
            json.dumps(metadata) if metadata else None
        ))
        
        upload_id = cursor.lastrowid
        self.conn.commit()
        
        return upload_id
    
    def store_marketing_data(self, upload_id: int, df: pd.DataFrame) -> None:
        """
        Store the actual marketing data from a DataFrame.
        
        Args:
            upload_id: The ID of the upload this data belongs to
            df: The pandas DataFrame containing the marketing data
        """
        # Ensure date is in the correct format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime(DATE_FORMAT)
        
        # Prepare columns that are standard in our schema
        standard_columns = ['date', 'office', 'division', 'channel', 'spend', 'roi', 'instructions', 'leads']
        
        # Convert DataFrame to list of tuples for insertion
        records = []
        for _, row in df.iterrows():
            # Extract standard columns, using None for missing ones
            record = [upload_id]
            for col in standard_columns:
                if col in df.columns:
                    record.append(row[col])
                else:
                    record.append(None)
            
            # Store any additional columns as JSON in other_metrics
            other_metrics = {}
            for col in df.columns:
                if col not in standard_columns:
                    other_metrics[col] = row[col]
            
            record.append(json.dumps(other_metrics) if other_metrics else None)
            records.append(tuple(record))
        
        # Insert all records
        placeholders = ','.join(['?'] * (len(standard_columns) + 2))  # +2 for upload_id and other_metrics
        cursor = self.conn.cursor()
        cursor.executemany(
            f'''
            INSERT INTO marketing_data 
            (upload_id, {', '.join(standard_columns)}, other_metrics)
            VALUES ({placeholders})
            ''', 
            records
        )
        
        self.conn.commit()
    
    def store_insight(self, insight_data: Dict) -> int:
        """
        Store an insight generated by MAE.
        
        Args:
            insight_data: Dictionary containing insight information
            
        Returns:
            The ID of the inserted insight
        """
        required_fields = ['type', 'description']
        for field in required_fields:
            if field not in insight_data:
                raise ValueError(f"Required field '{field}' missing from insight data")
        
        # Ensure date_generated is present
        if 'date_generated' not in insight_data:
            insight_data['date_generated'] = datetime.now().strftime(DATE_FORMAT)
        
        # Build dynamic query based on provided fields
        fields = list(insight_data.keys())
        placeholders = ','.join(['?'] * len(fields))
        values = [insight_data[field] for field in fields]
        
        cursor = self.conn.cursor()
        cursor.execute(
            f'''
            INSERT INTO insights ({', '.join(fields)})
            VALUES ({placeholders})
            ''', 
            values
        )
        
        insight_id = cursor.lastrowid
        self.conn.commit()
        
        return insight_id
    
    def store_interaction(self, user_query: str, response: str, 
                         related_insights: List[int] = None) -> int:
        """
        Store a user interaction with MAE.
        
        Args:
            user_query: The query from the user
            response: MAE's response
            related_insights: List of insight IDs related to this interaction
            
        Returns:
            The ID of the inserted interaction
        """
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO interactions (timestamp, user_query, response, related_insights)
        VALUES (?, ?, ?, ?)
        ''', (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            user_query,
            response,
            json.dumps(related_insights) if related_insights else None
        ))
        
        interaction_id = cursor.lastrowid
        self.conn.commit()
        
        return interaction_id
    
    def get_marketing_data(self, 
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          office: Optional[str] = None,
                          division: Optional[str] = None,
                          channel: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve marketing data with optional filters.
        
        Args:
            start_date: Filter data starting from this date (inclusive)
            end_date: Filter data up to this date (inclusive)
            office: Filter by office name
            division: Filter by division
            channel: Filter by marketing channel
            
        Returns:
            DataFrame containing the filtered marketing data
        """
        query = "SELECT * FROM marketing_data WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        if office:
            query += " AND office = ?"
            params.append(office)
        
        if division:
            query += " AND division = ?"
            params.append(division)
        
        if channel:
            query += " AND channel = ?"
            params.append(channel)
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        
        # Convert results to DataFrame
        results = cursor.fetchall()
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame([dict(row) for row in results])
        
        # Parse other_metrics JSON column into DataFrame columns
        if 'other_metrics' in df.columns and not df['other_metrics'].isna().all():
            for idx, row in df.iterrows():
                if pd.notna(row['other_metrics']):
                    other_metrics = json.loads(row['other_metrics'])
                    for key, value in other_metrics.items():
                        df.loc[idx, key] = value
        
        return df
    
    def get_insights(self, 
                    insight_type: Optional[str] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    office: Optional[str] = None,
                    channel: Optional[str] = None,
                    metric: Optional[str] = None,
                    status: Optional[str] = None,
                    limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieve insights with optional filters.
        
        Args:
            insight_type: Filter by insight type
            start_date: Filter insights generated after this date
            end_date: Filter insights generated before this date
            office: Filter by office
            channel: Filter by channel
            metric: Filter by metric
            status: Filter by status (e.g., 'new', 'acknowledged')
            limit: Maximum number of insights to return
            
        Returns:
            List of dictionaries containing the filtered insights
        """
        query = "SELECT * FROM insights WHERE 1=1"
        params = []
        
        if insight_type:
            query += " AND type = ?"
            params.append(insight_type)
        
        if start_date:
            query += " AND date_generated >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date_generated <= ?"
            params.append(end_date)
        
        if office:
            query += " AND office = ?"
            params.append(office)
        
        if channel:
            query += " AND channel = ?"
            params.append(channel)
        
        if metric:
            query += " AND metric = ?"
            params.append(metric)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY date_generated DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        
        results = cursor.fetchall()
        return [dict(row) for row in results]
    
    def get_recent_interactions(self, limit: int = 5) -> List[Dict]:
        """
        Retrieve the most recent user interactions.
        
        Args:
            limit: Maximum number of interactions to return
            
        Returns:
            List of dictionaries containing the recent interactions
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM interactions ORDER BY timestamp DESC LIMIT ?", 
            (limit,)
        )
        
        results = cursor.fetchall()
        return [dict(row) for row in results]
    
    def get_upload_history(self, limit: int = 10) -> List[Dict]:
        """
        Retrieve the history of uploads.
        
        Args:
            limit: Maximum number of uploads to return
            
        Returns:
            List of dictionaries containing the upload history
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM uploads ORDER BY upload_date DESC LIMIT ?", 
            (limit,)
        )
        
        results = cursor.fetchall()
        return [dict(row) for row in results]
    
    def get_all_offices(self) -> List[str]:
        """Get a list of all offices in the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT office FROM marketing_data ORDER BY office")
        return [row[0] for row in cursor.fetchall()]
    
    def get_all_divisions(self) -> List[str]:
        """Get a list of all divisions in the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT division FROM marketing_data ORDER BY division")
        return [row[0] for row in cursor.fetchall()]
    
    def get_all_channels(self) -> List[str]:
        """Get a list of all marketing channels in the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT channel FROM marketing_data ORDER BY channel")
        return [row[0] for row in cursor.fetchall()]
    
    def update_insight_status(self, insight_id: int, status: str) -> None:
        """
        Update the status of an insight.
        
        Args:
            insight_id: The ID of the insight to update
            status: The new status (e.g., 'acknowledged', 'resolved')
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE insights SET status = ? WHERE id = ?",
            (status, insight_id)
        )
        self.conn.commit()
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
