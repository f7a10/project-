import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import traceback

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DataVisualizer:
    """
    A class to generate visualizations from data files.
    """

    def __init__(self):
        """Initialize the DataVisualizer."""
        pass

    def generate_visualizations(self, file_paths):
        """
        Generate a set of visualizations from the provided files.

        Args:
            file_paths (list): List of paths to data files

        Returns:
            dict: Dictionary containing chart data for frontend visualization
        """
        try:
            logger.info(f"Generating visualizations for {len(file_paths)} files")

            # Only process the first file for now
            if file_paths and os.path.exists(file_paths[0]):
                file_path = file_paths[0]

                # Get file extension
                _, ext = os.path.splitext(file_path)

                # Generate appropriate visualizations based on file type
                if ext.lower() in ['.csv', '.xlsx', '.xls']:
                    return self._generate_tabular_visualizations(file_path)
                elif ext.lower() == '.json':
                    return self._generate_json_visualizations(file_path)
                else:
                    logger.warning(f"Unsupported file type: {ext}")
                    return self._generate_default_visualizations()
            else:
                logger.warning("No valid files provided for visualization")
                return self._generate_default_visualizations()

        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            logger.error(traceback.format_exc())
            return self._generate_default_visualizations()

    def _generate_tabular_visualizations(self, file_path):
        """Generate visualizations for tabular data files (CSV, Excel)."""
        try:
            # Load data
            df = None
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)

            if df is None or df.empty:
                logger.warning(f"Failed to load data from {file_path} or data is empty")
                return self._generate_default_visualizations()

            # Basic data profiling
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            date_cols = []

            # Try to identify date columns
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]':
                    date_cols.append(col)
                elif df[col].dtype == 'object':
                    try:
                        # Check if column can be converted to datetime
                        pd.to_datetime(df[col], errors='raise')
                        date_cols.append(col)
                    except:
                        pass

            visualizations = {}

            # Generate time series if date columns exist
            if date_cols and numeric_cols:
                date_col = date_cols[0]
                numeric_col = numeric_cols[0]

                # Convert to datetime if not already
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

                # Sort by date
                df_sorted = df.sort_values(by=date_col)

                # Group by month if span is large enough
                if len(df_sorted) > 0 and (df_sorted[date_col].max() - df_sorted[date_col].min()).days > 60:
                    df_grouped = df_sorted.set_index(date_col).resample('M').mean().reset_index()
                    date_labels = df_grouped[date_col].dt.strftime('%b %Y').tolist()
                    values = df_grouped[numeric_col].tolist()
                else:
                    date_labels = df_sorted[date_col].dt.strftime('%Y-%m-%d').tolist()
                    values = df_sorted[numeric_col].tolist()

                visualizations['performance'] = {
                    'labels': date_labels[:20],  # Limit to first 20 points
                    'values': [float(v) if not pd.isna(v) else 0 for v in values[:20]]
                }
            elif numeric_cols:
                # Create a line chart with the first numeric column
                col = numeric_cols[0]
                labels = df.index.astype(str).tolist()[:20]  # Limit to first 20 points
                values = df[col].tolist()[:20]

                # Convert numpy types to native Python types for JSON serialization
                values = [float(v) if not pd.isna(v) else 0 for v in values]

                visualizations['performance'] = {
                    'labels': labels,
                    'values': values
                }

            # Generate pie chart for categorical data if exists
            if categorical_cols:
                cat_col = categorical_cols[0]
                category_counts = df[cat_col].value_counts().head(5)  # Top 5 categories

                visualizations['category'] = {
                    'labels': category_counts.index.tolist(),
                    'values': category_counts.values.tolist()
                }

            # Generate bar chart for another numeric column if exists
            if len(numeric_cols) >= 2:
                col = numeric_cols[1]
                data_sample = df[col].head(10)

                visualizations['comparison'] = {
                    'labels': data_sample.index.astype(str).tolist(),
                    'values': [float(v) if not pd.isna(v) else 0 for v in data_sample.tolist()]
                }
            elif len(numeric_cols) == 1:
                # Use the same column for comparison but with different visualization
                col = numeric_cols[0]
                bins = pd.cut(df[col].dropna(), 6)
                bin_counts = bins.value_counts().sort_index()

                visualizations['comparison'] = {
                    'labels': [str(b) for b in bin_counts.index.astype(str)],
                    'values': bin_counts.tolist()
                }

            # Generate correlation heatmap for numeric columns
            if len(numeric_cols) >= 3:
                corr_matrix = df[numeric_cols[:5]].corr().fillna(0).values.tolist()

                visualizations['correlation'] = {
                    'labels': numeric_cols[:5],  # Limit to first 5 for readability
                    'values': [row[:5] for row in corr_matrix[:5]]
                }
            elif len(numeric_cols) == 2:
                # For 2 columns, create a simple 2x2 correlation matrix
                corr_matrix = df[numeric_cols].corr().fillna(0).values.tolist()

                visualizations['correlation'] = {
                    'labels': numeric_cols,
                    'values': corr_matrix
                }

            return visualizations

        except Exception as e:
            logger.error(f"Error in tabular visualization generation: {str(e)}")
            logger.error(traceback.format_exc())
            return self._generate_default_visualizations()

    def _generate_json_visualizations(self, file_path):
        """Generate visualizations for JSON data files."""
        try:
            # Load JSON data
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Determine if it's an array or object
            if isinstance(data, list):
                # Convert to DataFrame if it's an array of objects
                if all(isinstance(item, dict) for item in data):
                    df = pd.DataFrame(data)
                    return self._analyze_dataframe(df)
                else:
                    # Simple array
                    return {
                        'performance': {
                            'labels': [str(i) for i in range(len(data))],
                            'values': data if all(isinstance(x, (int, float)) for x in data) else list(range(len(data)))
                        }
                    }
            elif isinstance(data, dict):
                # Extract key-value pairs for visualization
                keys = list(data.keys())
                values = list(data.values())

                # Filter numeric values
                numeric_items = [(k, v) for k, v in zip(keys, values) if isinstance(v, (int, float))]

                if numeric_items:
                    num_keys, num_values = zip(*numeric_items)
                    return {
                        'category': {
                            'labels': num_keys,
                            'values': num_values
                        }
                    }
                else:
                    return self._generate_default_visualizations()
            else:
                return self._generate_default_visualizations()

        except Exception as e:
            logger.error(f"Error in JSON visualization generation: {str(e)}")
            logger.error(traceback.format_exc())
            return self._generate_default_visualizations()

    def _analyze_dataframe(self, df):
        """Analyze a DataFrame and generate visualizations."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        visualizations = {}

        # Line chart for first numeric column
        if numeric_cols:
            col = numeric_cols[0]
            visualizations['performance'] = {
                'labels': df.index.astype(str).tolist()[:20],
                'values': [float(v) if not pd.isna(v) else 0 for v in df[col].tolist()[:20]]
            }

        # Pie chart for first categorical column
        if categorical_cols:
            cat_col = categorical_cols[0]
            category_counts = df[cat_col].value_counts().head(5)

            visualizations['category'] = {
                'labels': category_counts.index.tolist(),
                'values': category_counts.values.tolist()
            }

        # Bar chart for second numeric column if exists
        if len(numeric_cols) >= 2:
            col = numeric_cols[1]
            data_sample = df[col].head(10)

            visualizations['comparison'] = {
                'labels': data_sample.index.astype(str).tolist(),
                'values': [float(v) if not pd.isna(v) else 0 for v in data_sample.tolist()]
            }

        # Correlation matrix for numeric columns
        if len(numeric_cols) >= 3:
            corr_matrix = df[numeric_cols[:5]].corr().fillna(0).values.tolist()

            visualizations['correlation'] = {
                'labels': numeric_cols[:5],
                'values': [row[:5] for row in corr_matrix[:5]]
            }

        return visualizations

    def _generate_default_visualizations(self):
        """Generate default visualizations when data processing fails."""
        return {
            'performance': {
                'labels': ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"],
                'values': [65, 59, 80, 81, 56, 55, 40]
            },
            'category': {
                'labels': ["Category A", "Category B", "Category C", "Category D"],
                'values': [30, 50, 20, 10]
            },
            'comparison': {
                'labels': ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
                'values': [12, 19, 3, 5, 2, 3]
            },
            'correlation': {
                'labels': ["A", "B", "C", "D", "E"],
                'values': [
                    [1, 0.8, 0.6, 0.2, 0.1],
                    [0.8, 1, 0.7, 0.3, 0.2],
                    [0.6, 0.7, 1, 0.5, 0.4],
                    [0.2, 0.3, 0.5, 1, 0.9],
                    [0.1, 0.2, 0.4, 0.9, 1]
                ]
            }
        }

    def generate_line_chart(self, file_path):
        """Generate a line chart visualization for a specific file."""
        try:
            df = self._load_dataframe(file_path)
            if df is None or df.empty:
                return None

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                return None

            col = numeric_cols[0]
            return {
                'type': 'line',
                'labels': df.index.astype(str).tolist()[:50],
                'values': [float(v) if not pd.isna(v) else 0 for v in df[col].tolist()[:50]],
                'title': f'Line Chart: {col}'
            }
        except Exception as e:
            logger.error(f"Error generating line chart: {str(e)}")
            return None

    def generate_bar_chart(self, file_path):
        """Generate a bar chart visualization for a specific file."""
        try:
            df = self._load_dataframe(file_path)
            if df is None or df.empty:
                return None

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            if categorical_cols and numeric_cols:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]

                # Group by category and calculate mean of numeric column
                grouped = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(10)

                return {
                    'type': 'bar',
                    'labels': grouped.index.tolist(),
                    'values': [float(v) if not pd.isna(v) else 0 for v in grouped.values.tolist()],
                    'title': f'Average {num_col} by {cat_col}'
                }
            elif numeric_cols:
                col = numeric_cols[0]
                data_sample = df[col].head(15)

                return {
                    'type': 'bar',
                    'labels': data_sample.index.astype(str).tolist(),
                    'values': [float(v) if not pd.isna(v) else 0 for v in data_sample.tolist()],
                    'title': f'Bar Chart: {col}'
                }
            else:
                return None
        except Exception as e:
            logger.error(f"Error generating bar chart: {str(e)}")
            return None

    def generate_pie_chart(self, file_path):
        """Generate a pie chart visualization for a specific file."""
        try:
            df = self._load_dataframe(file_path)
            if df is None or df.empty:
                return None

            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            if categorical_cols:
                cat_col = categorical_cols[0]
                category_counts = df[cat_col].value_counts().head(7)  # Top 7 categories

                return {
                    'type': 'pie',
                    'labels': category_counts.index.tolist(),
                    'values': category_counts.values.tolist(),
                    'title': f'Distribution of {cat_col}'
                }
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    num_col = numeric_cols[0]
                    bins = pd.cut(df[num_col].dropna(), 5).value_counts().sort_index()

                    return {
                        'type': 'pie',
                        'labels': [str(bin) for bin in bins.index.astype(str)],
                        'values': bins.values.tolist(),
                        'title': f'Distribution of {num_col} (Binned)'
                    }
                else:
                    return None
        except Exception as e:
            logger.error(f"Error generating pie chart: {str(e)}")
            return None

    def _load_dataframe(self, file_path):
        """Load a file into a pandas DataFrame."""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                return None

            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                return pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    return pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Try to convert nested dict to dataframe
                    try:
                        return pd.DataFrame.from_dict(data, orient='index')
                    except:
                        return None
                else:
                    return None
            else:
                logger.warning(f"Unsupported file type: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading dataframe from {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return None