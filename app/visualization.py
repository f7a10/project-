import pandas as pd
import numpy as np
import json
import logging
import traceback
from typing import Dict, Any, List, Optional, Union

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataVisualizer:
    """Class for generating visualizations from data files."""

    def __init__(self):
        """Initialize the DataVisualizer."""
        logger.info("DataVisualizer initialized")

    def generate_visualizations(self, data_files: List[str]) -> Dict[str, Any]:
        """
        Generate visualizations from a list of data files.

        Args:
            data_files: List of paths to data files

        Returns:
            Dict containing visualization data
        """
        try:
            if not data_files:
                logger.warning("No data files provided")
                return {"error": "No data files provided"}

            # Only process the first file for now (can be extended to handle multiple files)
            first_file = data_files[0]
            logger.info(f"Generating visualizations for file: {first_file}")

            # Load the dataframe
            df = self._load_file(first_file)
            if df is None:
                return {"error": "Failed to load file"}

            # Generate visualizations
            visualizations = {
                "line_chart": self.generate_line_chart(df),
                "bar_chart": self.generate_bar_chart(df),
                "pie_chart": self.generate_pie_chart(df),
                "histogram": self.generate_histogram(df),
                "scatter_plot": self.generate_scatter_plot(df),
                "heatmap": self.generate_heatmap(df)
            }

            logger.info(f"Generated {len(visualizations)} visualizations")
            return visualizations

        except Exception as e:
            logger.error(f"Error in generate_visualizations: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def _load_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load a file into a pandas DataFrame.

        Args:
            file_path: Path to the file

        Returns:
            Pandas DataFrame or None if loading fails
        """
        try:
            # Get file extension
            import os
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()

            # Load based on extension
            if ext == '.csv':
                try:
                    df = pd.read_csv(file_path)
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='latin1')
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif ext == '.json':
                try:
                    df = pd.read_json(file_path)
                except ValueError:
                    # Try loading as dictionary if array format fails
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        df = pd.DataFrame.from_dict(data, orient='index')
                    else:
                        df = pd.DataFrame(data)
            elif ext == '.txt' or ext == '.dat':
                # Try to infer delimiter
                df = pd.read_csv(file_path, sep=None, engine='python')
            else:
                logger.warning(f"Unsupported file type: {ext}")
                return None

            logger.info(f"Successfully loaded file with shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def generate_line_chart(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate data for a line chart from a dataframe.

        Args:
            df: Pandas DataFrame

        Returns:
            Dict with line chart data
        """
        try:
            # Check if we have datetime and numeric columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

            if not datetime_cols and not numeric_cols:
                # Try to convert potential date columns
                for col in df.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        try:
                            df[col] = pd.to_datetime(df[col])
                            datetime_cols.append(col)
                        except:
                            pass

            # If we have datetime and numeric columns, create a time series
            if datetime_cols and numeric_cols:
                date_col = datetime_cols[0]
                numeric_col = numeric_cols[0]

                # Sort by date and prepare data
                df_sorted = df.sort_values(by=date_col)

                # Ensure we don't have too many points (limit to 50)
                if len(df_sorted) > 50:
                    step = len(df_sorted) // 50 + 1
                    df_sorted = df_sorted.iloc[::step, :]

                # Format dates and extract values
                dates = df_sorted[date_col].dt.strftime('%Y-%m-%d').tolist()
                values = df_sorted[numeric_col].tolist()

                return {
                    "type": "line",
                    "labels": dates,
                    "datasets": [{
                        "label": numeric_col,
                        "data": values
                    }]
                }
            else:
                # If no datetime columns, use index as x-axis
                if numeric_cols:
                    numeric_col = numeric_cols[0]
                    values = df[numeric_col].tolist()
                    labels = list(range(len(values)))

                    return {
                        "type": "line",
                        "labels": labels,
                        "datasets": [{
                            "label": numeric_col,
                            "data": values
                        }]
                    }

                logger.warning("No suitable columns for line chart")
                return {"error": "No suitable columns for line chart"}

        except Exception as e:
            logger.error(f"Error generating line chart: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def generate_bar_chart(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate data for a bar chart from a dataframe.

        Args:
            df: Pandas DataFrame

        Returns:
            Dict with bar chart data
        """
        try:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

            if categorical_cols and numeric_cols:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]

                # Aggregate data by category
                grouped = df.groupby(cat_col)[num_col].mean().reset_index()

                # Sort by value and get top 10
                grouped = grouped.sort_values(by=num_col, ascending=False).head(10)

                return {
                    "type": "bar",
                    "labels": grouped[cat_col].tolist(),
                    "datasets": [{
                        "label": f"Average {num_col} by {cat_col}",
                        "data": grouped[num_col].tolist()
                    }]
                }
            else:
                logger.warning("No suitable columns for bar chart")
                return {"error": "No suitable columns for bar chart"}

        except Exception as e:
            logger.error(f"Error generating bar chart: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def generate_pie_chart(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate data for a pie chart from a dataframe.

        Args:
            df: Pandas DataFrame

        Returns:
            Dict with pie chart data
        """
        try:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            if categorical_cols:
                cat_col = categorical_cols[0]

                # Get value counts
                value_counts = df[cat_col].value_counts()

                # Limit to top 8 categories
                if len(value_counts) > 8:
                    other_count = value_counts[8:].sum()
                    value_counts = value_counts.iloc[:7]
                    value_counts['Other'] = other_count

                return {
                    "type": "pie",
                    "labels": value_counts.index.tolist(),
                    "datasets": [{
                        "data": value_counts.values.tolist()
                    }]
                }
            else:
                logger.warning("No suitable columns for pie chart")
                return {"error": "No suitable columns for pie chart"}

        except Exception as e:
            logger.error(f"Error generating pie chart: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def generate_histogram(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate data for a histogram from a dataframe.

        Args:
            df: Pandas DataFrame

        Returns:
            Dict with histogram data
        """
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

            if numeric_cols:
                num_col = numeric_cols[0]

                # Create histogram
                hist, bin_edges = np.histogram(df[num_col].dropna(), bins=10)

                # Create bin labels
                bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]

                return {
                    "type": "bar",
                    "labels": bin_labels,
                    "datasets": [{
                        "label": f"Distribution of {num_col}",
                        "data": hist.tolist()
                    }]
                }
            else:
                logger.warning("No suitable columns for histogram")
                return {"error": "No suitable columns for histogram"}

        except Exception as e:
            logger.error(f"Error generating histogram: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def generate_scatter_plot(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate data for a scatter plot from a dataframe.

        Args:
            df: Pandas DataFrame

        Returns:
            Dict with scatter plot data
        """
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

            if len(numeric_cols) >= 2:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]

                # Limit to 100 points for performance
                df_sample = df.sample(min(100, len(df))) if len(df) > 100 else df

                # Create dataset
                data = [{"x": float(x), "y": float(y)} for x, y in zip(df_sample[x_col], df_sample[y_col])]

                return {
                    "type": "scatter",
                    "datasets": [{
                        "label": f"{x_col} vs {y_col}",
                        "data": data
                    }]
                }
            else:
                logger.warning("Not enough numeric columns for scatter plot")
                return {"error": "Not enough numeric columns for scatter plot"}

        except Exception as e:
            logger.error(f"Error generating scatter plot: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def generate_heatmap(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate data for a correlation heatmap from a dataframe.

        Args:
            df: Pandas DataFrame

        Returns:
            Dict with heatmap data
        """
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

            if len(numeric_cols) >= 2:
                # Limit to 5 columns for readability
                selected_cols = numeric_cols[:5]

                # Calculate correlation matrix
                corr_matrix = df[selected_cols].corr().fillna(0)

                return {
                    "type": "heatmap",
                    "labels": selected_cols,
                    "data": corr_matrix.values.tolist()
                }
            else:
                logger.warning("Not enough numeric columns for heatmap")
                return {"error": "Not enough numeric columns for heatmap"}

        except Exception as e:
            logger.error(f"Error generating heatmap: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def summarize_csv(self, file_path: str) -> Dict[str, Any]:
        """
        Generate a summary of a CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            Dict with CSV summary
        """
        try:
            # Load the dataframe
            df = self._load_file(file_path)
            if df is None:
                return {"error": "Failed to load file"}

            # Generate summary
            summary = {
                "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "missing_data": {col: int(df[col].isna().sum()) for col in df.columns},
                "numeric_columns": {},
                "categorical_columns": {}
            }

            # Summarize numeric columns
            for col in df.select_dtypes(include=["number"]).columns:
                try:
                    col_summary = {
                        "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                        "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                        "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                        "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                        "std": float(df[col].std()) if not pd.isna(df[col].std()) else None
                    }
                    summary["numeric_columns"][col] = col_summary
                except Exception as e:
                    logger.warning(f"Error summarizing numeric column {col}: {e}")

            # Summarize categorical columns
            for col in df.select_dtypes(exclude=["number"]).columns:
                try:
                    value_counts = df[col].value_counts().head(5)
                    # Convert any non-string keys to strings
                    counts = {str(k): int(v) for k, v in value_counts.items()}
                    summary["categorical_columns"][col] = counts
                except Exception as e:
                    logger.warning(f"Error summarizing categorical column {col}: {e}")

            return summary

        except Exception as e:
            logger.error(f"Error summarizing CSV file: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def export_visualization_data(self, visualization_data: Dict[str, Any], format: str = 'json') -> Union[str, Dict]:
        """
        Export visualization data in the specified format.

        Args:
            visualization_data: Dictionary with visualization data
            format: Output format ('json' or 'dict')

        Returns:
            Formatted data as string or dict
        """
        try:
            if format.lower() == 'json':
                return json.dumps(visualization_data, indent=2)
            else:
                return visualization_data

        except Exception as e:
            logger.error(f"Error exporting visualization data: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}