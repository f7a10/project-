import os
import pandas as pd
import numpy as np
import logging
import json
import traceback
from typing import List, Dict, Any, Union, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """Class for processing data files."""

    def __init__(self, upload_folder='uploads'):
        """Initialize the DataProcessor with the upload folder path."""
        self.upload_folder = upload_folder
        logger.info(f"DataProcessor initialized with upload folder: {upload_folder}")

        # Ensure upload folder exists
        os.makedirs(upload_folder, exist_ok=True)

    def process_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Process a list of data files and return a summary.

        Args:
            file_paths: List of paths to data files

        Returns:
            Dictionary with file summaries
        """
        results = {}

        if not file_paths:
            logger.warning("No file paths provided to process_files")
            return {"error": "No files provided", "success": False}

        logger.debug(f"Processing {len(file_paths)} files: {file_paths}")

        for file_path in file_paths:
            try:
                logger.info(f"Processing file: {file_path}")

                if not os.path.exists(file_path):
                    logger.error(f"File does not exist: {file_path}")
                    filename = os.path.basename(file_path)
                    results[filename] = {"error": "File does not exist"}
                    continue

                # Get file extension
                _, ext = os.path.splitext(file_path)
                ext = ext.lower()

                # Load data based on extension
                try:
                    df = self._load_file(file_path, ext)
                    if df is None:
                        filename = os.path.basename(file_path)
                        results[filename] = {"error": f"Failed to load file with extension {ext}"}
                        continue
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {str(e)}")
                    logger.error(traceback.format_exc())
                    filename = os.path.basename(file_path)
                    results[filename] = {"error": f"Error loading file: {str(e)}"}
                    continue

                # Clean the data
                df = self._clean_data(df)

                # Generate summary
                file_summary = self._summarize_dataframe(df)

                # Add to results
                filename = os.path.basename(file_path)
                results[filename] = file_summary
                logger.info(f"Successfully processed {filename}")

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                logger.error(traceback.format_exc())
                # Add error information to results
                filename = os.path.basename(file_path)
                results[filename] = {"error": str(e)}

        # Check if any files were successfully processed
        if not any("error" not in v for v in results.values()):
            logger.warning("All files failed to process")
            results["success"] = False
        else:
            results["success"] = True

        return results

    def _load_file(self, file_path: str, ext: str) -> Optional[pd.DataFrame]:
        """
        Load a file into a pandas DataFrame based on extension.

        Args:
            file_path: Path to the file
            ext: File extension

        Returns:
            Pandas DataFrame or None if loading fails
        """
        try:
            if ext == '.csv':
                # Try different encodings if one fails
                try:
                    df = pd.read_csv(file_path)
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='latin1')
                logger.info(f"Loaded CSV file with shape: {df.shape}")

            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                logger.info(f"Loaded Excel file with shape: {df.shape}")

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
                logger.info(f"Loaded JSON file with shape: {df.shape}")

            elif ext == '.txt' or ext == '.dat':
                # Try to infer delimiter
                df = pd.read_csv(file_path, sep=None, engine='python')
                logger.info(f"Loaded text file with shape: {df.shape}")

            else:
                logger.warning(f"Unsupported file type: {ext}")
                return None

            return df

        except Exception as e:
            logger.error(f"Error in _load_file for {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the dataframe.

        Args:
            df: Pandas DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        try:
            # Make a copy to avoid modifying the original
            df_clean = df.copy()

            # Drop rows that are all NaN
            df_clean = df_clean.dropna(how='all')

            # Convert object columns that are numeric to numeric type
            for col in df_clean.columns:
                if df_clean[col].dtype == 'object':
                    try:
                        # Try to convert object columns to numeric if they contain numbers
                        numeric_values = pd.to_numeric(df_clean[col], errors='coerce')
                        # If more than 80% of values are numeric, convert the column
                        if numeric_values.notna().sum() / len(numeric_values) > 0.8:
                            df_clean[col] = numeric_values
                    except:
                        pass

            return df_clean

        except Exception as e:
            logger.error(f"Error in _clean_data: {str(e)}")
            logger.error(traceback.format_exc())
            return df  # Return original if cleaning fails

    def _summarize_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of a dataframe.

        Args:
            df: Pandas DataFrame to summarize

        Returns:
            Dictionary with summary statistics
        """
        try:
            # Basic info
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
            logger.error(f"Error summarizing dataframe: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def save_uploaded_file(self, file, session_id: str) -> str:
        """
        Save an uploaded file to the upload folder.

        Args:
            file: The file object from the request
            session_id: The session ID to organize files

        Returns:
            Path to the saved file
        """
        try:
            # Create session directory if it doesn't exist
            session_dir = os.path.join(self.upload_folder, session_id)
            os.makedirs(session_dir, exist_ok=True)

            # Generate a safe filename
            filename = file.filename
            safe_filename = os.path.join(session_dir, filename)

            # Save the file
            file.save(safe_filename)
            logger.info(f"Saved file to {safe_filename}")

            return safe_filename

        except Exception as e:
            logger.error(f"Error saving uploaded file: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_uploaded_files(self, session_id: str) -> List[str]:
        """
        Get a list of uploaded files for a session.

        Args:
            session_id: The session ID

        Returns:
            List of file paths
        """
        try:
            session_dir = os.path.join(self.upload_folder, session_id)
            if not os.path.exists(session_dir):
                logger.warning(f"Session directory does not exist: {session_dir}")
                return []

            # Get all files in the session directory
            files = [os.path.join(session_dir, f) for f in os.listdir(session_dir)
                    if os.path.isfile(os.path.join(session_dir, f))]

            return files

        except Exception as e:
            logger.error(f"Error getting uploaded files: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def delete_uploaded_file(self, filename: str, session_id: str) -> bool:
        """
        Delete an uploaded file.

        Args:
            filename: Name of the file to delete
            session_id: The session ID

        Returns:
            True if successfully deleted, False otherwise
        """
        try:
            file_path = os.path.join(self.upload_folder, session_id, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
                return True
            else:
                logger.warning(f"File does not exist: {file_path}")
                return False

        except Exception as e:
            logger.error(f"Error deleting uploaded file: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def generate_sample_data(self, data_type: str = 'sales') -> pd.DataFrame:
        """
        Generate sample data for demo purposes.

        Args:
            data_type: Type of sample data to generate

        Returns:
            DataFrame with sample data
        """
        try:
            if data_type == 'sales':
                # Generate random sales data
                np.random.seed(42)
                dates = pd.date_range(start='2023-01-01', periods=100)
                data = {
                    'date': dates,
                    'product': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], size=100),
                    'region': np.random.choice(['North', 'South', 'East', 'West'], size=100),
                    'sales': np.random.randint(100, 1000, size=100),
                    'units': np.random.randint(1, 50, size=100),
                }
                return pd.DataFrame(data)
            else:
                logger.warning(f"Unknown data type: {data_type}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error generating sample data: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()