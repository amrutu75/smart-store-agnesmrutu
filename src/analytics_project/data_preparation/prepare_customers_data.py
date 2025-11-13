"""
scripts/data_preparation/prepare_customers_data.py

This script reads data from the data/raw folder, cleans the data,
and writes the cleaned version to the data/prepared folder.

Tasks:
- Remove duplicates
- Handle missing values
- Remove outliers
- Ensure consistent formatting
"""

# ==========================
# Imports
# ==========================
import pathlib
import pandas as pd
from analytics_project.utils_logger import logger


# ==========================
# Path Constants
# ==========================
PROJECT_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR: pathlib.Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: pathlib.Path = DATA_DIR / "raw"
PREPARED_DATA_DIR: pathlib.Path = DATA_DIR / "prepared"

# Ensure the directories exist
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
PREPARED_DATA_DIR.mkdir(exist_ok=True)


# ==========================
# Functions
# ==========================
def read_raw_data(file_name: str) -> pd.DataFrame:
    """Read raw data from CSV."""
    file_path = RAW_DATA_DIR / file_name

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    logger.info(f"Reading data from {file_path}")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataframe with {len(df)} rows and {len(df.columns)} columns")

        # Optional data profiling
        logger.debug("Column datatypes:\n%s", df.dtypes)
        logger.debug("Number of unique values per column:\n%s", df.nunique())

    except pd.errors.EmptyDataError:
        logger.error(f"File is empty: {file_path}")
        raise
    except pd.errors.ParserError:
        logger.error(f"Error parsing CSV file: {file_path}")
        raise

    return df


def save_prepared_data(df: pd.DataFrame, file_name: str) -> None:
    """Save the prepared DataFrame to a CSV file."""
    output_path = PREPARED_DATA_DIR / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved prepared data to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save prepared data to {output_path}: {e}")
        raise


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame."""
    initial_count = len(df)
    df_cleaned = df.drop_duplicates()
    logger.info(f"Removed {initial_count - len(df_cleaned)} duplicate rows")
    return df_cleaned


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the DataFrame."""
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)
        logger.info(f"Filled missing values in '{col}' with mean {mean_value:.2f}")

    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)
        logger.info(f"Filled missing values in '{col}' with mode '{mode_value}'")

    return df


def remove_outliers(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """Remove outliers using the Z-score method."""
    from scipy import stats

    logger.info(f"FUNCTION START: remove_outliers with dataframe shape {df.shape}")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    df_cleaned = df[(abs(stats.zscore(df[numeric_cols])) < z_threshold).all(axis=1)]

    removed_count = len(df) - len(df_cleaned)
    logger.info(f"Removed {removed_count} outlier rows; new shape {df_cleaned.shape}")
    logger.info("FUNCTION END: remove_outliers")

    return df_cleaned


# ==========================
# Main Function
# ==========================
def main() -> None:
    """Main function for processing data."""
    logger.info("==================================")
    logger.info("STARTING prepare_customers_data.py")
    logger.info("==================================")

    logger.info(f"Root         : {PROJECT_ROOT}")
    logger.info(f"data/raw     : {RAW_DATA_DIR}")
    logger.info(f"data/prepared: {PREPARED_DATA_DIR}")

    input_file = "customers_data.csv"
    output_file = "customer_data_prepared.csv"

    df = read_raw_data(input_file)
    original_shape = df.shape

    logger.info(f"Initial columns: {', '.join(df.columns)}")
    logger.info(f"Initial shape: {df.shape}")

    # Clean column names
    original_columns = df.columns.tolist()
    df.columns = df.columns.str.strip()
    changed_columns = [
        f"{old} -> {new}" for old, new in zip(original_columns, df.columns) if old != new
    ]
    if changed_columns:
        logger.info(f"Renamed columns: {', '.join(changed_columns)}")

    # Data cleaning steps
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = remove_outliers(df)

    save_prepared_data(df, output_file)

    logger.info("==================================")
    logger.info(f"Original shape: {original_shape}")
    logger.info(f"Cleaned shape : {df.shape}")
    logger.info("FINISHED prepare_customers_data.py")
    logger.info("==================================")


# ==========================
# Entry Point
# ==========================
if __name__ == "__main__":
    main()
