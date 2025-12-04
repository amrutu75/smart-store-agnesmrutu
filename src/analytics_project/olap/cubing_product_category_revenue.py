"""Module for creating a multidimensional OLAP cube from sales data."""

import pathlib
import sqlite3
import pandas as pd
from analytics_project.utils_logger import logger

# -------------------------------
# Global constants for paths
# -------------------------------
THIS_DIR: pathlib.Path = pathlib.Path(__file__).resolve().parent
DW_DIR: pathlib.Path = THIS_DIR
PACKAGE_DIR: pathlib.Path = DW_DIR.parent
SRC_DIR: pathlib.Path = PACKAGE_DIR.parent
PROJECT_ROOT_DIR: pathlib.Path = SRC_DIR.parent

DATA_DIR: pathlib.Path = PROJECT_ROOT_DIR / "data"
WAREHOUSE_DIR: pathlib.Path = DATA_DIR / "warehouse"
DB_PATH: pathlib.Path = WAREHOUSE_DIR / "smart_sales.db"

OLAP_OUTPUT_DIR: pathlib.Path = DATA_DIR / "olap_cubing_outputs"
OLAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------
# Schema check
# -------------------------------
def check_schema(conn: sqlite3.Connection, required_tables: list) -> bool:
    """Verify that required tables exist in the SQLite database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    existing_tables = {row[0] for row in cursor.fetchall()}

    missing = [t for t in required_tables if t not in existing_tables]
    if missing:
        logger.error(f"Missing required tables in warehouse: {missing}")
        return False

    logger.info(f"All required tables found: {required_tables}")
    return True


# -------------------------------
# Data ingestion
# -------------------------------
def ingest_sales_data_from_dw() -> pd.DataFrame:
    """Fetch sales, customer, and product data from SQLite warehouse."""
    try:
        conn = sqlite3.connect(DB_PATH)

        # Check schema before running query
        required_tables = ["sale", "customer", "product"]
        if not check_schema(conn, required_tables):
            conn.close()
            raise RuntimeError("Warehouse schema check failed. Required tables missing.")

        query = """
        SELECT
            s.sale_id,
            s.customer_id,
            s.product_id,
            s.date,
            s.sale_amount,
            c.region,
            p.category,
            p.Supplier AS supplier
        FROM sale AS s
        LEFT JOIN customer AS c
            ON CAST(s.customer_id AS INTEGER) = c.customer_id
        LEFT JOIN product AS p
            ON s.product_id = p.product_id
        """

        df = pd.read_sql_query(query, conn)
        conn.close()
        logger.info("Sales, customer, and product data successfully loaded from smart_sales.db.")
        return df

    except Exception as e:
        logger.error(f"Error loading data from warehouse: {e}")
        raise


# -------------------------------
# Global cleaning
# -------------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean sale_amount and date columns globally."""
    # Ensure sale_amount is numeric
    df['sale_amount'] = pd.to_numeric(df['sale_amount'], errors='coerce')
    invalid_sales = df['sale_amount'].isna().sum()
    if invalid_sales > 0:
        logger.warning(f"{invalid_sales} rows had invalid sale_amount values and were dropped.")
    df = df.dropna(subset=['sale_amount'])

    # Drop the known bad date explicitly
    bad_date_count = (df['date'] == "2023-01-13").sum()
    if bad_date_count > 0:
        logger.warning(f"Dropping {bad_date_count} rows with known bad date '2023-01-13'.")
        df = df[df['date'] != "2023-01-13"]

    # Parse majority format mm/dd/yyyy
    df['ParsedDate'] = pd.to_datetime(df['date'], format="%m/%d/%Y", errors="coerce")

    # Fallback for any remaining invalids
    still_invalid = df['ParsedDate'].isna().sum()
    if still_invalid > 0:
        logger.warning(
            f"{still_invalid} rows could not be parsed with %m/%d/%Y, retrying with mixed inference."
        )
        df.loc[df['ParsedDate'].isna(), 'ParsedDate'] = pd.to_datetime(
            df.loc[df['ParsedDate'].isna(), 'date'], format="mixed", errors="coerce"
        )

    # Drop rows that remain invalid
    final_invalid = df['ParsedDate'].isna().sum()
    if final_invalid > 0:
        logger.warning(f"{final_invalid} rows had invalid date formats and were dropped.")
        df = df.dropna(subset=['ParsedDate'])

    return df


# -------------------------------
# OLAP cube creation
# -------------------------------
def create_olap_cube(df: pd.DataFrame, dimensions: list, metrics: dict) -> pd.DataFrame:
    """Create an OLAP cube by aggregating data across multiple dimensions."""
    try:
        grouped = df.groupby(dimensions).agg(metrics).reset_index()
        logger.info("Multidimensional OLAP cube created.")
        return grouped
    except Exception as e:
        logger.error(f"Error creating OLAP cube: {e}")
        raise


# -------------------------------
# Helper analytics functions
# -------------------------------
def sales_growth_by_category(df: pd.DataFrame) -> pd.DataFrame:
    df['Year'] = df['ParsedDate'].dt.year
    category_sales = df.groupby(['Year', 'category'])['sale_amount'].sum().reset_index()
    category_sales['Sales_Growth'] = (
        category_sales.groupby('category')['sale_amount'].pct_change().fillna(0)
    )
    return category_sales


def sales_by_region(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby('region')['sale_amount'].sum().reset_index()


def top_selling_products(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    product_sales = df.groupby('product_id')['sale_amount'].sum().reset_index()
    return product_sales.nlargest(top_n, 'sale_amount')


def sales_growth_by_date(df: pd.DataFrame) -> pd.DataFrame:
    df['Date'] = df['ParsedDate'].dt.date
    date_sales = df.groupby('Date')['sale_amount'].sum().reset_index()
    date_sales['Sales_Growth'] = date_sales['sale_amount'].pct_change().fillna(0)
    return date_sales


# -------------------------------
# Debug schema
# -------------------------------
def debug_schema_and_counts():
    """Log schema and row counts for warehouse tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for table in ["sale", "customer", "product"]:
        cursor.execute(f"PRAGMA table_info({table});")
        logger.info(f"Schema for {table}: {cursor.fetchall()}")

        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        logger.info(f"Row count for {table}: {cursor.fetchone()[0]}")

    conn.close()


# -------------------------------
# Main execution
# -------------------------------
def main():
    """Main function to execute OLAP cubing process."""
    logger.info("Starting OLAP Cubing process...")

    # Debug schema first
    debug_schema_and_counts()

    # Ingest data
    df = ingest_sales_data_from_dw()
    if df.empty:
        logger.error("No sales data available to process. Exiting.")
        return

    # Clean globally
    df = clean_dataframe(df)

    # Define cube dimensions and metrics
    dimensions = ['ParsedDate', 'region', 'category']
    metrics = {'sale_amount': 'sum', 'sale_id': 'count'}

    # Create OLAP cube
    cube = create_olap_cube(df, dimensions, metrics)
    cube.to_csv(OLAP_OUTPUT_DIR / "multidimensional_olap_cube.csv", index=False)
    logger.info("Saved multidimensional_olap_cube.csv")

    # Generate helper outputs
    sales_growth_by_category(df).to_csv(
        OLAP_OUTPUT_DIR / "sales_growth_by_category.csv", index=False
    )
    logger.info("Saved sales_growth_by_category.csv")

    sales_by_region(df).to_csv(OLAP_OUTPUT_DIR / "sales_by_region.csv", index=False)
    logger.info("Saved sales_by_region.csv")

    top_selling_products(df).to_csv(OLAP_OUTPUT_DIR / "top_selling_products.csv", index=False)
    logger.info("Saved top_selling_products.csv")

    sales_growth_by_date(df).to_csv(OLAP_OUTPUT_DIR / "sales_growth_by_date.csv", index=False)
    logger.info("Saved sales_growth_by_date.csv")

    logger.info("OLAP Cubing process completed successfully.")
    logger.info(f"Outputs saved to {OLAP_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
