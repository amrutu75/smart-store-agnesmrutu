"""ETL script to load prepared data into the data warehouse (SQLite database).

File: src/analytics_project/dw/etl_to_dw.py

This file assumes the following structure (yours may vary):

project_root/
│
├─ data/
│   ├─ raw/
│   ├─ prepared/
│   └─ warehouse/
│
└─ src/
    └─ analytics_project/
        ├─ data_preparation/
        ├─ dw/
        ├─ analytics/
        └─ utils_logger.py

By switching to a modern src/ layout and using __init__.py files,
we no longer need any sys.path modifications.

Remember to put __init__.py files (empty is fine) in each folder to make them packages.

NOTE on column names: This example uses inconsistent naming conventions for column names in the cleaned data.
A good business intelligence project would standardize these during data preparation.
Your names should be more standard after cleaning and pre-processing the data.

Database names generally follow snake_case conventions for SQL compatibility.
"snake_case" = all lowercase with underscores between words.
"""

# Imports at the top

import pathlib
import sqlite3
import sys

import pandas as pd

#: Project root is two levels up from this file
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

# Import the logger from utils_logger
from analytics_project.utils_logger import logger

# Global constants for paths and key directories

THIS_DIR: pathlib.Path = pathlib.Path(__file__).resolve().parent
DW_DIR: pathlib.Path = THIS_DIR  # src/analytics_project/dw
PACKAGE_DIR: pathlib.Path = DW_DIR.parent  # src/analytics_project/
SRC_DIR: pathlib.Path = PACKAGE_DIR.parent  # src/
PROJECT_ROOT_DIR: pathlib.Path = SRC_DIR.parent  # smart-store-agnesmrutu/

# Data directories
DATA_DIR: pathlib.Path = PROJECT_ROOT_DIR / "data"
RAW_DATA_DIR: pathlib.Path = DATA_DIR / "raw"
CLEAN_DATA_DIR: pathlib.Path = DATA_DIR / "prepared"
WAREHOUSE_DIR: pathlib.Path = DATA_DIR / "warehouse"

# Warehouse database location (SQLite)
DB_PATH: pathlib.Path = WAREHOUSE_DIR / "smart_sales.db"

# Recommended - log paths and key directories for debugging

logger.info(f"THIS_DIR:            {THIS_DIR}")
logger.info(f"DW_DIR:              {DW_DIR}")
logger.info(f"PACKAGE_DIR:         {PACKAGE_DIR}")
logger.info(f"SRC_DIR:             {SRC_DIR}")
logger.info(f"PROJECT_ROOT_DIR:    {PROJECT_ROOT_DIR}")

logger.info(f"DATA_DIR:            {DATA_DIR}")
logger.info(f"RAW_DATA_DIR:        {RAW_DATA_DIR}")
logger.info(f"CLEAN_DATA_DIR:      {CLEAN_DATA_DIR}")
logger.info(f"WAREHOUSE_DIR:       {WAREHOUSE_DIR}")
logger.info(f"DB_PATH:             {DB_PATH}")


def create_schema(cursor: sqlite3.Cursor) -> None:
    """Create tables in the data warehouse if they don't exist."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer (
            customer_id INTEGER PRIMARY KEY,
            name TEXT,
            region TEXT,
            join_date TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS product (
            product_id TEXT PRIMARY KEY,
            product_name TEXT,
            category TEXT,
            unit_price REAL,
            stockQuantity INTEGER,
            Supplier TEXT,
            discount_amount REAL
        )
    """)

    # FIXED: changed foreign key table names and column to match definitions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sale (
            sale_id INTEGER PRIMARY KEY,       -- from TransactionID
            date TEXT,                         -- from SaleDate (ISO 8601 format)
            customer_id TEXT,                  -- from CustomerID
            product_id TEXT,                   -- from ProductID
            state TEXT,                        -- from State
            discount_amount REAL,              -- from DiscountAmount
            sale_amount REAL,                  -- FIXED: renamed from sales_amount
            FOREIGN KEY (customer_id) REFERENCES customer(customer_id),
            FOREIGN KEY (product_id) REFERENCES product(product_id)
        )
    """)


# delete existing records function
def delete_existing_records(cursor: sqlite3.Cursor) -> None:
    """Delete all records from all tables in the warehouse."""
    logger.info("Deleting existing records from all tables.")
    cursor.execute("DELETE FROM sale")
    cursor.execute("DELETE FROM product")
    cursor.execute("DELETE FROM customer")


# Insert data functions for each table


def insert_customers(customers_df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    """Insert customer data into the customer table."""
    logger.info("Checking for duplicate customer IDs before insertion.")
    customers_df = customers_df.drop_duplicates(subset=["customer_id"])

    # Keep only columns that exist in the database
    db_columns = ["customer_id", "name", "region", "join_date"]
    customers_df = customers_df[[col for col in db_columns if col in customers_df.columns]]

    logger.info(f"Inserting {len(customers_df)} unique customer rows.")
    customers_df.to_sql("customer", conn, if_exists="append", index=False)


def insert_products(products_df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    """Insert product data into the product table."""
    logger.info("Checking for duplicate product IDs before insertion.")
    products_df = products_df.drop_duplicates(subset=["product_id"])

    # Keep only columns that exist in the database
    db_columns = [
        "product_id",
        "product_name",
        "category",
        "unit_price",
        "stockQuantity",
        "Supplier",
        "discount_amount",
    ]
    products_df = products_df[[col for col in db_columns if col in products_df.columns]]

    logger.info(f"Inserting {len(products_df)} unique product rows.")
    products_df.to_sql("product", conn, if_exists="append", index=False)


def insert_sales(sales_df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    """Insert sales data into the sale table."""
    logger.info("Checking for duplicate sale IDs before insertion.")
    sales_df = sales_df.drop_duplicates(subset=["sale_id"])

    # Keep only columns that exist in the database
    db_columns = [
        "sale_id",
        "date",
        "customer_id",
        "product_id",
        "state",
        "discount_amount",
        "sale_amount",
    ]
    sales_df = sales_df[[col for col in db_columns if col in sales_df.columns]]

    logger.info(f"Inserting {len(sales_df)} unique sale rows.")
    sales_df.to_sql("sale", conn, if_exists="append", index=False)


def load_data_to_db() -> None:
    """Load clean data into the data warehouse."""
    logger.info("Starting ETL: loading clean data into the warehouse.")

    # Make sure the warehouse directory exists
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)

    # If an old database exists, remove and recreate with the latest table definitions.
    if DB_PATH.exists():
        logger.info(f"Removing existing warehouse database at: {DB_PATH}")
        DB_PATH.unlink()

    conn: sqlite3.Connection | None = None

    try:
        # Connect to SQLite. Create the file if it doesn't exist
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create schema and clear existing records
        create_schema(cursor)
        delete_existing_records(cursor)

        # Load prepared data using pandas
        customers_df = pd.read_csv(CLEAN_DATA_DIR.joinpath("customers_data_prepared.csv"))
        products_df = pd.read_csv(CLEAN_DATA_DIR.joinpath("products_data_prepared.csv"))
        sales_df = pd.read_csv(CLEAN_DATA_DIR.joinpath("sales_data_prepared.csv"))

        # Rename clean columns to match database schema if necessary
        customers_df = customers_df.rename(
            columns={
                "CustomerID": "customer_id",
                "Name": "name",
                "Region": "region",
                "JoinDate": "join_date",
            }
        )
        logger.info(f"Customer columns (cleaned): {list(customers_df.columns)}")

        products_df = products_df.rename(
            columns={
                "ProductID": "product_id",
                "ProductName": "product_name",
                "Category": "category",
                "UnitPrice": "unit_price",
            }
        )
        logger.info(f"Product columns (cleaned):  {list(products_df.columns)}")

        # FIXED: typo in comment and aligned with schema
        sales_df = sales_df.rename(
            columns={
                "TransactionID": "sale_id",
                "CustomerID": "customer_id",
                "ProductID": "product_id",
                "SaleAmount": "sale_amount",
                "SaleDate": "date",  # FIXED: matched schema column name
            }
        )
        logger.info(f"Sales columns (cleaned):    {list(sales_df.columns)}")

        # Insert data into the database for all tables
        insert_customers(customers_df, conn)
        insert_products(products_df, conn)
        insert_sales(sales_df, conn)

        # Commit the transaction
        conn.commit()
        logger.info("ETL finished successfully. Data loaded into the warehouse.")
    finally:
        # Regardless of success or failure, close the DB connection if it exists
        if conn is not None:
            logger.info("Closing database connection.")
            conn.close()


if __name__ == "__main__":
    load_data_to_db()
