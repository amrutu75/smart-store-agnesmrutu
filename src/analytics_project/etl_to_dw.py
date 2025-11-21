"""
ETL script to load prepared data into the data warehouse (SQLite database).

File: src/analytics_project/dw/etl_to_dw.py

Purpose:
    Loads cleaned data (from /data/prepared) into the warehouse database (/data/warehouse/smart_sales.db).
    Ensures schema alignment between prepared CSVs and the database.
"""

# ===============================================================
# Imports
# ===============================================================

import pathlib
import sqlite3
import pandas as pd
from analytics_project.utils_logger import logger


# ===============================================================
# Global path constants
# ===============================================================

THIS_DIR: pathlib.Path = pathlib.Path(__file__).resolve().parent
DW_DIR: pathlib.Path = THIS_DIR
PACKAGE_DIR: pathlib.Path = DW_DIR.parent
SRC_DIR: pathlib.Path = PACKAGE_DIR.parent
PROJECT_ROOT_DIR: pathlib.Path = SRC_DIR.parent

DATA_DIR: pathlib.Path = PROJECT_ROOT_DIR / "data"
RAW_DATA_DIR: pathlib.Path = DATA_DIR / "raw"
CLEAN_DATA_DIR: pathlib.Path = DATA_DIR / "prepared"
WAREHOUSE_DIR: pathlib.Path = DATA_DIR / "warehouse"

DB_PATH: pathlib.Path = WAREHOUSE_DIR / "smart_sales.db"

# Log key directories for debugging
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


# ===============================================================
# Schema creation and table management
# ===============================================================


def create_schema(cursor: sqlite3.Cursor) -> None:
    """Create tables in the data warehouse if they don't exist."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer (
            customer_id INTEGER PRIMARY KEY,
            name TEXT,
            region TEXT,
            join_date TEXT,
            LoyaltyPoints REAL,
            ContactMethod TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS product (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT,
            category TEXT,
            unit_price REAL,
            StockQuantity INTEGER,
            Supplier TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sale (
            sale_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product_id INTEGER,
            sale_amount REAL,
            sale_date TEXT,
            FOREIGN KEY (customer_id) REFERENCES customer (customer_id),
            FOREIGN KEY (product_id) REFERENCES product (product_id)
        )
    """)


# ===============================================================
# Insert functions (idempotent)
# ===============================================================


def insert_customers(
    customers_df: pd.DataFrame, cursor: sqlite3.Cursor, conn: sqlite3.Connection
) -> None:
    """Insert only new customer rows to avoid UNIQUE constraint errors."""
    existing_ids = pd.read_sql("SELECT customer_id FROM customer", conn)
    if not existing_ids.empty:
        customers_df = customers_df[~customers_df['customer_id'].isin(existing_ids['customer_id'])]
    if not customers_df.empty:
        logger.info(f"Inserting {len(customers_df)} new customer rows.")
        customers_df.to_sql("customer", conn, if_exists="append", index=False)
    else:
        logger.info("No new customer rows to insert.")


def insert_products(
    products_df: pd.DataFrame, cursor: sqlite3.Cursor, conn: sqlite3.Connection
) -> None:
    """Insert only new product rows to avoid UNIQUE constraint errors."""
    existing_ids = pd.read_sql("SELECT product_id FROM product", conn)
    if not existing_ids.empty:
        products_df = products_df[~products_df['product_id'].isin(existing_ids['product_id'])]
    if not products_df.empty:
        logger.info(f"Inserting {len(products_df)} new product rows.")
        products_df.to_sql("product", conn, if_exists="append", index=False)
    else:
        logger.info("No new product rows to insert.")


def insert_sales(sales_df: pd.DataFrame, cursor: sqlite3.Cursor, conn: sqlite3.Connection) -> None:
    """Insert only new sale rows to avoid UNIQUE constraint errors."""
    existing_ids = pd.read_sql("SELECT sale_id FROM sale", conn)
    if not existing_ids.empty:
        sales_df = sales_df[~sales_df['sale_id'].isin(existing_ids['sale_id'])]
    if not sales_df.empty:
        logger.info(f"Inserting {len(sales_df)} new sale rows.")
        sales_df.to_sql("sale", conn, if_exists="append", index=False)
    else:
        logger.info("No new sale rows to insert.")


# ===============================================================
# ETL main process
# ===============================================================


def load_data_to_db() -> None:
    """Load clean data into the data warehouse."""
    logger.info("Starting ETL: loading clean data into the warehouse.")

    # Make sure the warehouse directory exists
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # (Re)create schema
        create_schema(cursor)

        # ---------------------------------------------------------------
        # Load prepared CSVs
        # ---------------------------------------------------------------
        customers_df = pd.read_csv(CLEAN_DATA_DIR / "customers_data_prepared.csv")
        products_df = pd.read_csv(CLEAN_DATA_DIR / "products_data_prepared.csv")
        sales_df = pd.read_csv(CLEAN_DATA_DIR / "sales_data_prepared.csv")

        # Remove duplicates
        customers_df = customers_df.drop_duplicates()
        products_df = products_df.drop_duplicates()
        sales_df = sales_df.drop_duplicates()

        # ---------------------------------------------------------------
        # Rename columns to match database schema
        # ---------------------------------------------------------------
        customers_df = customers_df.rename(
            columns={
                "CustomerID": "customer_id",
                "Name": "name",
                "Email": "email",
                "Region": "region",
                "JoinDate": "join_date",
                # LoyaltyPoints and ContactMethod already match schema
            }
        )
        logger.info(f"Customer columns (cleaned): {list(customers_df.columns)}")

        products_df = products_df.rename(
            columns={
                "ProductID": "product_id",
                "ProductName": "product_name",
                "Category": "category",
                "UnitPrice": "unit_price",
                # StockQuantity and Supplier already match schema
            }
        )
        logger.info(f"Product columns (cleaned): {list(products_df.columns)}")

        sales_df = sales_df.rename(
            columns={
                "TransactionID": "sale_id",
                "SaleDate": "sale_date",
                "CustomerID": "customer_id",
                "ProductID": "product_id",
                "CampaignID": "campaign_id",
                "SaleAmount": "sale_amount",
                "DiscountAmount": "discount_amount",
                "state": "state",
            }
        )
        logger.info(f"Sales columns (cleaned): {list(sales_df.columns)}")

        # ---------------------------------------------------------------
        # Load data into warehouse tables (idempotent)
        # ---------------------------------------------------------------
        insert_customers(customers_df, cursor, conn)
        insert_products(products_df, cursor, conn)
        insert_sales(sales_df, cursor, conn)

        conn.commit()
        logger.info("ETL finished successfully. Data loaded into the warehouse.")

    finally:
        if conn is not None:
            logger.info("Closing database connection.")
            conn.close()


# ===============================================================
# Script entrypoint
# ===============================================================

if __name__ == "__main__":
    load_data_to_db()
