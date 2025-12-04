"""Module 6: OLAP Goal Script (uses cubed results).

File: src/analytics_project/olap/goal_sales_by_day.py.

Module: analytics_project.olap.goal_sales_by_day

This script uses our precomputed cubed data set to get the information
we need to answer a specific business goal.

GOAL: Analyze sales data to determine which day of the week
consistently shows the lowest sales revenue.

ACTION: This can help inform decisions about reducing operating hours
or focusing marketing efforts on less profitable days.

PROCESS:
Group transactions by the day of the week.
Sum SaleAmount for each day.
Identify the day with the lowest total revenue.

This example assumes a cube data set with the following column names (yours will differ).
DayOfWeek,product_id,customer_id,sale_amount_sum,sale_amount_mean,sale_id_count,sale_ids
Friday,101,1001,6344.96,6344.96,1,[582]
etc.
"""

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
from analytics_project.utils_logger import logger

# Paths
# Global constants for paths and key directories

THIS_DIR: pathlib.Path = pathlib.Path(__file__).resolve().parent
DW_DIR: pathlib.Path = THIS_DIR  # src/analytics_project/olap/
PACKAGE_DIR: pathlib.Path = DW_DIR.parent  # src/analytics_project/
SRC_DIR: pathlib.Path = PACKAGE_DIR.parent  # src/
PROJECT_ROOT_DIR: pathlib.Path = SRC_DIR.parent  # project_root/

# Data directories
DATA_DIR: pathlib.Path = PROJECT_ROOT_DIR / "data"
WAREHOUSE_DIR: pathlib.Path = DATA_DIR / "warehouse"

# Warehouse database location (SQLite)
DB_PATH: pathlib.Path = WAREHOUSE_DIR / "smart_sales.db"

# OLAP output directory
OLAP_OUTPUT_DIR: pathlib.Path = DATA_DIR / "olap_cubing_outputs"

# CUBED File paths
CUBED_FILE: pathlib.Path = OLAP_OUTPUT_DIR / "multidimensional_olap_cube.csv"
SALES_BY_REGION_FILE: pathlib.Path = OLAP_OUTPUT_DIR / "sales_by_region.csv"
SALES_GROWTH_FILE: pathlib.Path = OLAP_OUTPUT_DIR / "sales_growth_by_date.csv"
TOP_PRODUCTS_FILE: pathlib.Path = OLAP_OUTPUT_DIR / "top_selling_products.csv"

# Results output directory
RESULTS_OUTPUT_DIR: pathlib.Path = DATA_DIR / "results"

# Recommended - log paths and key directories for debugging

logger.info(f"THIS_DIR:            {THIS_DIR}")
logger.info(f"DW_DIR:              {DW_DIR}")
logger.info(f"PACKAGE_DIR:         {PACKAGE_DIR}")
logger.info(f"SRC_DIR:             {SRC_DIR}")
logger.info(f"PROJECT_ROOT_DIR:    {PROJECT_ROOT_DIR}")

logger.info(f"DATA_DIR:            {DATA_DIR}")
logger.info(f"WAREHOUSE_DIR:       {WAREHOUSE_DIR}")
logger.info(f"DB_PATH:             {DB_PATH}")
logger.info(f"OLAP_OUTPUT_DIR:     {OLAP_OUTPUT_DIR}")
logger.info(f"RESULTS_OUTPUT_DIR:  {RESULTS_OUTPUT_DIR}")

# Create output directory if it does not exist
OLAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create output directory for results if it doesn't exist
RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_sales_by_region():
    """Load and visualize sales by region."""
    try:
        df = pd.read_csv(SALES_BY_REGION_FILE)
        logger.info(f"Loaded sales_by_region with {len(df)} rows.")

        # Normalize region names
        df["region"] = df["region"].astype(str).str.strip().str.title()

        # Aggregate
        sales_by_region = df.groupby("region")["sale_amount"].sum().reset_index()
        sales_by_region.rename(
            columns={"sale_amount": "TotalSales", "region": "Region"}, inplace=True
        )
        sales_by_region.sort_values(by="TotalSales", inplace=True)

        logger.info(f"Sales by region:\n{sales_by_region}")

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.bar(sales_by_region["Region"], sales_by_region["TotalSales"], color="skyblue")
        plt.title("Total Sales by Region", fontsize=16)
        plt.xlabel("Region")
        plt.ylabel("Total Sales (USD)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(RESULTS_OUTPUT_DIR / "sales_by_region.png")
        plt.close()

        return sales_by_region
    except Exception as e:
        logger.error(f"Error analyzing sales by region: {e}")
        return pd.DataFrame()


def analyze_sales_growth():
    """Load and visualize sales growth by date."""
    try:
        df = pd.read_csv(SALES_GROWTH_FILE)
        logger.info(f"Loaded sales_growth_by_date with {len(df)} rows.")

        # Ensure numeric
        df["sale_amount"] = pd.to_numeric(df["sale_amount"], errors="coerce")
        df["Sales_Growth"] = pd.to_numeric(df["Sales_Growth"], errors="coerce")

        logger.info(f"Sales growth preview:\n{df.head()}")

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.plot(df["Date"], df["sale_amount"], marker="o", label="Sales Amount")
        plt.plot(df["Date"], df["Sales_Growth"], marker="x", label="Sales Growth")
        plt.title("Sales Growth by Date", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("USD")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_OUTPUT_DIR / "sales_growth_by_date.png")
        plt.close()

        return df
    except Exception as e:
        logger.error(f"Error analyzing sales growth: {e}")
        return pd.DataFrame()


def analyze_top_products():
    """Load and visualize top selling products."""
    try:
        df = pd.read_csv(TOP_PRODUCTS_FILE)
        logger.info(f"Loaded top_selling_products with {len(df)} rows.")

        df["sale_amount"] = pd.to_numeric(df["sale_amount"], errors="coerce")
        df = df.sort_values(by="sale_amount", ascending=False)

        logger.info(f"Top products:\n{df.head(10)}")

        # Visualization
        plt.figure(figsize=(12, 6))
        plt.bar(df["product_id"].astype(str).head(10), df["sale_amount"].head(10), color="green")
        plt.title("Top 10 Selling Products", fontsize=16)
        plt.xlabel("Product ID")
        plt.ylabel("Total Sales (USD)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(RESULTS_OUTPUT_DIR / "top_selling_products.png")
        plt.close()

        return df
    except Exception as e:
        logger.error(f"Error analyzing top products: {e}")
        return pd.DataFrame()


def main():
    logger.info("Starting analysis of 3 cubes...")

    sales_by_region = analyze_sales_by_region()
    sales_growth = analyze_sales_growth()
    top_products = analyze_top_products()

    logger.info("Analysis complete. Visualizations saved to results folder.")


if __name__ == "__main__":
    main()
