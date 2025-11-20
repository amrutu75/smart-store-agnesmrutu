-- File: 90_create_sales.sql
-- Compatible with SQLite NOT DUCKDB 
-- DATE Field Note:
-- - SQLite: Uses TEXT for date fields (store as 'YYYY-MM-DD' ISO format)
-- - DuckDB and others: Support native DATE type (can parse ISO-formatted TEXT)
-- - CSV Columns: TransactionID,SaleDate,CustomerID,ProductID,StoreID,CampaignID,SaleAmount
-- - In DuckDB we don't use the Key and Foreign Key constraints
-- - In SQLite we use the Key and Foreign Key constraints

DROP TABLE IF EXISTS sales;

CREATE TABLE sales (
    sale_id INTEGER PRIMARY KEY,           -- from TransactionID
    date TEXT,                             -- from SaleDate (ISO 8601 format)
    customer_id TEXT,                      -- from CustomerID
    product_id TEXT,                       -- from ProductID
    state TEXT,                            -- from State
    discount_amount REAL,                  -- from DiscountAmount
    sales_amount REAL,                     -- from SaleAmount
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);