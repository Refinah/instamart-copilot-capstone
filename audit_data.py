import os
import sqlite3
import pandas as pd
import numpy as np

def run_part_a():
    os.makedirs(os.path.join("data", "clean"), exist_ok=True)
    os.makedirs("sql", exist_ok=True)

    # 1. Audit Raw Data & Load
    customers_raw = pd.read_csv("data/raw/customers.csv")
    products_raw = pd.read_csv("data/raw/products.csv")
    orders_raw = pd.read_csv("data/raw/orders.csv")
    tickets_raw = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)

    print(f"Raw Orders Shape: {orders_raw.shape}")
    print(f"Exact Duplicates in Orders: {orders_raw.duplicated().sum()}")
    print(f"Pending Orders: {(orders_raw['status'] == 'pending').sum()}")
    print(f"Distinct Order IDs: {orders_raw['order_id'].nunique()}")

    # 2. Clean Customers
    cust = customers_raw.copy()
    # (a) Standardize city: strip spaces and apply title case
    cust['city'] = cust['city'].astype(str).str.strip().str.title()
    
    # (b) Map gender spelling to male, female, unknown
    def clean_gender(g):
        g_str = str(g).strip().lower()
        if g_str in ['m', 'male']:
            return 'male'
        elif g_str in ['f', 'female']:
            return 'female'
        else:
            return 'unknown'
    cust['gender'] = cust['gender'].apply(clean_gender)

    # (c) Handle ages: outside 10-100 treated as missing, fill missing with city median
    cust['age'] = pd.to_numeric(cust['age'], errors='coerce')
    cust.loc[(cust['age'] < 10) | (cust['age'] > 100), 'age'] = np.nan
    city_medians = cust.groupby('city')['age'].transform('median')
    cust['age'] = cust['age'].fillna(city_medians).round().astype(int)

    # (d) Convert signup_date to datetime
    cust['signup_date'] = pd.to_datetime(cust['signup_date'])

    # 3. Clean Orders
    ord_df = orders_raw.copy()
    # (a) Convert order_ts to datetime
    ord_df['order_ts'] = pd.to_datetime(ord_df['order_ts'])

    # (b) Remove exact duplicate rows
    ord_df = ord_df.drop_duplicates()

    # (c) Keep only the record with the latest order_ts per order_id
    ord_df = ord_df.sort_values('order_ts').groupby('order_id', as_index=False).last()

    # (d) Convert order_amount to numeric, fill missing using product price
    ord_df['order_amount_num'] = pd.to_numeric(ord_df['order_amount'], errors='coerce')
    # Merge product price
    ord_df = ord_df.merge(products_raw[['product_id', 'price_inr']], on='product_id', how='left')
    calc_amount = round(ord_df['price_inr'] * ord_df['quantity'] * (1 - ord_df['discount_pct'] / 100), 2)
    ord_df['order_amount'] = ord_df['order_amount_num'].fillna(calc_amount)
    ord_df = ord_df.drop(columns=['order_amount_num', 'price_inr'])

    # (e) Set negative delivery_minutes to missing
    ord_df['delivery_minutes'] = pd.to_numeric(ord_df['delivery_minutes'], errors='coerce')
    ord_df.loc[ord_df['delivery_minutes'] < 0, 'delivery_minutes'] = np.nan

    # Impute missing delivery time for completed orders using category median
    ord_df = ord_df.merge(products_raw[['product_id', 'category']], on='product_id', how='left')
    completed_mask = ord_df['status'] == 'completed'
    cat_medians = ord_df[completed_mask].groupby('category')['delivery_minutes'].transform('median')
    ord_df.loc[completed_mask, 'delivery_minutes'] = ord_df.loc[completed_mask, 'delivery_minutes'].fillna(cat_medians)
    ord_df['delivery_minutes'] = ord_df['delivery_minutes'].round()
    ord_df = ord_df.drop(columns=['category'])

    # Format order_ts back to string YYYY-MM-DD HH:MM:SS
    ord_df['order_ts'] = ord_df['order_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # 4. Store Clean Tables
    cust.to_csv("data/clean/customers.csv", index=False)
    products_raw.to_csv("data/clean/products.csv", index=False)
    ord_df.to_csv("data/clean/orders.csv", index=False)

    conn = sqlite3.connect("data/instamart.db")
    cust.to_sql("customers", conn, if_exists="replace", index=False)
    products_raw.to_sql("products", conn, if_exists="replace", index=False)
    ord_df.to_sql("orders", conn, if_exists="replace", index=False)
    conn.close()

    # 5. Validation Checkpoints
    assert len(ord_df) == 12309, f"Expected 12309 rows, got {len(ord_df)}"
    assert (ord_df['status'] == 'completed').sum() == 11110
    assert (ord_df['status'] == 'cancelled').sum() == 1199
    assert ord_df['order_amount'].isna().sum() == 0
    assert ord_df.loc[ord_df['status'] == 'completed', 'delivery_minutes'].isna().sum() == 0
    assert ord_df.loc[ord_df['status'] == 'cancelled', 'delivery_minutes'].isna().sum() == 1199
    assert cust['age'].between(18, 60).all()
    assert cust['age'].isna().sum() == 0
    
    print("ALL CHECKPOINTS PASSED")

if __name__ == "__main__":
    run_part_a()