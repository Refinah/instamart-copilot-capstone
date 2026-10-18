import os
import sqlite3
import pandas as pd
import numpy as np

def clean_and_store():
    raw_dir = "data/raw"
    clean_dir = "data/clean"
    os.makedirs(clean_dir, exist_ok=True)

    # --- Clean Customers ---
    cust = pd.read_csv(os.path.join(raw_dir, "customers.csv"))
    cust["city"] = cust["city"].astype(str).str.strip().str.title()
    
    def clean_gender(g):
        g_lower = str(g).strip().lower()
        if g_lower in ["male", "m"]:
            return "male"
        elif g_lower in ["female", "f"]:
            return "female"
        else:
            return "unknown"
    cust["gender"] = cust["gender"].apply(clean_gender)

    cust["age"] = pd.to_numeric(cust["age"], errors="coerce")
    cust.loc[(cust["age"] < 10) | (cust["age"] > 100), "age"] = np.nan
    city_medians = cust.groupby("city")["age"].transform("median")
    cust["age"] = cust["age"].fillna(city_medians).round().astype(int)

    cust["signup_date"] = pd.to_datetime(cust["signup_date"]).dt.strftime("%Y-%m-%d")
    cust.to_csv(os.path.join(clean_dir, "customers.csv"), index=False)

    # --- Clean Products ---
    prod = pd.read_csv(os.path.join(raw_dir, "products.csv"))
    prod.to_csv(os.path.join(clean_dir, "products.csv"), index=False)

    # --- Clean Orders ---
    orders = pd.read_csv(os.path.join(raw_dir, "orders.csv"), keep_default_na=False)
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    
    orders = orders.drop_duplicates()
    orders = orders.sort_values(by="order_ts").groupby("order_id", as_index=False).last()

    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    prod_map = prod.set_index("product_id")[["price_inr", "category"]].to_dict("index")
    
    def fix_amount(row):
        if pd.isna(row["order_amount"]):
            p_info = prod_map.get(row["product_id"], {"price_inr": 0})
            return round(p_info["price_inr"] * row["quantity"] * (1 - row["discount_pct"] / 100), 2)
        return row["order_amount"]
    
    orders["order_amount"] = orders.apply(fix_amount, axis=1)

    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

    orders["category"] = orders["product_id"].map(lambda pid: prod_map.get(pid, {}).get("category", "Grocery"))
    cat_medians = orders[orders["status"] == "completed"].groupby("category")["delivery_minutes"].transform("median")
    
    completed_mask = orders["status"] == "completed"
    orders.loc[completed_mask & orders["delivery_minutes"].isna(), "delivery_minutes"] = cat_medians[completed_mask]
    orders["delivery_minutes"] = orders["delivery_minutes"].round()

    orders = orders.drop(columns=["category"])
    orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    orders.to_csv(os.path.join(clean_dir, "orders.csv"), index=False)

    # --- Store in SQLite ---
    db_path = "data/instamart.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cust.to_sql("customers", conn, index=False, if_exists="replace")
    prod.to_sql("products", conn, index=False, if_exists="replace")
    orders.to_sql("orders", conn, index=False, if_exists="replace")
    conn.close()
    print("Cleaned data stored successfully in CSV and SQLite.")

if __name__ == "__main__":
    clean_and_store()