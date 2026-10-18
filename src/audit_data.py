import os
import pandas as pd

def audit_data():
    raw_dir = "data/raw"
    customers = pd.read_csv(os.path.join(raw_dir, "customers.csv"))
    products = pd.read_csv(os.path.join(raw_dir, "products.csv"))
    orders = pd.read_csv(os.path.join(raw_dir, "orders.csv"), keep_default_na=False)
    tickets = pd.read_csv(os.path.join(raw_dir, "tickets.csv"), keep_default_na=False)

    print("--- AUDIT RESULTS ---")
    print(f"Customers shape: {customers.shape}")
    print(f"Products shape: {products.shape}")
    print(f"Orders shape: {orders.shape}")
    print(f"Tickets shape: {tickets.shape}")

    exact_dups = orders.duplicated().sum()
    pending_rows = (orders["status"] == "pending").sum()
    distinct_orders = orders["order_id"].nunique()
    
    non_numeric_amt = pd.to_numeric(orders["order_amount"], errors="coerce").isna().sum()
    
    delivery_num = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    neg_delivery = (delivery_num < 0).sum()
    blank_delivery = (orders["delivery_minutes"] == "").sum()

    blank_age = customers["age"].replace("", pd.NA).isna().sum()
    valid_ages = pd.to_numeric(customers["age"], errors="coerce")
    out_of_bounds_age = ((valid_ages < 10) | (valid_ages > 100)).sum()
    distinct_cities = customers["city"].nunique()
    non_blank_gender = customers[customers["gender"] != ""]["gender"]
    distinct_genders = non_blank_gender.nunique()
    blank_gender = (customers["gender"] == "").sum()

    print(f"Orders rows: {len(orders)}")
    print(f"Exact duplicate rows: {exact_dups}")
    print(f"Pending rows: {pending_rows}")
    print(f"Distinct order_id: {distinct_orders}")
    print(f"Non-numeric order_amount: {non_numeric_amt}")
    print(f"Negative delivery_minutes: {neg_delivery}")
    print(f"Blank delivery_minutes: {blank_delivery}")
    print(f"Blank age: {blank_age}")
    print(f"Age outside 10-100: {out_of_bounds_age}")
    print(f"Distinct raw cities: {distinct_cities}")
    print(f"Distinct non-blank genders: {distinct_genders}")
    print(f"Blank gender: {blank_gender}")

if __name__ == "__main__":
    audit_data()