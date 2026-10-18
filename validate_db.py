import sqlite3
import pandas as pd

def validate():
    conn = sqlite3.connect("data/instamart.db")
    
    # Check 1: table existence
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)["name"].tolist()
    assert all(t in tables for t in ["customers", "products", "orders"]), "Missing tables!"

    # Check 2: row counts
    cust = pd.read_sql("SELECT * FROM customers", conn)
    orders = pd.read_sql("SELECT * FROM orders", conn)
    assert len(cust) == 1200
    assert len(orders) == 12309

    # Check 3: completed orders count
    comp_count = pd.read_sql("SELECT COUNT(*) as cnt FROM orders WHERE status='completed'", conn)["cnt"].iloc[0]
    assert comp_count == 11110

    # Check 4: cancelled orders count
    canc_count = pd.read_sql("SELECT COUNT(*) as cnt FROM orders WHERE status='cancelled'", conn)["cnt"].iloc[0]
    assert canc_count == 1199

    # Check 5: missing order amounts
    null_amt = pd.read_sql("SELECT COUNT(*) as cnt FROM orders WHERE order_amount IS NULL", conn)["cnt"].iloc[0]
    assert null_amt == 0

    # Check 6: completed order revenue sum
    rev_sum = pd.read_sql("SELECT SUM(order_amount) as s FROM orders WHERE status='completed'", conn)["s"].iloc[0]
    assert abs(rev_sum - 2998668.80) < 0.01

    # Check 7: no pending orders
    pend_count = pd.read_sql("SELECT COUNT(*) as cnt FROM orders WHERE status='pending'", conn)["cnt"].iloc[0]
    assert pend_count == 0

    # Check 8: city distribution
    delhi_count = pd.read_sql("SELECT COUNT(*) as cnt FROM customers WHERE city='Delhi'", conn)["cnt"].iloc[0]
    assert delhi_count == 210

    conn.close()
    print("ALL CHECKPOINTS PASSED")

if __name__ == "__main__":
    validate()