-- Q1: WHERE then HAVING
SELECT customer_id, SUM(order_amount) AS total_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING total_spend > 8000;

-- Q2: Three-table join for revenue by city and category
SELECT c.city, p.category, SUM(o.order_amount) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category;

-- Q3: Scalar subquery for products priced above average catalogue price
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products);

-- Q4: LEFT JOIN for customers with no completed orders
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL;

-- Q5: CTE for customers spending above average
WITH cust_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) AS avg_total FROM cust_spend
)
cs SELECT cs.customer_id, cs.total_spend
FROM cust_spend cs, avg_spend as
WHERE cs.total_spend > as.avg_total;

-- Q6: Ranking window functions by city spend
WITH city_spend AS (
    SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) as rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) as dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) as row_num
FROM city_spend;

-- Q7: Running total of daily completed revenue
SELECT SUBSTR(order_ts, 1, 10) AS day,
       SUM(order_amount) AS daily_revenue,
       SUM(SUM(order_amount)) OVER (ORDER BY SUBSTR(order_ts, 1, 10)) AS cumulative_revenue
FROM orders
WHERE status = 'completed'
GROUP BY day;