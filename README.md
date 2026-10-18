Instamart Customer Intelligence & Support Copilot — Capstone ProjectOverviewSwiggy Instamart delivers groceries, dairy, snacks, personal-care items, and small electronics within minutes across six metro cities (Chennai, Mumbai, Bengaluru, Delhi, Hyderabad, Pune). As order volumes scale, data integrity, churn prediction, and support ticket management become critical bottlenecks.This repository implements the Customer Intelligence & Support Copilot, a unified internal tool built across four connected parts:Part A: Data Foundation — Audits, cleans, stores, and queries messy raw order and customer exports in a trusted SQLite database (data/instamart.db).

Part B: Customer Segments & Lapse Prediction — Computes RFM metrics, applies K-means clustering and PCA to segment customers, and trains a tuned machine learning model to predict customer lapse behavior.

Part C: Support Ticket Triage with Guardrails — Sanitizes PII with regular expressions, enforces input guardrails, handles resilient LLM calls with exponential backoff and jitter, validates outputs, and routes tickets intelligently based on customer value and priority.

Part D: Policy RAG Assistant & Refund Agent — Chunks and indexes policy documents using TF-IDF retrieval, grounds answers, and implements a ReAct refund agent equipped with SQLite and ledger tools, complete with safety limits, timeouts, and state checkpointing.ReproduceTo replicate the entire pipeline from a clean clone, run the following numbered shell commands in order:

Bash# 1. Create and activate virtual environment

python -m venv venv
venv\Scripts\Activate  # On macOS/Linux use: source venv/bin/activate

# 2. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Generate synthetic raw data and policies
python generate_data.py

# 4. Part A: Audit, clean, and store data in SQLite, then validate
python src/audit_data.py
python src/clean_and_store.py
python src/validate_db.py

# 5. Part B: Run customer segmentation and lapse prediction models
python src/customer_segmentation.py
python src/lapse_prediction.py

# 6. Part C: Execute support ticket triage pipeline
python src/triage_pipeline.py

# 7. Part D: Execute policy retrieval and ReAct agent scenarios
python src/policy_agent.py
Part A — Data Foundation: Clean, Store, QueryAudit StatisticsRaw Orders Rows: 13,084Exact Duplicate Rows: 119Pending Rows: 656Distinct order_id Values: 12,309Non-numeric order_amount Entries: 120Negative delivery_minutes: 106Blank delivery_minutes: 2,541Blank Ages: 61Ages outside 10–100: 6Distinct Raw City Spellings: 24Distinct Non-blank Gender Spellings: 10Blank Genders: 30Cleaning Rules & Post-Cleaning StatusCustomers: Standardized cities (stripped and title-cased into 6 cities), normalized genders into male, female, or unknown, and imputed missing/out-of-bounds ages using city medians. Signup dates converted to datetimes.Orders: Converted timestamps, removed exact duplicates, retained only the latest record for multi-status orders, imputed unparseable amounts using catalogue prices and quantities, set negative delivery times to missing, and imputed completed delivery times via category medians.Database: Stored cleaned tables into CSVs under data/clean/ and SQLite database data/instamart.db. Completed order revenue sum equals INR 2,998,668.80. Validation script verified all checkpoints (ALL CHECKPOINTS PASSED).

Part B — Customer Segments and Lapse PredictionRFM Table: Built for 1,175 customers as of 2025-10-01.
K-means Clustering: Evaluated $K \in [2, 8]$. Selected $K=4$ based on inertia reduction, silhouette score ($0.526$), and business interpretability. Segments named: VIP Champions, Frequent Buyers, At-Risk Lapsers, and Casual Shoppers.PCA Projection: Applied PCA(n_components=2) yielding an explained variance ratio of about [0.634, 0.218] (total ~0.852).Lapse Model: Trained on pre-cutoff orders (cutoff = 2025-09-01). Compared Logistic Regression and Random Forest using GridSearchCV (cv=5, scoring=f1). The tuned Random Forest model achieved a test F1 score exceeding 0.65 on the positive (lapsed) class.Class-Balance Experiment: Oversampling the minority lapsed class raised recall from ~0.64 to ~0.85 while maintaining stable precision (~0.6).Part C — Support Ticket Triage with GuardrailsInput Guardrails & Redaction: Filtered 4 empty tickets (skipped_empty) and 4 excessively long tickets (>400 tokens, flagged_long). Regex substitution successfully masked 55 emails, 91 phone numbers, and 132 order IDs (total 278 substitutions), with post-assertion checks confirming zero PII leaks.Resilient Calls & Validation: Wrapped model calls with exponential backoff and jitter (base delay 0.05s). Successfully recovered from all 19 simulated TransientLLMError failures. Output validation produced 135 classified_json and 12 classified_keyword responses.Prompt Comparison: Zero-shot prompt achieved 85.7% accuracy, while the few-shot prompt achieved 100% accuracy.Outputs: Saved triage outputs to outputs/triage_results.csv and budget demo to outputs/triage_budget_demo.csv.Part D — Policy RAG Assistant & Refund AgentChunking & Retrieval: Split policy documents into 40-word chunks with 10-word overlap (15 chunks total). TF-IDF vectorizer and cosine similarity successfully retrieved relevant policy context.Agent Scenarios Results:S1: REFUND | INR 128.25 | 5 LLM callsS2: REFUND | INR 135.00 | 5 LLM callsS3: ESCALATE | INR 1,999.00 | 6 LLM callsS4: DENIED | INR 0.00 | 2 LLM callsS5: DENIED | INR 0.00 | 2 LLM callsS6: NOT_FOUND | INR 0.00 | 2 LLM callsS7: FALLBACK_MAX_ITERATIONS (Loop trigger) | 6 LLM callsSafety Demonstrations:Token-limit demo (token_limit=200): Stopped correctly with FALLBACK_TOKEN_LIMIT.Timeout demo (time_limit_seconds=0): Stopped correctly with FALLBACK_TIMEOUT.Crash & Resume Demo (S1): Successfully restarted via resume_agent(run_id). Tools executed across the crash and resume in order:get_ordersearch_policycalculate_refundissue_refund(Final status: REFUND, INR 128.25)