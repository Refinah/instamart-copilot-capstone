import sys
import os

# Dynamically find the project root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Diagnostic check to verify mock_llm exists in root
mock_path = os.path.join(ROOT_DIR, "mock_llm.py")
if not os.path.exists(mock_path):
    print(f"ERROR: mock_llm.py was not found at expected path: {mock_path}")
    print("Please ensure mock_llm.py is placed directly in your repository root folder.")
    sys.exit(1)

import re
import random
import time
import json
import pandas as pd
import mock_llm

def run_triage_pipeline():
    tickets_path = "data/raw/tickets.csv"
    tickets_df = pd.read_csv(tickets_path, keep_default_na=False)
    segments = pd.read_csv("outputs/customer_segments.csv")
    prem_set = set(segments[segments["is_premium"] == True]["customer_id"])

    # 1. Input Guardrails & Redaction
    email_pat = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
    phone_pat = re.compile(r"\b[6-9]\d{9}\b")
    order_pat = re.compile(r"ORD-\d{6}")

    sub_counts = {"email": 0, "phone": 0, "order_id": 0}

    def process_ticket_text(txt):
        nonlocal sub_counts
        if not txt or not str(txt).strip():
            return "", "skipped_empty"
        
        tokens_est = mock_llm.est_tokens(txt)
        if tokens_est > 400:
            return txt, "flagged_long"

        # Redact PII
        e_matches = email_pat.findall(txt)
        p_matches = phone_pat.findall(txt)
        o_matches = order_pat.findall(txt)

        sub_counts["email"] += len(e_matches)
        sub_counts["phone"] += len(p_matches)
        sub_counts["order_id"] += len(o_matches)

        redacted = email_pat.sub("[EMAIL]", txt)
        redacted = phone_pat.sub("[PHONE]", redacted)
        redacted = order_pat.sub("[ORDER_ID]", redacted)

        # Assertion test: re-applying patterns finds nothing
        assert len(email_pat.findall(redacted)) == 0
        assert len(phone_pat.findall(redacted)) == 0
        assert len(order_pat.findall(redacted)) == 0

        return redacted, "ok"

    few_shot_sys = "CLASSIFY support tickets into exactly one of: billing, delivery, technical, fraud. Return JSON only: {\"category\": \"...\", \"reason\": \"...\". EXAMPLES: Text: unauthorized transaction on card ORD-123456 -> {\"category\": \"fraud\", \"reason\": \"unauthorized\"}, Text: app crashes in cart -> {\"category\": \"technical\", \"reason\": \"crash\"}"

    def call_with_retry(messages):
        base_delay = 0.05
        for n in range(3):
            try:
                return mock_llm.chat(messages)
            except mock_llm.TransientLLMError as e:
                wait = base_delay * (2 ** n) * random.uniform(0.9, 1.15)
                time.sleep(wait)
        raise Exception("Max retries exceeded")

    results = []
    token_budget = 100000
    running_tokens = 0

    mock_llm.reset()

    for idx, row in tickets_df.iterrows():
        tid = row["ticket_id"]
        cid = row["customer_id"]
        txt = row["text"]
        true_cat = row["true_category"]

        clean_txt, status = process_ticket_text(txt)
        if status != "ok":
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": status,
                "category": "", "segment_name": "", "priority": "NORMAL", "route": "human_review", "tokens_used": 0
            })
            continue

        if running_tokens >= token_budget:
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "budget_exhausted",
                "category": "", "segment_name": "", "priority": "NORMAL", "route": "human_review", "tokens_used": 0
            })
            continue

        messages = [
            {"role": "system", "content": few_shot_sys},
            {"role": "user", "content": clean_txt}
        ]

        try:
            resp = call_with_retry(messages)
            usage = resp.usage.total_tokens
            running_tokens += usage
            content = resp.choices[0].message.content

            category = ""
            resp_status = ""
            try:
                data = json.loads(content)
                if data.get("category") in ["billing", "delivery", "technical", "fraud"]:
                    category = data["category"]
                    resp_status = "classified_json"
                else:
                    resp_status = "flagged_invalid_output"
            except json.JSONDecodeError:
                words = re.findall(r"[a-z]+", content.lower())
                allowed = [c for c in ["billing", "delivery", "technical", "fraud"] if c in words]
                if len(allowed) == 1:
                    category = allowed[0]
                    resp_status = "classified_keyword"
                else:
                    resp_status = "flagged_invalid_output"

            seg_row = segments[segments["customer_id"] == cid]
            seg_name = seg_row["segment_name"].iloc[0] if len(seg_row) > 0 else ""
            is_prem = cid in prem_set

            if resp_status in ["flagged_invalid_output"]:
                route, priority = "human_review", "NORMAL"
            elif category == "fraud":
                route, priority = "security_team", "HIGH"
            elif is_prem:
                route, priority = "premium_desk", "HIGH"
            else:
                route, priority = "support_queue", "NORMAL"

            results.append({
                "ticket_id": tid, "customer_id": cid, "status": resp_status,
                "category": category, "segment_name": seg_name, "priority": priority, "route": route, "tokens_used": usage
            })

        except Exception as e:
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "error",
                "category": "", "segment_name": "", "priority": "NORMAL", "route": "human_review", "tokens_used": 0
            })

    os.makedirs("outputs", exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df.to_csv("outputs/triage_results.csv", index=False)
    res_df.head(50).to_csv("outputs/triage_budget_demo.csv", index=False)

    print(f"Total redactor substitutions: {sub_counts}")
    print("Saved outputs/triage_results.csv and outputs/triage_budget_demo.csv")

if __name__ == "__main__":
    run_triage_pipeline()