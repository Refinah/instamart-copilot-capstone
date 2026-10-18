import sys
import os

# Ensure project root is in Python path for mock_llm import
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import re
import json
import sqlite3
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

POL_DIR = "data/policies"
DB_PATH = "data/instamart.db"

# --- Task 1: Chunking ---
def chunk_documents():
    chunks = []
    for fn in sorted(os.listdir(POL_DIR)):
        if not fn.endswith(".txt"):
            continue
        path = os.path.join(POL_DIR, fn)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        words = text.split()
        chunk_size, overlap = 40, 10
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append({"source": fn, "text": chunk_text})
            if end == len(words):
                break
            start += chunk_size - overlap
    return chunks

CHUNKS = chunk_documents()
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform([c["text"] for c in CHUNKS])

def retrieve(query, k=3):
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, chunk_vectors).flatten()
    top_indices = scores.argsort()[::-1][:k]
    results = []
    for idx in top_indices:
        results.append({
            "source": CHUNKS[idx]["source"],
            "score": float(scores[idx]),
            "text": CHUNKS[idx]["text"]
        })
    return results

# --- Task 5: Define Tools ---
def get_order(order_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id=?", (order_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}

def search_policy(query):
    res = retrieve(query, k=3)
    return {"results": res}

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None or delivery_minutes > 45:
        return {"refund_inr": float(order_amount)}
    return {"refund_inr": 0.0}

def log_ledger(action_type, order_id, refund_inr):
    os.makedirs("outputs", exist_ok=True)
    ledger_path = "outputs/refund_ledger.jsonl"
    entry = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "action": action_type, "order_id": order_id, "refund_inr": refund_inr}
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def issue_refund(order_id, refund_inr):
    log_ledger("ISSUE", order_id, refund_inr)
    return {"refund_inr": float(refund_inr)}

def escalate_to_human(order_id, refund_inr):
    log_ledger("ESCALATE", order_id, refund_inr)
    return {"refund_inr": float(refund_inr)}

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

# --- Task 6 & 7: ReAct Loop & Safety Guardrails ---
def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10, run_id=None):
    start_time = time.monotonic()
    messages = [
        {"role": "system", "content": "AGENT: You are an Instamart refund assistant. Use tools to verify orders, check policies, calculate refunds, and issue or escalate them."},
        {"role": "user", "content": user_message}
    ]
    
    tokens_used = 0
    steps = []
    executed_tools = []
    state_dir = "state"
    os.makedirs(state_dir, exist_ok=True)

    for iteration in range(max_iterations):
        if (time.monotonic() - start_time) >= time_limit_seconds:
            return {"status": "FALLBACK_TIMEOUT", "final_answer": "Handoff: Timeout exceeded.", "steps": steps, "llm_calls": iteration, "tokens_used": tokens_used}

        resp = mock_llm.chat(messages, tools=TOOL_MAP)
        usage = resp.usage.total_tokens
        tokens_used += usage

        if tokens_used > token_limit:
            return {"status": "FALLBACK_TOKEN_LIMIT", "final_answer": "Handoff: Token limit exceeded.", "steps": steps, "llm_calls": iteration + 1, "tokens_used": tokens_used}

        msg = resp.choices[0].message
        
        if not msg.tool_calls:
            final_ans = msg.content
            status = final_ans.split(":")[0].strip() if ":" in final_ans else "DENIED"
            return {"status": status, "final_answer": final_ans, "steps": steps, "llm_calls": iteration + 1, "tokens_used": tokens_used}

        messages.append({"role": "assistant", "content": msg.content, "tool_calls": [tc.__dict__ for tc in msg.tool_calls]})
        
        for tc in msg.tool_calls:
            t_name = tc.name
            t_args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
            
            func = TOOL_MAP.get(t_name)
            res = func(**t_args) if func else {"error": "unknown tool"}
            
            executed_tools.append(t_name)
            res_str = json.dumps(res)
            
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": t_name, "content": res_str})
            steps.append({"tool": t_name, "arguments": t_args, "observation": res})

        if run_id:
            ckpt = {"run_id": run_id, "scenario_id": scenario_id, "messages": messages, "tokens_used": tokens_used, "step": iteration, "executed_tools": executed_tools}
            with open(os.path.join(state_dir, f"{run_id}.json"), "w", encoding="utf-8") as f:
                json.dump(ckpt, f)

    return {"status": "FALLBACK_MAX_ITERATIONS", "final_answer": "Handoff: Max iterations reached.", "steps": steps, "llm_calls": max_iterations, "tokens_used": tokens_used}

if __name__ == "__main__":
    scenarios = pd.read_csv("data/raw/agent_scenarios.csv")
    os.makedirs("outputs", exist_ok=True)
    runs = []
    for _, row in scenarios.iterrows():
        res = run_agent(row["message"], row["scenario_id"])
        runs.append({
            "scenario_id": row["scenario_id"],
            "status": res["status"],
            "final_answer": res["final_answer"],
            "llm_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"],
            "steps": res["steps"]
        })
    with open("outputs/agent_runs.jsonl", "w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")
    print("Agent scenarios executed and saved to outputs/agent_runs.jsonl")