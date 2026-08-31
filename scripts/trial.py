#!/usr/bin/env python3
"""trial.py - run a short, time-boxed GRPO trial and report whether it looks promising.

Standalone: needs only taco_alley.db and data/*.jsonl. Everything else (SQL
sandbox, reward functions, prompt construction, evaluation) is defined here.

    python trial.py --config trials/baseline.yaml --minutes 15

What it does, in order:

  1. Validates every key in your config against the *installed* GRPOConfig and
     stops with a suggestion if a field does not exist.
  2. Pre-flight: samples G completions at your training sampling settings and
     reports the fraction of groups that will produce a non-zero advantage.
     Aborts if the model cannot produce runnable SQL - no point training.
  3. Baseline eval on a small held-out slice.
  4. Trains until the clock runs out.
  5. Post eval, then writes results/<name>.json and prints a comparison table
     across every trial you have run so far.

Compare previous runs without training:  python trial.py --compare
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sqlite3
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

# --------------------------------------------------------------------------
# Config file
# --------------------------------------------------------------------------
DEFAULT_MODEL_KWARGS = {"dtype": "bfloat16", "attn_implementation": "sdpa"}


def load_config(path: str) -> dict:
    text = Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError:
            sys.exit("pyyaml not installed - use a .json config or `pip install pyyaml`")
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    cfg.setdefault("name", Path(path).stem)
    cfg.setdefault("model", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    cfg.setdefault("model_kwargs", DEFAULT_MODEL_KWARGS)
    cfg.setdefault("lora", {"r": 32, "lora_alpha": 64, "lora_dropout": 0.05,
                            "target_modules": "all-linear", "bias": "none",
                            "task_type": "CAUSAL_LM"})
    cfg.setdefault("grpo", {})
    return cfg


def build_grpo_config(kwargs: dict, output_dir: str):
    """Check every key against the installed GRPOConfig before constructing it."""
    from trl import GRPOConfig

    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    unknown = sorted(set(kwargs) - valid)
    if unknown:
        print("\nconfig keys not present in your installed TRL:", file=sys.stderr)
        for u in unknown:
            near = sorted(v for v in valid
                          if any(w in v for w in u.split("_") if len(w) > 3))
            hint = f"  -> did you mean: {', '.join(near[:6])}?" if near else ""
            print(f"  {u!r}{hint}", file=sys.stderr)
        sys.exit(1)

    if "reward_weights" not in kwargs:
        print("note: reward_weights not set - every reward function gets weight 1.0. "
              "Set it explicitly in the config so trials stay comparable.")
    forced = {"output_dir": output_dir, "report_to": "none",
              "remove_unused_columns": False, "save_strategy": "no",
              "dataloader_num_workers": 0}
    for k, v in forced.items():
        if kwargs.get(k) not in (None, v) and k in kwargs:
            print(f"note: overriding {k}={kwargs[k]!r} with {v!r} for this trial")
        kwargs[k] = v
    return GRPOConfig(**kwargs)


# --------------------------------------------------------------------------
# SQL sandbox
# --------------------------------------------------------------------------
DB_PATH = "taco_alley.db"
TODAY = "2026-08-24"
QUERY_STEP_BUDGET = 50_000_000     # above the heaviest gold query (~13M)
QUERY_TIMEOUT_S = 2.0

_local = threading.local()


def get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
        conn.execute("PRAGMA query_only = ON")
        _local.conn = conn
    return conn


THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
FENCE_RE = re.compile(r"```(?:sql)?\s*(.+?)```", re.S | re.I)
SELECT_RE = re.compile(r"\b(SELECT|WITH)\b.*", re.S | re.I)
FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA|VACUUM)\b", re.I)


def extract_sql(text: str):
    text = THINK_RE.sub("", text)
    m = FENCE_RE.search(text)
    candidate = m.group(1) if m else None
    if candidate is None:
        m = SELECT_RE.search(text)
        candidate = m.group(0) if m else None
    if candidate is None:
        return None
    return candidate.strip().rstrip(";").strip()


def is_single_read_only(sql: str) -> bool:
    if not sql or FORBIDDEN.search(sql):
        return False
    if ";" in sql.strip().rstrip(";"):
        return False
    return bool(re.match(r"\s*(SELECT|WITH)\b", sql, re.I))


def canon(v):
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, int):
        return float(v)
    if isinstance(v, str):
        return v.strip()
    return v


@lru_cache(maxsize=8192)
def run_query(sql: str):
    """-> (status, rows, n_cols). n_cols comes from cursor.description, so an
    empty result set still reports its shape. Cached: every reward function
    asks for the same gold result, and all G members of a group share one."""
    if not is_single_read_only(sql):
        return "unsafe", (), 0
    conn = get_conn()
    deadline = time.time() + QUERY_TIMEOUT_S
    state = {"steps": 0}

    def guard():
        state["steps"] += 1
        return 1 if (state["steps"] > QUERY_STEP_BUDGET or time.time() > deadline) else 0

    conn.set_progress_handler(guard, 1000)
    try:
        cur = conn.execute(sql)
        rows = tuple(tuple(canon(v) for v in r) for r in cur.fetchmany(200))
        return "ok", rows, (len(cur.description) if cur.description else 0)
    except Exception:
        return "error", (), 0
    finally:
        conn.set_progress_handler(None, 0)


def result_signature(sql: str, ordered: bool):
    status, rows, _ = run_query(sql)
    if status != "ok":
        return status, None
    return "ok", (rows if ordered else tuple(sorted(rows, key=repr)))


def is_match(text: str, gold_sql: str) -> bool:
    ordered = bool(re.search(r"\bORDER\s+BY\b", gold_sql, re.I))
    _s, g_rows = result_signature(gold_sql, ordered)
    status, p_rows = result_signature(extract_sql(text) or "", ordered)
    return status == "ok" and g_rows is not None and p_rows == g_rows


TABLE_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.I)
NEAR_MISS_CEILING = 0.8


@lru_cache(maxsize=1)
def known_tables() -> frozenset:
    return frozenset(r[0].lower() for r in get_conn().execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"))


def tables_used(sql: str) -> frozenset:
    """Real table names only - intersecting with the schema drops aliases,
    subquery names and anything invented, so this cannot be gamed."""
    return frozenset(t.lower() for t in TABLE_RE.findall(sql or "")) & known_tables()


def num_close(a: float, b: float) -> float:
    """Relative closeness floored at 0. 17 vs 16 -> 0.94; 17 vs 1969 -> 0.01."""
    if a == b:
        return 1.0
    return max(0.0, 1.0 - abs(a - b) / max(abs(a), abs(b), 1e-9))


def cell_sim(a, b) -> float:
    if isinstance(a, float) and isinstance(b, float):
        return num_close(a, b)
    if isinstance(a, str) and isinstance(b, str):
        return 1.0 if a == b else (0.5 if a.lower() == b.lower() else 0.0)
    return 1.0 if a == b else 0.0


def row_sim(r1, r2) -> float:
    if not r1 or not r2:
        return 0.0
    n, m = len(r1), len(r2)
    return (sum(cell_sim(a, b) for a, b in zip(r1, r2)) / min(n, m)) * (min(n, m) / max(n, m))


def content_score(pred, gold) -> float:
    """Soft F1: greedily pair each gold row with its best unused pred row."""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    if len(pred) > 50 or len(gold) > 50:          # keep the cost bounded
        a, b = {repr(r) for r in pred}, {repr(r) for r in gold}
        return len(a & b) / len(a | b)
    used, matched = set(), 0.0
    for grow in gold:
        best, best_i = 0.0, None
        for i, prow in enumerate(pred):
            if i in used:
                continue
            sim = row_sim(prow, grow)
            if sim > best:
                best, best_i = sim, i
        if best_i is not None:
            used.add(best_i)
            matched += best
    return 2 * matched / (len(pred) + len(gold))


def shape_score(pred, gold, pcols: int, gcols: int) -> float:
    col = 1.0 if pcols == gcols else max(0.0, 1 - abs(pcols - gcols) / max(gcols, 1))
    if not pred and not gold:
        rows = 1.0
    elif not pred or not gold:
        rows = 0.0
    else:
        rows = min(len(pred), len(gold)) / max(len(pred), len(gold))
    return 0.5 * col + 0.5 * rows


def partial_credit(pred_sql: str, gold_sql: str) -> float:
    """Graded 0-1 credit that still discriminates when NO rows match.

    Plain Jaccard was binary here: most gold answers are a single scalar, so
    {(17,)} vs {(19,)} shared no rows and scored 0 - identical to a query that
    returned 43,841 rows from the wrong table. Every executes-but-wrong
    completion tied, which is what left a third of groups with no gradient.

    Exact matches score 1.0; everything else is scaled to at most
    NEAR_MISS_CEILING, so the model cannot learn to be approximately right."""
    p_status, p_rows, p_cols = run_query(pred_sql)
    g_status, g_rows, g_cols = run_query(gold_sql)
    if p_status != "ok" or g_status != "ok":
        return 0.0
    gold_tables, pred_tables = tables_used(gold_sql), tables_used(pred_sql)
    if not pred_tables & gold_tables:
        return 0.0                                # gate: "SELECT 1" farms nothing
    ordered = bool(re.search(r"\bORDER\s+BY\b", gold_sql, re.I))
    if (p_rows == g_rows if ordered
            else sorted(p_rows, key=repr) == sorted(g_rows, key=repr)):
        return 1.0
    tbl = len(pred_tables & gold_tables) / len(pred_tables | gold_tables)
    return NEAR_MISS_CEILING * (0.20 * tbl
                                + 0.30 * shape_score(p_rows, g_rows, p_cols, g_cols)
                                + 0.50 * content_score(p_rows, g_rows))


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------
ENUM_HINTS = {
    ("stores", "state"):              ["PA", "NJ", "DE"],
    ("stores", "region"):             ["Northeast", "Mid-Atlantic"],
    ("employees", "role"):            ["Crew", "Shift Lead", "Assistant Manager", "General Manager"],
    ("menu_items", "category"):       ["Taco", "Burrito", "Bowl", "Side", "Drink", "Dessert"],
    ("loyalty_members", "tier"):      ["Bronze", "Silver", "Gold"],
    ("orders", "channel"):            ["In-Store", "Drive-Thru", "Mobile App", "Delivery"],
    ("orders", "payment_method"):     ["Card", "Cash", "Mobile Wallet", "Gift Card"],
    ("orders", "status"):             ["Completed", "Refunded", "Cancelled"],
    ("customer_tickets", "channel"):  ["Phone", "Email", "App", "In-Person", "Social"],
    ("customer_tickets", "category"): ["Complaint", "Compliment", "Question", "Refund Request"],
    ("customer_tickets", "priority"): ["Low", "Medium", "High"],
    ("customer_tickets", "status"):   ["Open", "In Progress", "Resolved", "Closed"],
}


def render_schema() -> str:
    conn = get_conn()
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    out = []
    for t in tables:
        cols = []
        for _, name, ctype, _, _, pk in conn.execute(f"PRAGMA table_info({t})"):
            bit = f"{name} {ctype}" + (" PK" if pk else "")
            if (t, name) in ENUM_HINTS:
                bit += " in (" + ", ".join(repr(v) for v in ENUM_HINTS[(t, name)]) + ")"
            cols.append(bit)
        out.append(f"{t}(\n  " + ",\n  ".join(cols) + "\n)")
    fks = [f"{t}.{r[3]} -> {r[2]}.{r[4]}"
           for t in tables for r in conn.execute(f"PRAGMA foreign_key_list({t})")]
    return "\n".join(out) + "\n\nForeign keys:\n" + "\n".join(f"  {f}" for f in fks)


def system_prompt() -> str:
    return f"""You are a SQL analyst for Taco Alley, a taco restaurant chain.
Convert the user's question into a single SQLite SELECT statement.

Schema:
{render_schema()}

Rules:
- Today's date is {TODAY}. Resolve relative dates against it.
- Timestamps are TEXT in 'YYYY-MM-DD HH:MM:SS' form; dates are 'YYYY-MM-DD'. Half-open ranges
  (>= start AND < end) are the safe way to filter a month.
- Revenue and sales questions mean orders with status = 'Completed' unless stated otherwise.
- Return exactly one SELECT statement. No INSERT, UPDATE, DELETE, or DDL.
- Reply with only the query inside a ```sql fenced block. No explanation."""


# --------------------------------------------------------------------------
# Rewards
# --------------------------------------------------------------------------
def _texts(completions):
    return [c[0]["content"] if isinstance(c, list) else c for c in completions]


def reward_format(completions, gold_sql, **kw):
    out = []
    for text in _texts(completions):
        sql = extract_sql(text)
        if sql is None:
            out.append(0.0)
        elif not is_single_read_only(sql):
            out.append(-1.0)
        elif FENCE_RE.search(THINK_RE.sub("", text)):
            out.append(1.0)
        else:
            out.append(0.5)
    return out


def reward_executes(completions, gold_sql, **kw):
    return [1.0 if run_query(extract_sql(t) or "")[0] == "ok" else 0.0
            for t in _texts(completions)]


def reward_result_match(completions, gold_sql, **kw):
    return [1.0 if is_match(t, g) else 0.0 for t, g in zip(_texts(completions), gold_sql)]


def reward_partial_credit(completions, gold_sql, **kw):
    return [partial_credit(extract_sql(t) or "", g)
            for t, g in zip(_texts(completions), gold_sql)]


# Order matters: reward_weights in the config lines up with this list.
REWARD_FUNCS = [reward_format, reward_executes, reward_result_match, reward_partial_credit]


# --------------------------------------------------------------------------
# Generation helpers
# --------------------------------------------------------------------------
def chat(tok, row, cfg_kwargs):
    msgs = row["prompt"]
    extra = cfg_kwargs.get("chat_template_kwargs") or {}
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **extra)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def preflight(model, tok, rows, g: dict, k: int, max_new: int):
    """Predict frac_reward_zero_std before spending the clock on training."""
    import torch

    pad = tok.pad_token_id or tok.eos_token_id
    exec_ok = trunc = useful = 0
    lens, distinct, samples = [], [], []
    with torch.no_grad():
        for row in rows:
            enc = tok(chat(tok, row, g), return_tensors="pt").to(model.device)
            plen = enc["input_ids"].shape[1]
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, use_cache=True,
                                 temperature=g.get("temperature", 1.0),
                                 top_p=g.get("top_p", 1.0), top_k=g.get("top_k", 0) or 0,
                                 num_return_sequences=k, pad_token_id=pad)
            texts = tok.batch_decode(out[:, plen:], skip_special_tokens=True)
            hits, sqls = 0, set()
            for i, text in enumerate(texts):
                n = int((out[i, plen:] != pad).sum())
                lens.append(n)
                trunc += int(n >= max_new)
                sqls.add(extract_sql(text))
                if run_query(extract_sql(text) or "")[0] == "ok":
                    exec_ok += 1
                if is_match(text, row["gold_sql"]):
                    hits += 1
            distinct.append(len(sqls))
            useful += int(0 < hits < k)
            if len(samples) < 3:
                samples.append({"question": row["question"], "completion": texts[0][:600]})
    n = len(rows) * k
    return dict(exec_rate=exec_ok / n, useful=useful / len(rows),
                distinct=sum(distinct) / len(distinct), mean_len=sum(lens) / len(lens),
                trunc_rate=trunc / n, samples=samples)


def mini_eval(model, tok, rows, g: dict, max_new: int, batch_size: int = 4, n_samples: int = 4):
    """Greedy exec accuracy, plus a few completions so failures stay legible.

    Sampling a handful here beats printing completions every logging step: you
    see what the model does at the end of the run, not forty snapshots of the
    middle of it."""
    import torch

    pad = tok.pad_token_id or tok.eos_token_id
    tok.padding_side = "left"
    hits, samples = [], []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            enc = tok([chat(tok, r, g) for r in chunk], return_tensors="pt",
                      padding=True).to(model.device)
            plen = enc["input_ids"].shape[1]
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 use_cache=True, pad_token_id=pad)
            for text, row in zip(tok.batch_decode(out[:, plen:], skip_special_tokens=True), chunk):
                ok = is_match(text, row["gold_sql"])
                hits.append(1.0 if ok else 0.0)
                # keep misses preferentially - the hits all look the same
                if len(samples) < n_samples and (not ok or not samples):
                    samples.append({"question": row["question"], "matched": ok,
                                    "sql": (extract_sql(text) or text)[:400],
                                    "gold": row["gold_sql"][:400]})
    return sum(hits) / max(len(hits), 1), samples


# --------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------
def make_callbacks(seconds: float):
    from transformers import TrainerCallback

    class TimeLimit(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            self.deadline = time.time() + seconds

        def on_step_end(self, args, state, control, **kw):
            if time.time() > self.deadline:
                print(f"\n[time limit reached at step {state.global_step}]")
                control.should_training_stop = True
            return control

    class Progress(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            self.t0 = time.time()

        def on_log(self, args, state, control, logs=None, **kw):
            if not logs or "reward" not in logs:
                return
            m = lambda k, d=float("nan"): logs.get(k, d)
            print(f"  step {state.global_step:>4} "
                  f"{(time.time()-self.t0)/60:>5.1f}m "
                  f"reward={m('reward'):>6.3f} "
                  f"match={m('rewards/reward_result_match/mean'):>5.2f} "
                  f"exec={m('rewards/reward_executes/mean'):>5.2f} "
                  f"zero_std={m('frac_reward_zero_std'):>5.2f} "
                  f"len={m('completions/mean_length'):>6.1f} "
                  f"ent={m('entropy'):>6.3f}", flush=True)

    return [TimeLimit(), Progress()]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
TRACK = ["reward", "rewards/reward_result_match/mean", "rewards/reward_executes/mean",
         "rewards/reward_format/mean", "rewards/reward_partial_credit/mean",
         "frac_reward_zero_std", "completions/mean_length",
         "completions/clipped_ratio", "entropy", "reward_std", "grad_norm"]


def summarize(log_history: list[dict]) -> dict:
    out = {}
    for key in TRACK:
        vals = [e[key] for e in log_history if isinstance(e.get(key), (int, float))]
        if len(vals) < 2:
            continue
        n = max(1, len(vals) // 4)
        out[key] = {"first": sum(vals[:n]) / n, "last": sum(vals[-n:]) / n,
                    "max": max(vals), "n": len(vals)}
        out[key]["delta"] = out[key]["last"] - out[key]["first"]
    return out


def verdict(pre: dict, summ: dict, eval_delta: float) -> list[str]:
    """Blunt heuristics, so a 15-minute trial ends with a call rather than a table."""
    notes = []
    zs = summ.get("frac_reward_zero_std", {})
    match = summ.get("rewards/reward_result_match/mean", {})
    ent = summ.get("entropy", {})
    clip = summ.get("completions/clipped_ratio", {})
    if pre["exec_rate"] < 0.15:
        notes.append(f"FAIL sampling: only {pre['exec_rate']:.0%} of samples run at all")
    if zs and zs["last"] > 0.6:
        notes.append(f"STARVED: {zs['last']:.0%} of groups give no gradient - raise num_generations")
    if ent and ent["last"] < 0.05:
        notes.append("COLLAPSED or fully masked: entropy ~0 - check truncation and sampling")
    if ent and ent["last"] > 2.0:
        notes.append("TOO RANDOM: entropy > 2 - lower temperature or restore top_k")
    if clip and clip["last"] > 0.25:
        notes.append(f"TRUNCATING: {clip['last']:.0%} clipped - raise max_completion_length")
    if match and match["delta"] > 0.03:
        notes.append(f"PROMISING: result_match +{match['delta']:.3f} during the trial")
    if eval_delta > 0.02:
        notes.append(f"PROMISING: held-out eval +{eval_delta:.1%}")
    if not notes:
        notes.append("INCONCLUSIVE: nothing moved much - run longer or change one variable")
    return notes


def compare(results_dir: Path):
    rows = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    if not rows:
        print("no trial results yet")
        return
    hdr = (f"{'trial':<22}{'G':>4}{'lr':>9}{'temp':>6}{'steps':>7}"
           f"{'zero_std':>10}{'match':>8}{'eval':>8}{'Δeval':>8}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        g = r["grpo"]
        s = r["summary"].get("rewards/reward_result_match/mean", {})
        z = r["summary"].get("frac_reward_zero_std", {})
        print(f"{r['name']:<22}{g.get('num_generations', 0):>4}"
              f"{g.get('learning_rate', 0):>9.1e}{g.get('temperature', 1.0):>6.2f}"
              f"{r['steps']:>7}{z.get('last', float('nan')):>10.2f}"
              f"{s.get('last', float('nan')):>8.3f}{r['eval_after']:>8.1%}"
              f"{r['eval_after']-r['eval_before']:>+8.1%}")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--minutes", type=float, default=15.0, help="training time budget")
    ap.add_argument("--db", default="taco_alley.db")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--eval-split", default="eval_templates")
    ap.add_argument("--preflight-n", type=int, default=16, help="prompts for the pre-flight probe")
    ap.add_argument("--eval-n", type=int, default=24, help="prompts for before/after eval")
    ap.add_argument("--force", action="store_true", help="train even if pre-flight fails")
    ap.add_argument("--compare", action="store_true", help="print past trials and exit")
    args = ap.parse_args()

    results_dir = Path(args.results)
    results_dir.mkdir(exist_ok=True)
    if args.compare:
        compare(results_dir)
        return
    if not args.config:
        ap.error("--config is required (or use --compare)")

    global DB_PATH
    DB_PATH = args.db

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer

    cfg = load_config(args.config)
    g = dict(cfg["grpo"])
    trial_dir = results_dir / cfg["name"]
    grpo_args = build_grpo_config(dict(g), str(trial_dir))
    max_new = g.get("max_completion_length", 256)

    # ---- data ----
    SYSTEM = system_prompt()

    def load(split):
        rows = [json.loads(l) for l in open(f"{args.data_dir}/{split}.jsonl")]
        for r in rows:
            r["prompt"] = [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": r["question"]}]
        return rows

    train_rows = load("train")
    eval_rows = load(args.eval_split)[:args.eval_n]
    train_ds = Dataset.from_list(train_rows).shuffle(seed=0)

    # ---- model ----
    mk = dict(cfg["model_kwargs"])
    if isinstance(mk.get("dtype"), str):
        mk["dtype"] = getattr(torch, mk["dtype"])
    print(f"loading {cfg['model']} ...")
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], device_map="cuda", **mk)

    plens = [len(tok(chat(tok, r, g))["input_ids"]) for r in train_rows[:50]]
    print(f"prompt tokens: max={max(plens)} mean={sum(plens)/len(plens):.0f}")

    # ---- pre-flight ----
    k = g.get("num_generations", 8)
    print(f"\npre-flight: {args.preflight_n} prompts x {k} samples "
          f"(temp={g.get('temperature', 1.0)}, top_p={g.get('top_p', 1.0)}, top_k={g.get('top_k', 0)})")
    model.eval()
    pre = preflight(model, tok, train_rows[:args.preflight_n], g, k, max_new)
    print(f"  exec_rate={pre['exec_rate']:.1%}  distinct={pre['distinct']:.1f}/{k}  "
          f"useful_groups={pre['useful']:.1%}  mean_len={pre['mean_len']:.0f}  "
          f"trunc={pre['trunc_rate']:.1%}")
    print(f"  -> expect frac_reward_zero_std ~ {1 - pre['useful']:.2f}")
    print("\n  sample completion:\n    " + pre["samples"][0]["completion"][:300].replace("\n", "\n    "))
    # Gate on the quantity that decides whether GRPO can learn at all: the
    # fraction of groups that will produce a non-zero advantage. A low exec_rate
    # is worth knowing but is not disqualifying on its own - a model that is
    # wrong in varied ways still gives GRPO something to work with.
    if not args.force:
        if pre["exec_rate"] < 0.15:
            sys.exit(f"\npre-flight failed: exec_rate={pre['exec_rate']:.1%} - the model is not "
                     "producing runnable SQL at these sampling settings. Check temperature/top_k "
                     "and the sample completion above. Override with --force.")
        if pre["useful"] < 0.10:
            sys.exit(f"\npre-flight failed: only {pre['useful']:.0%} of groups would produce a "
                     "gradient. Raise num_generations, or the task is out of reach for this model. "
                     "Override with --force.")
    if pre["useful"] < 0.35:
        print(f"  NOTE: {pre['useful']:.0%} useful groups is thin but trainable. Most groups are "
              f"all-wrong, so raising num_generations buys more than tuning the optimizer.")

    # ---- baseline ----
    print(f"\nbaseline eval on {len(eval_rows)} {args.eval_split} prompts ...")
    eval_before, _ = mini_eval(model, tok, eval_rows, g, max_new)
    print(f"  exec accuracy: {eval_before:.1%}")
    torch.cuda.empty_cache()

    # ---- train ----
    print(f"\ntraining for up to {args.minutes:.0f} minutes "
          f"(stops at the first step boundary past the limit)\n")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=REWARD_FUNCS,
        args=grpo_args,
        train_dataset=train_ds,
        peft_config=LoraConfig(**cfg["lora"]),
        callbacks=make_callbacks(args.minutes * 60),
    )
    print(f"  generation_batch_size = {getattr(trainer.args, 'generation_batch_size', '?')}")
    print(f"  prompts per optimizer step = "
          f"{grpo_args.per_device_train_batch_size * grpo_args.gradient_accumulation_steps // k}\n")
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    train_min = (time.time() - t0) / 60

    # ---- after ----
    trainer.model.eval()
    eval_after, after_samples = mini_eval(trainer.model, tok, eval_rows, g, max_new)
    summ = summarize(trainer.state.log_history)

    record = dict(name=cfg["name"], model=cfg["model"], grpo=g, lora=cfg["lora"],
                  minutes=round(train_min, 1), steps=trainer.state.global_step,
                  preflight={k2: v for k2, v in pre.items() if k2 != "samples"},
                  samples_before=pre["samples"], samples_after=after_samples,
                  eval_split=args.eval_split,
                  eval_before=eval_before, eval_after=eval_after, summary=summ,
                  peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                  log_history=trainer.state.log_history)
    (results_dir / f"{cfg['name']}.json").write_text(json.dumps(record, indent=1))

    # ---- report ----
    print(f"\n{'metric':<42}{'first':>10}{'last':>10}{'delta':>10}")
    for key in TRACK:
        if key in summ:
            s = summ[key]
            print(f"{key:<42}{s['first']:>10.3f}{s['last']:>10.3f}{s['delta']:>+10.3f}")
    print(f"\n{'eval ('+args.eval_split+')':<42}{eval_before:>10.1%}{eval_after:>10.1%}"
          f"{eval_after-eval_before:>+10.1%}")
    print(f"{'steps / minutes / peak GB':<42}{trainer.state.global_step:>10}"
          f"{train_min:>10.1f}{record['peak_gb']:>10.1f}")
    print("\npost-training samples (misses first):")
    for smp in after_samples[:2]:
        print(f"  Q    : {smp['question']}")
        print(f"  {'HIT ' if smp['matched'] else 'MISS'} : {smp['sql'][:220]}")
        if not smp["matched"]:
            print(f"  gold : {smp['gold'][:220]}")
        print()

    print("verdict:")
    for line in verdict(pre, summ, eval_after - eval_before):
        print("  " + line)
    compare(results_dir)


if __name__ == "__main__":
    main()
