"""Cross-judge probe: re-judge the frozen Phase 1 captions with Gemini 2.5 Flash.

Off-cluster diagnostic. Reuses the released judge prompt, the field-for-field
`GalaxyDecisionTree` schema, the released decision-path construction, and the
same audit + paired-bootstrap scoring as ``phase1_v3`` -- the ONLY change is the
judge: Gemma+XGrammar is replaced by ``google/gemini-2.5-flash`` through
OpenRouter with provider-native JSON-schema structured output.

Purpose: (1) does the GPT>Qwen ranking survive an independent, stronger judge?
(2) does the ``edge-on-bulge`` 0% failure persist under native structured
decoding (=> intrinsic flat-schema fragility) or clear up (=> Gemma/XGrammar
amplification)?

The OpenRouter key is read from the environment or C:/Trung/Life/research/.env
and never printed. A hard cumulative-cost cap aborts runaway spend.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

from aion_reimp.caption_audit import (  # noqa: E402
    audit_metrics,
    audit_rows,
    paired_accuracy_delta,
    paired_path_score_delta,
    path_audit_metrics,
    path_audit_rows,
    write_audit,
    write_path_audit,
)
from aion_reimp.morphology import (  # noqa: E402
    GalaxyDecisionTree,
    SCHEMA_SHA256,
    parse_morphology_response,
)

MODEL_ID = "google/gemini-2.5-flash"
BASE_URL = "https://openrouter.ai/api/v1"
PROMPT_FILE = REPO / "data" / "prompts" / "galaxybench_galaxyzoo_judge_released_v1.txt"
FROZEN_PROMPT_SHA256 = "ffcf61ac2eadde5c6a80a5e7b15ef4d29c30533189bd8f8e0c2abd19d84fbe10"
LABELS = REPO / "data" / "caption_screen_64" / "caption_screen_labels.parquet"
ARMS = {
    "gpt41mini": REPO / "results" / "phase1_v3" / "gpt41mini_descriptions.jsonl",
    "qwen3vl_8b": REPO / "results" / "phase1_v3" / "qwen_descriptions.jsonl",
}
SEED = 20260713
BOOTSTRAP = 2000
COST_CAP_USD = 1.00


def load_key() -> str:
    import os

    key = os.environ.get("OPEN_ROUTER_KEY")
    if key:
        return key
    for env_path in (Path("C:/Trung/Life/research/.env"), REPO.parents[1] / ".env"):
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("OPEN_ROUTER_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPEN_ROUTER_KEY not found in env or C:/Trung/Life/research/.env")


def strict_schema() -> Dict[str, Any]:
    """Sanitize the pydantic schema for OpenAI-style strict structured output."""
    schema = copy.deepcopy(GalaxyDecisionTree.model_json_schema())

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            node.pop("title", None)
            if "properties" in node:
                node["required"] = list(node["properties"].keys())
                node["additionalProperties"] = False
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    return schema


def build_prompt_template() -> str:
    template = PROMPT_FILE.read_text(encoding="utf-8").rstrip("\r\n")
    actual = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if actual != FROZEN_PROMPT_SHA256:
        raise SystemExit(f"Released judge prompt hash mismatch: {actual}")
    return template


def load_descriptions(path: Path) -> List[Dict[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        rows.append({"object_id": str(obj["object_id"]), "description": str(obj["description"])})
    ids = [r["object_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"Duplicate object_id in {path}")
    return rows


def read_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(json.loads(line)["object_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def call_gemini(client, template: str, description: str, schema: Dict[str, Any],
                strict: bool, disable_thinking: bool) -> tuple[str, Dict[str, Any]]:
    prompt = template.replace("{description}", description)
    extra_body: Dict[str, Any] = {"usage": {"include": True}}
    if disable_thinking:
        extra_body["reasoning"] = {"max_tokens": 0}
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_completion_tokens=1024,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "galaxy_decision_tree", "strict": strict, "schema": schema},
        },
        extra_body=extra_body,
    )
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty response content")
    usage = response.usage.model_dump() if response.usage is not None else {}
    meta = {
        "request_id": response.id,
        "returned_model": response.model,
        "provider": getattr(response, "provider", None),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "reported_cost_usd": usage.get("cost"),
    }
    return content, meta


def judge_arm(client, arm: str, desc_path: Path, out_dir: Path, template: str,
              schema: Dict[str, Any], strict: bool, disable_thinking: bool,
              limit: Optional[int], cost_state: Dict[str, float]) -> Dict[str, Any]:
    raw_path = out_dir / f"{arm}_judge_raw.jsonl"
    ext_path = out_dir / f"{arm}_extractions.jsonl"
    err_path = out_dir / f"{arm}_errors.jsonl"
    rows = load_descriptions(desc_path)
    if limit is not None:
        rows = rows[:limit]
    done = read_done(raw_path) | read_done(err_path)
    parse_errors = 0
    with raw_path.open("a", encoding="utf-8") as raw_h, \
            ext_path.open("a", encoding="utf-8") as ext_h, \
            err_path.open("a", encoding="utf-8") as err_h:
        for row in rows:
            oid = row["object_id"]
            if oid in done:
                continue
            content, meta = call_gemini(
                client, template, row["description"], schema, strict, disable_thinking
            )
            if meta["reported_cost_usd"] is not None:
                cost_state["total"] += float(meta["reported_cost_usd"])
            raw_h.write(json.dumps({"object_id": oid, "raw_response": content, **meta}, sort_keys=True) + "\n")
            raw_h.flush()
            try:
                result = parse_morphology_response(oid, strip_fences(content))
            except Exception as error:  # noqa: BLE001
                parse_errors += 1
                err_h.write(json.dumps({
                    "object_id": oid, "error_type": type(error).__name__,
                    "error": str(error), "raw_response": content,
                }, sort_keys=True) + "\n")
                err_h.flush()
                continue
            ext_h.write(json.dumps(result.as_record(), sort_keys=True) + "\n")
            ext_h.flush()
            print(f"  [{arm}] {oid}  cost_so_far=${cost_state['total']:.4f}", flush=True)
            if cost_state["total"] > COST_CAP_USD:
                raise SystemExit(f"Cost cap ${COST_CAP_USD} exceeded (${cost_state['total']:.4f})")
    return {"raw_path": raw_path, "ext_path": ext_path, "err_path": err_path,
            "parse_errors": parse_errors, "n_input": len(rows)}


def extractions_df(ext_path: Path) -> pd.DataFrame:
    records = [json.loads(l) for l in ext_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO / "results" / "phase1_gemini_crossjudge_v1")
    parser.add_argument("--limit", type=int, default=None, help="objects per arm (dry run)")
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("--thinking", dest="disable_thinking", action="store_false", default=True)
    parser.add_argument("--audit-only", action="store_true", help="skip judging; rebuild audits")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    template = build_prompt_template()
    schema = strict_schema()

    from openai import OpenAI

    client = OpenAI(api_key=load_key(), base_url=BASE_URL, timeout=120.0, max_retries=3)
    cost_state = {"total": 0.0}

    arm_out: Dict[str, Dict[str, Any]] = {}
    if not args.audit_only:
        for arm, desc_path in ARMS.items():
            print(f"Judging {arm} with {MODEL_ID} (strict={args.strict}, "
                  f"thinking={'off' if args.disable_thinking else 'on'})", flush=True)
            arm_out[arm] = judge_arm(client, arm, desc_path, out_dir, template, schema,
                                     args.strict, args.disable_thinking, args.limit, cost_state)

    # ---- audits (identical scoring to phase1_v3) ----
    labels = pd.read_parquet(LABELS)[["object_id", "decision_tree"]]
    labels["object_id"] = labels["object_id"].astype(str)
    arm_caps: Dict[str, pd.DataFrame] = {}
    path_rows: Dict[str, pd.DataFrame] = {}
    ans_rows: Dict[str, pd.DataFrame] = {}
    per_arm_meta: Dict[str, Any] = {}
    for arm in ARMS:
        caps = extractions_df(out_dir / f"{arm}_extractions.jsonl")
        caps["object_id"] = caps["object_id"].astype(str)
        arm_caps[arm] = caps
        lab = labels[labels["object_id"].isin(caps["object_id"])]
        write_path_audit(caps, lab, out_dir / f"{arm}_audit", BOOTSTRAP, SEED)
        write_audit(caps, lab, out_dir / f"{arm}_audit", BOOTSTRAP, SEED)
        path_rows[arm] = path_audit_rows(caps, lab)
        ans_rows[arm] = audit_rows(caps, lab)
        per_arm_meta[arm] = {"scored_objects": int(len(caps))}

    comparison = {
        "protocol": ("free-form descriptions judged by google/gemini-2.5-flash via OpenRouter "
                     "with native JSON-schema structured output; released GalaxyBench prompt, "
                     "field-for-field GalaxyDecisionTree schema, released path score"),
        "judge_model": {
            "model_id": MODEL_ID, "engine": "openrouter_json_schema",
            "strict": args.strict, "thinking": "off" if args.disable_thinking else "on",
        },
        "extractor_prompt_sha256": FROZEN_PROMPT_SHA256,
        "schema_sha256_local": SCHEMA_SHA256,
        "primary_metric": "released_decision_path_overlap",
        "secondary_metric": "per_question_accuracy",
        "seed": SEED, "bootstrap_samples": BOOTSTRAP,
        "reported_cost_usd_total": round(cost_state["total"], 6),
        "scored_objects": per_arm_meta,
        "released_path_overlap": {
            "gpt41mini": path_audit_metrics(path_rows["gpt41mini"], BOOTSTRAP, SEED),
            "qwen3vl_8b": path_audit_metrics(path_rows["qwen3vl_8b"], BOOTSTRAP, SEED),
            "paired_delta_gpt_minus_qwen": paired_path_score_delta(
                path_rows["qwen3vl_8b"], path_rows["gpt41mini"], BOOTSTRAP, SEED),
        },
        "per_question_diagnostic": {
            "gpt41mini": audit_metrics(ans_rows["gpt41mini"], BOOTSTRAP, SEED),
            "qwen3vl_8b": audit_metrics(ans_rows["qwen3vl_8b"], BOOTSTRAP, SEED),
            "paired_delta_gpt_minus_qwen": paired_accuracy_delta(
                ans_rows["qwen3vl_8b"], ans_rows["gpt41mini"], BOOTSTRAP, SEED),
        },
    }
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nTotal reported cost: ${cost_state['total']:.4f}")
    po = comparison["released_path_overlap"]
    print(f"PATH OVERLAP  gpt {po['gpt41mini']['mean_score']:.4f} | "
          f"qwen {po['qwen3vl_8b']['mean_score']:.4f} | "
          f"delta {po['paired_delta_gpt_minus_qwen']['point']:.4f} "
          f"{po['paired_delta_gpt_minus_qwen']['ci95']}")
    for arm in ARMS:
        eob = comparison["per_question_diagnostic"][arm]["by_question"].get("edge-on-bulge")
        if eob:
            print(f"  edge-on-bulge [{arm}] acc={eob['accuracy']:.3f} "
                  f"abstain={eob['abstention_rate']:.3f} n={eob['answers']}")
    print(f"\nWrote {out_dir/'comparison.json'}")


if __name__ == "__main__":
    main()
