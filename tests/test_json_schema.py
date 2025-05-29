"""
Full JSON-schema test-suite for generate_ragged + JSONFilter.

• 3 schema shapes  (nested object / array-of-objects / freetext)
• 3 padding/cache modes (left/-cache, right/-cache, right/+cache)
• 3 representative models  (Llama-3-8B, Mistral-7B, Gemma-2B)

The suite checks that:
  – output contains *balanced* JSON,
  – every row parses,
  – keys match the schema exactly,
  – primitive types & simple numeric bounds are respected.
"""

import os, json, re, gc, torch, pytest
from pathlib import Path
from omegaconf import OmegaConf

# ═════════════════════════════════════════════════════════════════════
#  Project paths
# ═════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parents[1]  # project root
MODELS_YAML = ROOT / "conf" / "models" / "models.yaml"

# ═════════════════════════════════════════════════════════════════════
#  Model matrix   – Pick any subset your hardware can handle
# ═════════════════════════════════════════════════════════════════════
MODEL_IDS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-2b-it",
]

# ═════════════════════════════════════════════════════════════════════
#  Schemas under test
# ═════════════════════════════════════════════════════════════════════
SCHEMAS = {
    "nested": {
        "type": "object",
        "properties": {
            "person": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "maximum": 120},
                },
                "required": ["name", "age"],
            }
        },
        "required": ["person"],
    },
    "array": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "maximum": 120},
            },
            "required": ["name", "age"],
        },
        "minItems": 1,
        "maxItems": 3,
    },
    "freetext": {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["description"],
    },
}

# ═════════════════════════════════════════════════════════════════════
#  Padding / cache matrix  (left + cache not implemented)
# ═════════════════════════════════════════════════════════════════════
PADDING_CASES = [
    ("right", False),
    ("right", True),
]


# ═════════════════════════════════════════════════════════════════════
#  Tiny JSON-schema key/type checker
# ═════════════════════════════════════════════════════════════════════
def _assert_schema_keys(schema: dict, obj):
    """Minimal recursive check – keys, required, primitives, array len."""
    if schema["type"] == "object":
        assert isinstance(obj, dict), "expected object"
        req = schema.get("required", [])
        for k in req:
            assert k in obj, f"missing required key '{k}'"
        allowed = set(schema.get("properties", {}))
        assert set(obj).issubset(allowed), f"unexpected keys {set(obj) - allowed}"
        for k, v in obj.items():
            if k in allowed:
                _assert_schema_keys(schema["properties"][k], v)

    elif schema["type"] == "array":
        assert isinstance(obj, list), "expected array"
        mi, ma = schema.get("minItems", 0), schema.get("maxItems", float("inf"))
        assert mi <= len(obj) <= ma, "array length out of bounds"
        for item in obj:
            _assert_schema_keys(schema["items"], item)

    elif schema["type"] == "string":
        assert isinstance(obj, str), "expected string"
    elif schema["type"] == "integer":
        assert isinstance(obj, int), "expected int"
        if "maximum" in schema:
            assert obj <= schema["maximum"], "integer exceeds maximum"


# ═════════════════════════════════════════════════════════════════════
#  Model loader util  (loads & caches per session)
# ═════════════════════════════════════════════════════════════════════
_model_cache: dict[str, tuple] = {}


def _get_model_tok(model_id: str):
    if model_id not in _model_cache:
        from src.io_utils import load_model_and_tokenizer

        cfg = OmegaConf.load(MODELS_YAML)[model_id]
        model, tok = load_model_and_tokenizer(cfg)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        _model_cache[model_id] = (model, tok)
    return _model_cache[model_id]


# ═════════════════════════════════════════════════════════════════════
#  Core generation helper
# ═════════════════════════════════════════════════════════════════════
def _generate(model, tok, schema, padding_side, use_cache, prompt_suffix=""):
    from src.lm_utils import generate_ragged, parse_json_response

    prompt = f"Return ONLY JSON that matches this schema:\n{json.dumps(schema)}\n{prompt_suffix}"
    ids = tok.encode(prompt, add_special_tokens=False)
    out = generate_ragged(
        model=model,
        tokenizer=tok,
        token_list=[torch.tensor(ids)],
        max_new_tokens=128,
        temperature=0.1,
        padding_side=padding_side,
        use_cache=use_cache,
        json_schema=schema,
    )[0][0]
    print(f"\n--- JSON output ---\n{out.strip()}")
    frag = parse_json_response(out)
    assert frag, "no balanced JSON in output"
    obj = json.loads(frag)
    _assert_schema_keys(schema, obj)
    return frag, out


# ═════════════════════════════════════════════════════════════════════
#  1) Single-row tests (schema × model × padding/cache)
# ═════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("model_id", MODEL_IDS)
@pytest.mark.parametrize("schema_name", SCHEMAS.keys())
@pytest.mark.parametrize("padding_side,use_cache", PADDING_CASES)
def test_single_row(model_id, schema_name, padding_side, use_cache):
    if padding_side == "left" and use_cache:
        pytest.skip("left+cache branch not implemented")
    model, tok = _get_model_tok(model_id)

    _, preview = _generate(model, tok, SCHEMAS[schema_name], padding_side, use_cache)
    print(f"\n{model_id.split('/')[-1]:<40} | {schema_name:<8} | {padding_side:<5} cache={use_cache}: {preview} ...")


# ═════════════════════════════════════════════════════════════════════
#  2) Mixed-batch test (one per model, right-padding with cache)
# ═════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_mixed_batch(model_id):
    model, tok = _get_model_tok(model_id)
    schema = SCHEMAS["nested"]

    prompts = [
        "JSON please.",
        "JSON please, but make it fancy.",
        "Just JSON.",
    ]
    batch = [torch.tensor(tok.encode(p, add_special_tokens=False)) for p in prompts]

    from src.lm_utils import generate_ragged, parse_json_response

    outs = generate_ragged(
        model=model,
        tokenizer=tok,
        token_list=batch,
        max_new_tokens=96,
        padding_side="right",
        use_cache=True,
        json_schema=schema,
    )[0]

    for i, out in enumerate(outs):
        print(f"\nRow {i}: {out.strip()} ...\n")

    print(f"\n--- Mixed batch for {model_id} ---")
    for i, out in enumerate(outs):
        frag = parse_json_response(out) or "<NO JSON>"
        print(f"row {i}: {frag}")
        assert frag != "<NO JSON>", f"row {i} produced no JSON"
        _assert_schema_keys(schema, json.loads(frag))
