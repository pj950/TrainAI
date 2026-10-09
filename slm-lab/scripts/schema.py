"""Intent Router label schema shared by all scripts.

All scripts (data generation, training, evaluation, inference) import the
definitions here so training data, gold labels and metrics stay consistent.
"""

# System prompt used in every conversational sample.
SYSTEM_PROMPT = (
    "你是工业设备Intent Router，只能输出规定JSON，不要输出任何多余文字。"
    "字段固定为：intent, domain, need_rag, need_graph, missing_parameters。"
)

# Allowed values.
INTENTS = ["knowledge_query", "diagnosis", "action", "out_of_scope"]
DOMAINS = ["equipment", "maintenance", "process", "safety", "unknown"]

# Keys that a valid prediction must contain, in canonical order.
REQUIRED_KEYS = ["intent", "domain", "need_rag", "need_graph", "missing_parameters"]


def make_label(intent, domain, need_rag, need_graph, missing=None):
    """Build a label dict in canonical key order."""
    return {
        "intent": intent,
        "domain": domain,
        "need_rag": need_rag,
        "need_graph": need_graph,
        "missing_parameters": missing or [],
    }
