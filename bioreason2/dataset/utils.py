from typing import Any, Dict
from functools import lru_cache


# =========== Configuration ===========
# GO aspect order and mapping
ASPECT_ORDER = ["MF", "CC", "BP"]
ASPECT_TO_COLUMN = {"MF": "go_mf", "BP": "go_bp", "CC": "go_cc"}
ASPECT_FULL_NAMES = {
    "MF": "Molecular Function",
    "BP": "Biological Process",
    "CC": "Cellular Component",
}
# ASPECT_SPECIAL_TOKENS = {
#     "MF": {"start": "<|MF_START|>", "end": "<|MF_END|>"},
#     "BP": {"start": "<|BP_START|>", "end": "<|BP_END|>"},
#     "CC": {"start": "<|CC_START|>", "end": "<|CC_END|>"},
# }
ASPECT_SPECIAL_TOKENS = {
    "MF": {"start": "Analysing Molecular Function:", "end": ""},
    "BP": {"start": "Analysing Biological Process:", "end": ""},
    "CC": {"start": "Analysing Cellular Component:", "end": ""},
}
ASPECT_SUMMARY_TOKENS = {
    "start": "<|GO_SUMMARY_START|>",
    "end": "<|GO_SUMMARY_END|>",
}
ASPECT_ROOT_TERMS = {
    "MF": "GO:0003674",  # molecular_function
    "BP": "GO:0008150",  # biological_process
    "CC": "GO:0005575",  # cellular_component
}
FUNCTION_SUMMARY_TOKENS = {
    "start": "<|FUNCTION_SUMMARY_START|>",
    "end": "<|FUNCTION_SUMMARY_END|>",
}
# INTERPRO_SUMMARY_TOKENS = {
#     "start": "<|INTERPRO_SUMMARY_START|>",
#     "end": "<|INTERPRO_SUMMARY_END|>",
# }
INTERPRO_SUMMARY_TOKENS = {
    "start": "InterPro terms:",
    "end": "",
}


def truncate_protein(example: Dict[str, Any], max_length: int = 2048) -> Dict[str, Any]:
    """Truncate a protein sequence to a maximum length"""
    example["sequence"] = example["sequence"][:max_length]
    return example


def filter_go_terms_to_leaf_terms(go_terms_list, go_dag):
    """Keep only leaf GO terms for a single protein: terms that are NOT ancestors
    of any other term in this protein's GO-term list (using is_a ∪ part_of only).
    Preserves input order; ignores IDs not present in go_dag.

    Assumes go_dag was built once from go-basic.obo with relationships loaded
    (e.g., get_godag(..., optional_attrs={'relationship'})).
    """
    if not go_terms_list:
        return []

    # ---------- In-order dedup and drop unknowns ----------
    seen = set()
    terms = []
    for tid in go_terms_list:
        if tid in go_dag and tid not in seen:
            seen.add(tid)
            terms.append(tid)
    if len(terms) <= 1:
        return terms
    term_set = set(terms)

    # ---------- One-time lazy indexes on the DAG (persist across calls) ----------
    # Immediate parents via is_a ∪ part_of
    if not hasattr(go_dag, "_isapartof_parents"):
        go_dag._isapartof_parents = {}
    # Transitive ancestor cache via is_a ∪ part_of
    if not hasattr(go_dag, "_isapartof_ancestors"):
        go_dag._isapartof_ancestors = {}

    parents_index = go_dag._isapartof_parents
    anc_cache = go_dag._isapartof_ancestors

    def _compute_immediate_parents(go_id):
        """Collect immediate parents under is_a ∪ part_of for a single node."""
        node = go_dag.get(go_id)
        if node is None:
            return ()
        out = set()

        # is_a parents (GOATOOLS typically stores these in .parents)
        ps = getattr(node, "parents", ())
        if ps:
            for p in ps:
                pid = getattr(p, "id", p)
                if pid:
                    out.add(pid)

        # part_of via relationship maps (name varies by version)
        relmap = getattr(node, "relationships", None) or getattr(
            node, "relationship", None
        )
        if relmap:
            for p in relmap.get("part_of", ()):
                pid = getattr(p, "id", p)
                if pid:
                    out.add(pid)

        # sometimes part_of exists as a direct attribute
        if hasattr(node, "part_of"):
            for p in getattr(node, "part_of"):
                pid = getattr(p, "id", p)
                if pid:
                    out.add(pid)

        return tuple(out)

    @lru_cache(maxsize=None)
    def _parents(go_id):
        """Immediate parents via is_a ∪ part_of with on-DAG memo."""
        par = parents_index.get(go_id)
        if par is None:
            par = _compute_immediate_parents(go_id)
            parents_index[go_id] = par
        return par

    def _ancestors(go_id):
        """Transitive parents via is_a ∪ part_of with on-DAG memo."""
        hit = anc_cache.get(go_id)
        if hit is not None:
            return hit
        seen_ids = set()
        stack = list(_parents(go_id))
        while stack:
            pid = stack.pop()
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            stack.extend(_parents(pid))  # keep traversing through intermediates
        res = frozenset(seen_ids)
        anc_cache[go_id] = res
        return res

    # ---------- Depth proxy (ancestor count) for pruning order ----------
    # Deep terms (more ancestors) come first → their ancestors get marked early.
    sorted_terms = sorted(terms, key=lambda t: len(_ancestors(t)), reverse=True)

    keep = set()
    removed_ancestors = (
        set()
    )  # ancestors of already-kept deeper terms (restricted to our set)

    # ---------- Prune ancestors quickly in a single pass ----------
    for t in sorted_terms:
        if t in removed_ancestors:
            continue
        keep.add(t)
        # Only mark ancestors that are in our list; avoids growing the set needlessly
        removed_ancestors.update(term_set.intersection(_ancestors(t)))

    # ---------- Return in original input order ----------
    return [t for t in terms if t in keep]


def format_go_terms_with_names(go_terms_list, go_dag):
    """Format GO terms as 'GO:XXXXXX term_name' pairs"""
    formatted_terms = []
    for go_id in go_terms_list:
        if go_id in go_dag:
            term_obj = go_dag[go_id]
            formatted_terms.append(f"{go_id} {term_obj.name}")
        else:
            formatted_terms.append(f"{go_id} (not found in GO DAG)")

    return formatted_terms
