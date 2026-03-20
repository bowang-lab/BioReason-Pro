from typing import Any, Dict
from pprint import pprint


def format_cafa5_for_protein_llm(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a CAFA5 example into the required chat format for Protein-LLM.
    """
    return {
        "prompt": [
            # {
            #     "role": "system",
            #     "content": example["prompt"]["system"].strip(),
            # },
            {
                "role": "user",
                "content": [
                    {"type": "protein", "text": None},
                    {"type": "go_graph", "text": None},
                    {
                        "type": "text",
                        "text": f"{example['prompt']['system'].strip()}\n\n{example['prompt']['user'].strip()}",
                    },
                ],
            },
            {
                "role": "assistant",
                "reasoning_content": f"{example['prompt']['assistant_reasoning'].strip()}",
                "content": [
                    {
                        "type": "text",
                        "text": f"{example['prompt']['assistant_answer'].strip()}",
                    },
                ],
            },
        ],
        "protein_sequences": [
            example["sequence"],
        ],
        "structure_path": example.get("structure_path"),
        "go_aspect": example.get("go_aspect"),
        "answer": example["prompt"]["assistant_answer"].strip(),
        "ground_truth_go_terms": example.get("ground_truth_go_terms", ""),
    }


if __name__ == "__main__":
    from datasets import load_dataset

    ds = load_dataset("wanglab/cafa5", name="cafa5_reasoning", cache_dir="cafa5_reasoning_cache")
    train_df = ds["train"].to_pandas()

    # Get a specific protein
    prot_id = "A0A078CGE6"
    protein_data = train_df[train_df["protein_id"] == prot_id].iloc[0]

    # Format the training example
    formatted_example = format_cafa5_for_protein_llm(protein_data)
    pprint(formatted_example)
