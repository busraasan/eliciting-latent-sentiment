"""
Generate and save clean/corrupted sentence pairs for the toy movie dataset.

Replicates the pairing logic from get_dataset() in utils/prompts.py without
requiring a loaded HookedTransformer model (token-length filtering is skipped).

Output: data/toy_movie/<split>.csv  with columns:
  clean, corrupted, clean_sentiment, corrupted_sentiment
"""

import csv
import os
import yaml

# ── Config ────────────────────────────────────────────────────────────────────

TEMPLATE = "I thought this movie was{ADJ}, I{VRB} it. \nConclusion: This movie is"

SPLITS = {
    "simple": {
        "pos_adj": "positive_core_adjectives",
        "neg_adj": "negative_core_adjectives",
    },
    "simple_train": {
        "pos_adj": "positive_adjectives_train",
        "neg_adj": "negative_adjectives_train",
    },
    "simple_test": {
        "pos_adj": "positive_adjectives_test",
        "neg_adj": "negative_adjectives_test",
    },
}

OUTPUT_DIR = os.path.join("data", "toy_movie")

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepend_space(words: list) -> list:
    """Match PromptsConfig.get() behaviour: prepend a space to each word."""
    return [" " + w.strip() for w in words]


def dedup(lst: list) -> list:
    seen = set()
    return [x for x in lst if x not in seen and not seen.add(x)]


def build_prompts(adjectives: list, verbs: list) -> list:
    """
    Pair adjectives with verbs cyclically (CircularList behaviour) and
    produce one prompt per adjective.
    """
    n = len(adjectives)
    return [
        TEMPLATE.format(ADJ=adjectives[i], VRB=verbs[i % len(verbs)])
        for i in range(n)
    ]


def make_pairs(pos_prompts: list, neg_prompts: list) -> list:
    """
    Replicate the interleaving + shift-by-one logic from get_dataset():

      all_prompts = [pos[0], neg[0], pos[1], neg[1], ...]
      clean       = all_prompts
      corrupted   = all_prompts[1:] + [all_prompts[0]]

    Returns a list of dicts with keys:
      clean, corrupted, clean_sentiment, corrupted_sentiment
    """
    n = min(len(pos_prompts), len(neg_prompts))
    all_prompts = []
    sentiments = []
    for i in range(n):
        all_prompts.append(pos_prompts[i])
        sentiments.append("positive")
        all_prompts.append(neg_prompts[i])
        sentiments.append("negative")

    corrupted = all_prompts[1:] + [all_prompts[0]]
    corrupted_sent = sentiments[1:] + [sentiments[0]]

    return [
        {
            "clean": all_prompts[i],
            "corrupted": corrupted[i],
            "clean_sentiment": sentiments[i],
            "corrupted_sentiment": corrupted_sent[i],
        }
        for i in range(len(all_prompts))
    ]


def save_csv(rows: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["clean", "corrupted", "clean_sentiment", "corrupted_sentiment"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows):>3} pairs → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_yaml("prompts.yaml")

    pos_verbs = dedup(prepend_space(cfg["positive_verbs"]))
    neg_verbs = dedup(prepend_space(cfg["negative_verbs"]))

    for split_name, keys in SPLITS.items():
        pos_adj_raw = cfg.get(keys["pos_adj"])
        neg_adj_raw = cfg.get(keys["neg_adj"])

        if pos_adj_raw is None or neg_adj_raw is None:
            print(f"Skipping '{split_name}': missing key(s) in prompts.yaml")
            continue

        pos_adj = dedup(prepend_space(pos_adj_raw))
        neg_adj = dedup(prepend_space(neg_adj_raw))

        pos_prompts = build_prompts(pos_adj, pos_verbs)
        neg_prompts = build_prompts(neg_adj, neg_verbs)

        pairs = make_pairs(pos_prompts, neg_prompts)
        save_csv(pairs, os.path.join(OUTPUT_DIR, f"{split_name}.csv"))


if __name__ == "__main__":
    main()
