"""
Diagnostic script: vérifie l'hypothèse de biais dans le sous-échantillonnage
de IMDB et amazon_polarity (et tout autre dataset DLK > 20k lignes).

Pour chaque taille N, on reproduit EXACTEMENT le chemin emprunté par
`get_dlk_dataset` (tri lex de str(idx) puis select), puis on calcule:
  - distribution des labels source (label=0 vs label=1)
  - longueur moyenne / max des reviews (caractères + tokens Llama-2)

Usage:
  python diagnose_imdb_bias.py
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

DATASETS_TO_CHECK = ["imdb", "amazon_polarity"]
TEXT_FIELD = {"imdb": "text", "amazon_polarity": "content"}
SIZES = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]


def lex_sorted_indices(n: int) -> list[int]:
    """Reproduit `sorted(range(n), key=lambda i: str(i))` (chemin buggé actuel)."""
    return sorted(range(n), key=lambda i: str(i))


def hash_sorted_indices(n: int) -> list[int]:
    """Reproduit le vrai `deterministic_shuffle` (sha256 de str(i))."""
    def k(i: int) -> int:
        return int(hashlib.sha256(str(i).encode("utf-8")).hexdigest(), 16)
    return sorted(range(n), key=k)


def try_load_tokenizer():
    try:
        from transformers import AutoTokenizer
        # Tokenizer Llama-2 (publique, pas de gated weights nécessaires)
        return AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    except Exception as e:
        print(f"[warn] Tokenizer Llama indisponible ({e}); on ne mesurera que les longueurs en caractères.")
        return None


def main():
    tokenizer = try_load_tokenizer()

    for ds_name in DATASETS_TO_CHECK:
        text_field = TEXT_FIELD[ds_name]
        print("\n" + "=" * 78)
        print(f"DATASET: {ds_name}  (champ texte = {text_field!r})")
        print("=" * 78)

        ds = load_dataset(ds_name)
        for split_name in ("train", "test"):
            if split_name not in ds:
                continue
            split = ds[split_name]
            n = len(split)
            print(f"\n--- split={split_name}  total={n} ---")

            labels_full = np.array(split["label"])
            print(f"distribution globale: label=0 -> {(labels_full == 0).sum()},  "
                  f"label=1 -> {(labels_full == 1).sum()}")

            # Trigge le chemin "optimisé" buggé uniquement si > 20000
            uses_buggy_path = n > 20000
            print(f"len > 20000 ? {uses_buggy_path}  "
                  f"=> {'CHEMIN BUGGÉ (tri lex)' if uses_buggy_path else 'chemin OK (sha256)'}")

            lex_idx = lex_sorted_indices(n)
            hash_idx = hash_sorted_indices(n)

            print(f"\n{'N':>7} | "
                  f"{'lex #l0':>7} {'lex #l1':>7} {'lex %pos':>9} | "
                  f"{'sha #l0':>7} {'sha #l1':>7} {'sha %pos':>9}")
            print("-" * 78)
            for N in SIZES:
                if N > n:
                    continue
                top_lex = lex_idx[:N]
                top_hash = hash_idx[:N]
                lab_lex = labels_full[top_lex]
                lab_hash = labels_full[top_hash]
                print(f"{N:>7} | "
                      f"{(lab_lex == 0).sum():>7} {(lab_lex == 1).sum():>7} "
                      f"{lab_lex.mean():>9.3f} | "
                      f"{(lab_hash == 0).sum():>7} {(lab_hash == 1).sum():>7} "
                      f"{lab_hash.mean():>9.3f}")

            # Longueurs (sur le subset N=2000 lex pour rester rapide)
            sample_size = min(2000, n)
            sample_idx = lex_idx[:sample_size]
            char_lens = []
            tok_lens = []
            texts_sample = split.select(sample_idx)[text_field]
            for t in texts_sample:
                char_lens.append(len(t))
            char_lens = np.array(char_lens)

            if tokenizer is not None:
                # batch tokenize sans padding/troncature
                enc = tokenizer(list(texts_sample), add_special_tokens=False)
                tok_lens = np.array([len(ids) for ids in enc["input_ids"]])

            print(f"\nLongueurs sur top-{sample_size} (chemin lex actuel):")
            print(f"  caractères: mean={char_lens.mean():.0f}  "
                  f"median={np.median(char_lens):.0f}  "
                  f"p95={np.percentile(char_lens, 95):.0f}  "
                  f"max={char_lens.max()}")
            if len(tok_lens):
                pct_over_512 = (tok_lens > 512).mean() * 100
                pct_over_1024 = (tok_lens > 1024).mean() * 100
                pct_over_2048 = (tok_lens > 2048).mean() * 100
                print(f"  tokens Llama-2: mean={tok_lens.mean():.0f}  "
                      f"median={np.median(tok_lens):.0f}  "
                      f"p95={np.percentile(tok_lens, 95):.0f}  "
                      f"max={tok_lens.max()}")
                print(f"  fraction reviews > 512 / 1024 / 2048 tokens: "
                      f"{pct_over_512:.1f}% / {pct_over_1024:.1f}% / {pct_over_2048:.1f}%")


if __name__ == "__main__":
    main()
