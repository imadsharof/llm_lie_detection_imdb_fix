import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
repeng_path = project_root / "White_Box_Lie_Detection"
if str(project_root) not in sys.path: sys.path.append(str(project_root))
if str(repeng_path) not in sys.path: sys.path.append(str(repeng_path))

from White_Box_Lie_Detection.repeng.datasets.elk import (
    dlk, race, arc, open_book_qa, common_sense_qa
)

def load_dataset_efficiency(dataset_name="commonsense_qa", split='train', random_seed=42):
    """
    Loads datast for the Efficiency phase, maintaining a strictly 1:1 balance.
    It returns ALL possible balanced pairs. The notebook will then slice 
    this into 5k (train) or 1k (test).
    """
    data_dict = {}
    name = dataset_name.lower().strip()
    if name == "csqa": name = "commonsense_qa"
    if name == "openbookqa": name = "open_book_qa"

    try:
        if name == "commonsense_qa": data_dict = common_sense_qa.get_common_sense_qa("repe")
        elif name == "race": data_dict = race.get_race("repe")
        elif name == "arc_easy": data_dict = arc.get_arc("easy", "repe")
        elif name == "arc_challenge": data_dict = arc.get_arc("challenge", "repe")
        elif name == "open_book_qa": data_dict = open_book_qa.get_open_book_qa("repe")
        elif name in ["imdb", "amazon_polarity", "ag_news", "dbpedia_14", "rte", "boolq"]:
            data_dict = dlk.get_dlk_dataset(name)
        else: return pd.DataFrame()
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return pd.DataFrame()

    processed_data = []
    
    # Process dictionary to extract data list
    for key, row in data_dict.items():
        if split and split != 'all' and row.split != split: continue
        item = {
            "id": row.group_id if row.group_id else key,
            "text": row.text,
            "label": 1 if row.label else 0,
            "dataset": row.dataset_id, "split": row.split
        }
        processed_data.append(item)
    
    df = pd.DataFrame(processed_data)
    if df.empty: return df

    # Enforce strictly 1 True and 1 False per question
    np.random.seed(random_seed)

    # Bug fix: pandas `groupby('id')` defaults to `sort=True`, which re-orders
    # groups alphabetically by their string id. Since `id = str(row_idx)` and
    # IMDB rows are pre-sorted by class on HF (rows 0..12499 = neg, 12500..
    # 24999 = pos), the lexicographic order puts all class-0 row_idx values
    # first ("0", "1", "10", "100", "1000", "10000".."12499", "13",...). The
    # downstream notebook does `df_raw.iloc[:N]` to build train/val splits, so
    # for any small N both train and val end up almost mono-class, killing the
    # probe AUC (~0.60 plateau). To fix this we shuffle `df` rows with a
    # seeded RNG before groupby and tell groupby NOT to re-sort, so that
    # `iloc[:N]` slices a representative class-balanced sample. Combined with
    # the sha256 shuffle in dlk.py (DLK_SHUFFLE_SEED), the source-class
    # distribution of the first N rows of the returned df is now ~50/50.
    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(len(df))
    df = df.iloc[perm].reset_index(drop=True)

    grouped = df.groupby('id', sort=False)
    balanced_samples = []

    for group_id, group_df in grouped:
        true_samples = group_df[group_df['label'] == 1]
        false_samples = group_df[group_df['label'] == 0]

        if len(true_samples) > 0 and len(false_samples) > 0:
            chosen_true = true_samples.sample(n=1, random_state=random_seed)
            chosen_false = false_samples.sample(n=1, random_state=random_seed)
            balanced_samples.append(chosen_true)
            balanced_samples.append(chosen_false)

    if len(balanced_samples) == 0:
        return pd.DataFrame()

    df_balanced = pd.concat(balanced_samples).reset_index(drop=True)
    return df_balanced

import gc
import torch
from tqdm.auto import tqdm
from Black_vs_White_Lie_Detection.Aligned_Comparison_BB_WB.wb_activations import get_activations_for_dataset


def robust_get_activations(df, model, tokenizer, device, initial_batch_size, desc="Extracting", target_size=None):
    results = []
    idx = 0
    total_to_extract = target_size if target_size is not None else len(df)
    pbar = tqdm(total=total_to_extract, desc=desc)
    current_bs = initial_batch_size

    # We iterate until we've processed the full df OR reached exactly our target limit
    while idx < len(df) and len(results) < total_to_extract:
        # Dynamically adjust the chunk size so we never overshoot our specific limit
        remaining = total_to_extract - len(results)
        bs_to_use = min(current_bs, remaining, len(df) - idx)

        chunk = df.iloc[idx:idx + bs_to_use]
        try:
            # We call the existing extraction function on this chunk
            chunk_results = get_activations_for_dataset(
                chunk, model, tokenizer, device=device, batch_size=bs_to_use, show_progress=False
            )
            results.extend(chunk_results)
            idx += bs_to_use
            pbar.update(bs_to_use)  # Exactly bs_to_use samples succeeded!

            # Flush completely after each successful chunk
            gc.collect()
            torch.cuda.empty_cache()

            # --- NEW: Speed recovery! ---
            # If it succeeds, immediately recover back to initial speed for the NEXT chunk
            current_bs = initial_batch_size

        except RuntimeError as e:
            # Check for Out Of Memory Error
            is_oom = "out of memory" in str(e).lower() or "outofmemoryerror" in str(getattr(e, "__class__", "")).lower()

            if is_oom:
                # Crucial: Clear traceback frames to release GPU memory tied to local variables (inputs, outputs)
                import traceback
                traceback.clear_frames(e.__traceback__)
                del e

                # Double collect to ensure frames are entirely wiped from memory
                gc.collect()
                torch.cuda.empty_cache()

                if bs_to_use > 1:
                    current_bs = max(1, current_bs // 2)
                    # Loop retries without advancing idx!
                else:
                    print(f"⚠️ Fatal OOM at batch size 1 at index {idx}. Retrying with left-truncation...")
                    success = False
                    # Drop the tokens aggressively from the left to save memory and preserve answer probe
                    for max_len in [2048, 1024, 768]:
                        try:
                            chunk_results = get_activations_for_dataset(
                                chunk, model, tokenizer, device=device, batch_size=1, show_progress=False, max_length=max_len
                            )
                            results.extend(chunk_results)
                            idx += 1
                            pbar.update(1)
                            success = True
                            print(f"✅ Survived by truncating to {max_len} tokens form the left!")
                            break
                        except RuntimeError as e2:
                            is_oom2 = "out of memory" in str(e2).lower() or "outofmemoryerror" in str(getattr(e2, "__class__", "")).lower()
                            if is_oom2:
                                import traceback
                                traceback.clear_frames(e2.__traceback__)
                                del e2
                                gc.collect()
                                torch.cuda.empty_cache()
                            else:
                                raise e2

                    if not success:
                        print(f"❌ Failed even with max_length=768. Skipping this single sample as a last resort.")
                        idx += 1

                    # Recover batch size after resolving the fatal sample
                    current_bs = initial_batch_size
            else:
                raise e

    pbar.close()
    return results

