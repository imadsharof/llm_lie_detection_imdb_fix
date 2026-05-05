from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from White_Box_Lie_Detection.repeng.datasets.elk.types import BinaryRow, DlkDatasetId, Split
from White_Box_Lie_Detection.repeng.datasets.utils.shuffles import (
    deterministic_shuffle,
    deterministic_shuffle_sort_fn,
)
from White_Box_Lie_Detection.repeng.datasets.utils.splits import split_train

# Graine utilisée pour le sous-échantillonnage déterministe des datasets DLK
# de plus de 20 000 lignes. Préfixée à `str(i)` avant le hash sha256, ce qui
# rend l'ordre reproductible ET facile à varier (utile pour une éventuelle
# validation multi-seed). Avant le fix, le chemin "optimisé" triait `str(i)`
# en clair (ordre lexicographique pur), ce qui produisait des sous-échantillons
# presque mono-classe sur IMDB / amazon_polarity (rows triées par classe à la
# source) et écrasait l'AUC du probe pour les petits N.
DLK_SHUFFLE_SEED = 42


@dataclass
class _DatasetSpec:
    name: str
    subset: str | None = None
    validation_name: str = "validation"


@dataclass
class _DlkTemplate:
    template: str
    labels: list[str]
    args: list[str]
    insert_label_options: bool = True


_DATASET_SPECS: dict[DlkDatasetId, _DatasetSpec] = {
    # Sentiment classification
    "imdb": _DatasetSpec("imdb", validation_name="test"),
    "amazon_polarity": _DatasetSpec("amazon_polarity", validation_name="test"),
    # Topic classification
    "ag_news": _DatasetSpec("ag_news", validation_name="test"),
    "dbpedia_14": _DatasetSpec("dbpedia_14", validation_name="test"),
    # NLI
    "rte": _DatasetSpec("super_glue", "rte"),
    # N.B.: We skip QNLI because we can't find the prompt templates.
    # Story completion
    "copa": _DatasetSpec("super_glue", "copa"),
    # N.B.: We skip story_cloze because it requires filling in a form to access.
    # Question answering
    "boolq": _DatasetSpec("super_glue", "boolq"),
    # Common sense reasoning
    "piqa": _DatasetSpec("piqa"),
    # Other formats:
    "boolq/simple": _DatasetSpec("super_glue", "boolq"),
    "imdb/simple": _DatasetSpec("imdb", validation_name="test"),
}


def get_dlk_dataset(dataset_id: DlkDatasetId, limit: int = 60000):
    dataset_spec = _DATASET_SPECS[dataset_id]
    try:
        # Standard datasets now complain if trust_remote_code=True is passed unnecessarily
        dataset: Any = load_dataset(dataset_spec.name, dataset_spec.subset)
    except Exception as e:
        # fallback for script-based datasets
        dataset: Any = load_dataset(dataset_spec.name, dataset_spec.subset, trust_remote_code=True)
    
    return {
        **_get_dlk_dataset(dataset_id, dataset, split="train", limit=limit),
        **_get_dlk_dataset(dataset_id, dataset, split="validation", limit=limit),
    }


def _get_dlk_dataset(
    dataset_id: DlkDatasetId,
    dataset: Any,
    split: Split,
    limit: int,
) -> dict[str, BinaryRow]:
    dataset_spec = _DATASET_SPECS[dataset_id]
    if split == "train":
        hf_split = "train"
    elif split == "validation":
        hf_split = dataset_spec.validation_name
    else:
        raise ValueError(split)

    results = {}
    # dataset = dataset.shuffle(seed=42) # Skipping shuffle to avoid processing time on massive datasets
    
    ds_split = dataset[hf_split]
    
    items_to_process = []

    # Optimization for large datasets: replicate `deterministic_shuffle` (sha256 of
    # str(index)) on indices alone, without materialising the full dataset content.
    # Bug fix: the previous version sorted by `str(i)` directly (plain lexicographic
    # order), which is NOT a shuffle — it produced a heavily biased ordering. On
    # IMDB/amazon_polarity (rows pre-sorted by class on HF), this caused the first
    # N picked rows to be drastically class-imbalanced for small N (e.g. IMDB
    # N <= 2000 → 100% label=0), poisoning probe training with a spurious
    # template-token shortcut.
    if len(ds_split) > 20000:
        all_indices = range(len(ds_split))
        sorted_indices = sorted(
            all_indices,
            key=lambda i: deterministic_shuffle_sort_fn(
                f"{DLK_SHUFFLE_SEED}-{i}", None
            ),
        )
        target_indices = sorted_indices[:limit]
        ds_trimmed = ds_split.select(target_indices)
        # Preserve the ORIGINAL row index so that downstream `group_id`,
        # `split_train(row_id=...)` and false-label rotation stay consistent.
        items_to_process = zip(target_indices, ds_trimmed)
    else:
        iterable = enumerate(ds_split)
        items_to_process = deterministic_shuffle(
            iterable, lambda row: str(row[0])
        )[:limit]

    for row_idx, row in items_to_process:
        assert "label" in row and type(row["label"]) == int, row
        template = _TEMPLATES[dataset_id]
        true_label: int = row["label"]
        false_label_options = list(range(len(template.labels)))
        false_label_options.remove(true_label)
        
        # For the script check, let's yield ALL false_labels instead of just one
        # so that when the user inspects one question, they see all possible options.
        labels = [true_label] + false_label_options

        for label_idx, label in enumerate(labels):
            format_args: dict[str, str] = dict(label=template.labels[label])
            if template.insert_label_options:
                format_args["label1"] = template.labels[labels[0]]
                format_args["label2"] = template.labels[labels[1]] if len(labels) > 1 else ""
            for arg in template.args:
                format_args[arg] = row[arg]
            prompt = template.template.format(**format_args)
            results[f"{dataset_id}-{row_idx}-{label_idx}-{split}"] = BinaryRow(
                dataset_id=dataset_id,
                split=split_train(split, seed=dataset_id, row_id=str(row_idx)),
                group_id=str(row_idx),
                text=prompt,
                label=label == true_label,
                format_args=format_args,
                # As labels are sorted, true/false questions will be consistent.
                answer_type=str(label_idx),
            )
    return results


# Taken from <https://arxiv.org/pdf/2212.03827.pdf> appendix 1.
_TEMPLATES: dict[DlkDatasetId, _DlkTemplate] = {
    "ag_news": _DlkTemplate(
        'Consider the following example: "{text}"\n'
        "Choice 1: {label1}\n"
        "Choice 2: {label2}\n"
        "Between choice 1 and choice 2, the topic of this example is {label}",
        labels=["World", "Sports", "Business", "Sci/Tech"],
        args=["text"],
    ),
    "amazon_polarity": _DlkTemplate(
        'Consider the following example: "{content}"\n'
        "Choice 1: {label1}\n"
        "Choice 2: {label2}\n"
        "Between choice 1 and choice 2, the sentiment of this example is {label}",
        labels=["Negative", "Positive"],
        args=["content"],
    ),
    "boolq": _DlkTemplate(
        "Passage: {passage}\n"
        "After reading this passage, I have a question: {question}? True or False?\n"
        "{label}",
        labels=["False", "True"],
        args=["passage", "question"],
        insert_label_options=False,
    ),
    "boolq/simple": _DlkTemplate(
        "{passage}\nQuestion: {question}?\nAnswer: {label}",
        labels=["no", "yes"],
        args=["passage", "question"],
        insert_label_options=False,
    ),
    "copa": _DlkTemplate(
        'Consider the following premise: "{premise}"\n'
        "Choice 1: {choice1}\n"
        "Choice 2: {choice2}\n"
        "Q: Which one is more likely to be the {question}, choice 1 or choice 2?\n"
        "{label}",
        labels=["Choice 1", "Choice 2"],
        args=["premise", "question", "choice1", "choice2"],
        insert_label_options=False,
    ),
    "dbpedia_14": _DlkTemplate(
        'Consider the following example: "{content}"\n'
        "Choice 1: {label1}\n"
        "Choice 2: {label2}\n"
        "Between choice 1 and choice 2, the topic of this example is {label}",
        labels=[
            "Company",
            "EducationalInstitution",
            "Artist",
            "Athlete",
            "OfficeHolder",
            "MeanOfTransportation",
            "Building",
            "NaturalPlace",
            "Village",
            "Animal",
            "Plant",
            "Album",
            "Film",
            "WrittenWork",
        ],
        args=["content"],
    ),
    "imdb": _DlkTemplate(
        'Consider the following example: "{text}"\n'
        "Between {label1} and {label2}, the sentiment of this example is {label}",
        labels=["Negative", "Positive"],
        args=["text"],
    ),
    "imdb/simple": _DlkTemplate(
        "{text}\nQuestion: What is the sentiment of the review?\nAnswer: {label}",
        labels=["negative", "positive"],
        args=["text"],
        insert_label_options=False,
    ),
    "piqa": _DlkTemplate(
        "Goal: {goal}\n"
        "Which is the correct ending?\n"
        "Choice 1: {sol1}\n"
        "Choice 2: {sol2}\n"
        "{label}",
        labels=["Choice 1", "Choice 2"],
        args=["goal", "sol1", "sol2"],
        insert_label_options=False,
    ),
    "rte": _DlkTemplate(
        "{premise}\n"
        'Question: Does this imply that "{hypothesis}", yes or no?\n'
        "{label}",
        labels=["yes", "no"],
        args=["premise", "hypothesis"],
        insert_label_options=False,
    ),
}
