from __future__ import annotations

from collections.abc import Sequence


def make_cv_splits(
    *,
    n_samples: int,
    k_folds: int = 5,
    random_seed: int = 42,
    groups: Sequence[str] | None = None,
) -> list[tuple[list[int], list[int]]]:
    if n_samples < 2:
        raise ValueError("at least two samples are required for cross-validation")

    folds = min(k_folds, n_samples)
    if split_policy(n_samples=n_samples, k_folds=folds, groups=groups) == "group_kfold":
        from sklearn.model_selection import GroupKFold

        splitter = GroupKFold(n_splits=folds)
        return [
            (train_indices.tolist(), test_indices.tolist())
            for train_indices, test_indices in splitter.split(range(n_samples), groups=groups)
        ]

    from sklearn.model_selection import KFold

    splitter = KFold(n_splits=folds, shuffle=True, random_state=random_seed)
    return [
        (train_indices.tolist(), test_indices.tolist())
        for train_indices, test_indices in splitter.split(range(n_samples))
    ]


def split_policy(
    *,
    n_samples: int,
    k_folds: int,
    groups: Sequence[str] | None = None,
) -> str:
    if groups is None:
        return "kfold"
    usable_groups = {str(group) for group in groups if str(group).strip()}
    if len(groups) == n_samples and len(usable_groups) >= min(k_folds, n_samples):
        return "group_kfold"
    return "kfold"
