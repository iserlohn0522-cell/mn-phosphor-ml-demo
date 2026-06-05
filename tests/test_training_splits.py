from __future__ import annotations

from phosphor_ml.training.splits import make_cv_splits, split_policy


def test_grouped_splits_keep_same_group_out_of_validation_fold():
    groups = ["doi-a", "doi-a", "doi-b", "doi-b", "doi-c", "doi-c"]

    splits = make_cv_splits(n_samples=6, k_folds=3, random_seed=42, groups=groups)

    assert len(splits) == 3
    assert split_policy(n_samples=6, k_folds=3, groups=groups) == "group_kfold"
    for train_idx, test_idx in splits:
        train_groups = {groups[index] for index in train_idx}
        test_groups = {groups[index] for index in test_idx}
        assert train_groups.isdisjoint(test_groups)


def test_splits_fall_back_to_kfold_when_groups_are_too_sparse():
    groups = ["doi-a", "doi-a", "", "", "", ""]

    splits = make_cv_splits(n_samples=6, k_folds=3, random_seed=42, groups=groups)

    assert len(splits) == 3
    assert split_policy(n_samples=6, k_folds=3, groups=groups) == "kfold"
