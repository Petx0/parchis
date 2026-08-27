"""
Value calibration (docs/AGENT_REBUILD_PLAN.md Part 3 item 12 / Part 5.5):
bucket a net's predicted win probability into deciles and compare against
the actual frequency of winning, on held-out games -- the gate before any
self-play begins, since a miscalibrated value makes expectimax actively
harmful (a confidently-wrong leaf value poisons every node above it in the
max^n/chance-averaging recursion).
"""

import numpy as np

from parchis.az.net import NumpyAZNet, value_probs
from parchis.az.selfplay import examples_to_arrays


def _bucket_calibration(predicted, actual, n_buckets=10):
    """Pure bucketing/ECE computation over two same-length 1-D arrays:
    `predicted` (probabilities in [0, 1]) and `actual` (0/1 outcomes, or a
    fractional credit such as 1/num_players for a drawn/truncated game --
    calibration under a soft label is still well-defined: "was this seat's
    eventual credit close to what was predicted", the same generalization
    cross-entropy itself makes for soft targets).

    Returns:
        tuple(float, list[dict]): (ece, bucket_table).
        ece: sum over non-empty buckets of (bucket_size / n) *
            |mean_predicted - actual_frequency| -- the standard expected
            calibration error.
        bucket_table: [{'bucket', 'n', 'mean_predicted', 'actual_frequency'}, ...],
            one entry per non-empty bucket, in bucket order.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if predicted.shape != actual.shape:
        raise ValueError(f"shape mismatch: predicted={predicted.shape} actual={actual.shape}")
    if predicted.size == 0:
        raise ValueError("_bucket_calibration requires at least one example")

    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    bucket_idx = np.clip(np.digitize(predicted, edges[1:-1]), 0, n_buckets - 1)

    n = predicted.size
    bucket_table = []
    ece = 0.0
    for b in range(n_buckets):
        mask = bucket_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_predicted = float(predicted[mask].mean())
        actual_frequency = float(actual[mask].mean())
        bucket_table.append({
            'bucket': b, 'n': count,
            'mean_predicted': mean_predicted, 'actual_frequency': actual_frequency,
        })
        ece += (count / n) * abs(mean_predicted - actual_frequency)

    return ece, bucket_table


def value_calibration(model, test_examples, num_players, n_buckets=10):
    """
    Dict-based convenience wrapper around value_calibration_arrays, for
    parchis.az.selfplay.generate_games' list-of-dicts output directly. For
    an already-packed, disk-scale dataset (e.g. loaded from shards --
    parchis.az.train.split_shards / _load_and_concat_shards), call
    value_calibration_arrays directly instead of reconstructing one
    Python dict per decision just to unpack it again.
    """
    if not test_examples:
        raise ValueError("value_calibration requires at least one held-out example")
    X, _policy_targets, value_targets = examples_to_arrays(test_examples, num_players)
    return value_calibration_arrays(model, X, value_targets, n_buckets=n_buckets)


def value_calibration_arrays(model, X, value_targets, n_buckets=10):
    """
    Array-native core: for every held-out decision, the net's predicted
    P(that decision's own mover wins) against whether that seat actually
    went on to win (or the fractional 1/num_players credit for a
    truncated game).

    X: (n, input_size) float32 encodings. value_targets: (n, num_players)
    float32 -- both the encoding (input) and value_targets (target) must
    be built in the SAME mover-relative channel order
    (parchis.az.selfplay.generate_games' convention: index 0 is always
    "the deciding seat", matching parchis.az.encoding's own channel
    order) -- so both the net's predicted channel 0 and the target's
    channel 0 already refer to the same seat, with no remapping needed
    here (contrast parchis.az.agent.NetEvaluator, which DOES remap
    relative-to-absolute, because search.py's own contract is
    absolute-seat-indexed).
    """
    if X.shape[0] == 0:
        raise ValueError("value_calibration_arrays requires at least one example")

    numpy_model = NumpyAZNet.from_torch(model)
    _policy_logits, value_logits = numpy_model.forward(X)
    probs = value_probs(value_logits)
    predicted = probs[:, 0]
    actual = value_targets[:, 0]

    return _bucket_calibration(predicted, actual, n_buckets=n_buckets)
