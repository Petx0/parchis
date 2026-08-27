"""
Training step and the Phase 2 bootstrap training loop
(docs/AGENT_REBUILD_PLAN.md Part 3 item 11, Part 4's hyperparameter table).

The full round loop this module is eventually meant to hold (replay
buffer, promotion gate, continuous self-play -- see Part 3's package
layout and Phase 3's description) is NOT built yet. What's here is Phase
2's own, simpler shape: supervised-train the value head against the
seat-win distribution and the policy head against the moves actually
played, once, on a fixed pre-generated dataset (parchis.az.selfplay),
with a held-out validation split and early stopping -- exactly item 11's
scope, no more.

Value loss uses torch's class-PROBABILITY form of cross-entropy (target
tensor shaped like the logits, each row summing to 1) rather than the
integer-class-label form, so a one-hot win target and a 1/num_players
draw-vector target (Part 4: "Truncation ... scored as a draw") are handled
by the exact same loss with no special-casing.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from parchis.az.net import AZNet
from parchis.az.selfplay import examples_to_arrays


def _validate_split_fractions(train_frac, val_frac):
    if not 0.0 < train_frac < 1.0 or not 0.0 <= val_frac < 1.0 or train_frac + val_frac >= 1.0:
        raise ValueError(f"invalid split fractions: train={train_frac}, val={val_frac}")


def split_indices_by_game(game_indices, train_frac=0.8, val_frac=0.1, seed=0):
    """Array-native form of split_by_game, for an already-packed dataset
    (e.g. loaded from disk) where materializing one Python dict per
    decision would be wasteful at multi-million-row scale. `game_indices`:
    (n,) array, one entry per decision (parchis.az.selfplay.generate_games'
    'game_index', matching examples_to_arrays' row order).

    Returns:
        tuple(np.ndarray, np.ndarray, np.ndarray): (train_mask, val_mask,
        test_mask), boolean, shape (n,), for indexing X/policy_targets/
        value_targets/etc. directly (e.g. X[train_mask]).
    """
    _validate_split_fractions(train_frac, val_frac)
    game_indices = np.asarray(game_indices)
    unique_games = np.unique(game_indices)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_games)
    n = len(shuffled)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    train_games = shuffled[:n_train]
    val_games = shuffled[n_train:n_train + n_val]
    test_games = shuffled[n_train + n_val:]

    return (
        np.isin(game_indices, train_games),
        np.isin(game_indices, val_games),
        np.isin(game_indices, test_games),
    )


def split_by_game(examples, train_frac=0.8, val_frac=0.1, seed=0):
    """Split `examples` (parchis.az.selfplay.generate_games' output, each
    tagged with 'game_index') into (train, val, test) example lists, split
    at the GAME level, never per-decision: decisions from the same game
    are highly correlated (most share, or nearly share, the same
    outcome), so shuffling individual decisions across the split would
    leak a "held-out" game's likely result into training/validation. See
    split_indices_by_game for the array-native equivalent (used for an
    already-packed, disk-scale dataset).

    test_frac is implicitly 1 - train_frac - val_frac; the test split is
    what the calibration gate (item 12) and any final reporting should use
    -- val is for early stopping DURING training, not final evaluation.
    """
    _validate_split_fractions(train_frac, val_frac)

    game_indices = np.array([ex['game_index'] for ex in examples])
    train_mask, val_mask, test_mask = split_indices_by_game(
        game_indices, train_frac=train_frac, val_frac=val_frac, seed=seed,
    )
    train = [ex for ex, keep in zip(examples, train_mask) if keep]
    val = [ex for ex, keep in zip(examples, val_mask) if keep]
    test = [ex for ex, keep in zip(examples, test_mask) if keep]
    return train, val, test


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _forward_losses(model, X, policy_targets, value_targets):
    """policy_targets: (batch,) int64 class indices. value_targets:
    (batch, num_players) class-PROBABILITY targets (one-hot or a draw
    vector) -- see module docstring."""
    policy_logits, value_logits = model(X)
    policy_loss = F.cross_entropy(policy_logits, policy_targets)
    value_loss = F.cross_entropy(value_logits, value_targets)
    return policy_loss, value_loss


def _train_step(model, optimizer, xb, pb, vb, value_loss_weight):
    """One gradient step on a single batch already on the right device.
    Returns (loss, value_loss, policy_loss) as plain floats. Shared by
    bootstrap_train_arrays and bootstrap_train_sharded so the two data-
    loading strategies (in-memory permutation vs. shard streaming) don't
    duplicate the optimization step itself."""
    policy_loss, value_loss = _forward_losses(model, xb, pb, vb)
    loss = policy_loss + value_loss_weight * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), value_loss.item(), policy_loss.item()


def _validate(model, X_val_t, policy_val_t, value_val_t, value_loss_weight):
    """Returns (val_loss, val_value_loss, val_policy_loss) as plain
    floats, model.eval()'d and under no_grad. Shared by both training
    loops for the same reason as _train_step."""
    model.eval()
    with torch.no_grad():
        val_policy_loss, val_value_loss = _forward_losses(model, X_val_t, policy_val_t, value_val_t)
        val_loss = (val_policy_loss + value_loss_weight * val_value_loss).item()
        return val_loss, val_value_loss.item(), val_policy_loss.item()


def _check_early_stopping(model, val_loss, epoch, best_val_loss, best_state, epochs_without_improvement, patience):
    """Shared bookkeeping: returns (best_val_loss, best_state,
    epochs_without_improvement, should_stop)."""
    if val_loss < best_val_loss - 1e-5:
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return val_loss, best_state, 0, False
    epochs_without_improvement += 1
    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})", flush=True)
        return best_val_loss, best_state, epochs_without_improvement, True
    return best_val_loss, best_state, epochs_without_improvement, False


def bootstrap_train(train_examples, val_examples, num_players, **kwargs):
    """Dict-based convenience wrapper around bootstrap_train_arrays, for
    parchis.az.selfplay.generate_games' list-of-dicts output directly (fine
    at the scale parchis/tests/test_train.py exercises; for a disk-scale
    dataset already packed into arrays, call bootstrap_train_arrays
    directly and use split_indices_by_game -- materializing one Python
    dict per decision at multi-million-row scale is wasteful). See
    bootstrap_train_arrays for the full parameter list and return value.
    """
    X_train, policy_train, value_train = examples_to_arrays(train_examples, num_players)
    X_val, policy_val, value_val = examples_to_arrays(val_examples, num_players)
    return bootstrap_train_arrays(
        X_train, policy_train, value_train, X_val, policy_val, value_val, num_players, **kwargs,
    )


def bootstrap_train_arrays(X_train, policy_train, value_train, X_val, policy_val, value_val,
                            num_players, hidden_sizes=(256, 256), learning_rate=1e-3,
                            weight_decay=1e-4, batch_size=512, max_epochs=50, patience=5,
                            value_loss_weight=1.0, seed=0, log_every=1, init_state_dict=None):
    """
    AdamW + cosine LR schedule + weight decay (Part 4's table); early
    stopping on validation loss (policy_loss + value_loss_weight *
    value_loss -- Phase 2 doesn't mandate a specific weighting between
    heads; the default is equal weight, but the policy head's loss runs
    larger and improves faster than the value head's on the shared trunk,
    so a run that specifically needs a stronger value signal for
    search-based evaluation (item 13's gate) can raise
    value_loss_weight), with the best-validation-loss epoch's weights
    restored at the end.

    X_train/X_val: (n, input_size) float32. policy_train/policy_val:
    either (n,) int64 class indices (Phase 2's hard imitation label) OR
    (n, 4) float32 class-probability targets (Phase 3's soft z_policy,
    parchis.az.selfplay.round_examples_to_arrays) -- F.cross_entropy
    dispatches on the target's own dtype/shape, so no branching is needed
    here. value_train/value_val: (n, num_players) float32 class-
    probability targets (one-hot, a draw vector, or Phase 3's blended
    z_value).

    init_state_dict: if given (e.g. loaded via torch.load from a prior
    round's checkpoint), the model WARM-STARTS from these weights instead
    of AZNet's own random init -- Part 3 Phase 3's "cap warm-start
    epochs": each self-play round retrains from the current champion,
    not from scratch. None (the default) preserves Phase 2's original
    from-scratch behavior exactly.

    Returns:
        tuple(AZNet, list[dict]): (best model, on CPU, eval() mode;
        per-epoch history: [{'epoch', 'train_loss', 'train_value_loss',
        'train_policy_loss', 'val_loss', 'val_value_loss',
        'val_policy_loss'}, ...]).
    """
    torch.manual_seed(seed)
    device = _device()

    model = AZNet(X_train.shape[1], num_players, hidden_sizes=hidden_sizes).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    X_train_t = torch.from_numpy(X_train)
    policy_train_t = torch.from_numpy(policy_train)
    value_train_t = torch.from_numpy(value_train)
    X_val_t = torch.from_numpy(X_val).to(device)
    policy_val_t = torch.from_numpy(policy_val).to(device)
    value_val_t = torch.from_numpy(value_val).to(device)

    n_train = X_train_t.shape[0]
    rng = np.random.default_rng(seed)

    history = []
    best_val_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        perm = rng.permutation(n_train)
        loss_sum = value_loss_sum = policy_loss_sum = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_train_t[idx].to(device)
            pb = policy_train_t[idx].to(device)
            vb = value_train_t[idx].to(device)

            loss, value_loss, policy_loss = _train_step(model, optimizer, xb, pb, vb, value_loss_weight)
            loss_sum += loss
            value_loss_sum += value_loss
            policy_loss_sum += policy_loss
            n_batches += 1

        scheduler.step()
        val_loss, val_value_loss, val_policy_loss = _validate(
            model, X_val_t, policy_val_t, value_val_t, value_loss_weight,
        )

        entry = {
            'epoch': epoch,
            'train_loss': loss_sum / n_batches,
            'train_value_loss': value_loss_sum / n_batches,
            'train_policy_loss': policy_loss_sum / n_batches,
            'val_loss': val_loss, 'val_value_loss': val_value_loss, 'val_policy_loss': val_policy_loss,
        }
        history.append(entry)
        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch}: train_loss={entry['train_loss']:.4f} "
                  f"val_loss={val_loss:.4f} (value={val_value_loss:.4f} policy={val_policy_loss:.4f})")

        best_val_loss, best_state, epochs_without_improvement, should_stop = _check_early_stopping(
            model, val_loss, epoch, best_val_loss, best_state, epochs_without_improvement, patience,
        )
        if should_stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu()
    model.eval()
    return model, history


def _load_shard(path):
    """Loads one shard file's (X, policy_targets, value_targets) --
    ignores game_indices/mover_seats, which split_shards/callers already
    used to decide train/val/test membership before this is called."""
    data = np.load(path)
    return data["X"], data["policy_targets"], data["value_targets"]


def _load_and_concat_shards(paths):
    Xs, ps, vs = [], [], []
    for path in paths:
        X, p, v = _load_shard(path)
        Xs.append(X)
        ps.append(p)
        vs.append(v)
    return np.concatenate(Xs), np.concatenate(ps), np.concatenate(vs)


def split_shards(shard_paths, train_frac=0.8, val_frac=0.1, seed=0):
    """Partition a list of shard file paths into (train, val, test)
    groups, at the SHARD level rather than by individual game -- for a
    corpus generated in independently-seeded shards (see
    parchis/az/selfplay.py-based sharded generation scripts) that's too
    large to hold in memory as one array (see bootstrap_train_sharded).
    Every game within a given shard is already independent of every other
    shard's games (distinct seeds), so a shard-level split carries no
    leakage risk beyond what a game-level split already avoids.

    Returns:
        tuple(list, list, list): (train_paths, val_paths, test_paths), each
        GUARANTEED non-empty (a 3-way split is meaningless with an empty
        group -- bootstrap_train_sharded's val_shard_paths in particular
        must be non-empty, since _load_and_concat_shards can't concatenate
        zero arrays). Requires at least 3 shards; raises ValueError below
        that, since 3 non-empty groups can't come from fewer than 3 items.
    """
    _validate_split_fractions(train_frac, val_frac)
    shard_paths = list(shard_paths)
    n = len(shard_paths)
    if n < 3:
        raise ValueError(f"split_shards needs at least 3 shards (>=1 each for train/val/test), got {n}")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n).tolist()
    n_val = max(1, min(n - 2, int(round(n * val_frac))))
    n_test = max(1, min(n - n_val - 1, int(round(n * (1.0 - train_frac - val_frac)))))
    n_train = n - n_val - n_test

    train_idx = shuffled[:n_train]
    val_idx = shuffled[n_train:n_train + n_val]
    test_idx = shuffled[n_train + n_val:]
    return (
        [shard_paths[i] for i in train_idx],
        [shard_paths[i] for i in val_idx],
        [shard_paths[i] for i in test_idx],
    )


def split_shards_train_val(shard_paths, val_frac=0.1, seed=0):
    """Two-way (train, val) shard split -- for parchis.az.round_loop's
    replay buffer (docs/AGENT_REBUILD_PLAN.md Part 3 Phase 3): unlike
    split_shards (Phase 2's one-time bootstrap, which also needs a held-
    out TEST group for the calibration gate), a round's own training step
    only needs train/val (early stopping during training; the real
    "test" of a round's output is the promotion duplicate-match, an
    entirely different kind of evaluation, not a held-out loss). Requires
    at least 2 shards (both groups non-empty); a round's replay buffer
    (the last ~3 rounds' shards) is expected to comfortably clear this.

    Returns:
        tuple(list, list): (train_paths, val_paths), both non-empty,
        disjoint, covering every shard.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be within (0, 1), got {val_frac}")
    shard_paths = list(shard_paths)
    n = len(shard_paths)
    if n < 2:
        raise ValueError(f"split_shards_train_val needs at least 2 shards (>=1 each for train/val), got {n}")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n).tolist()
    n_val = max(1, min(n - 1, int(round(n * val_frac))))
    val_idx = shuffled[:n_val]
    train_idx = shuffled[n_val:]
    return [shard_paths[i] for i in train_idx], [shard_paths[i] for i in val_idx]


def bootstrap_train_sharded(train_shard_paths, val_shard_paths, num_players,
                             hidden_sizes=(256, 256), learning_rate=1e-3, weight_decay=1e-4,
                             batch_size=4096, max_epochs=40, patience=6,
                             value_loss_weight=1.0, seed=0, log_every=1, init_state_dict=None):
    """
    Same optimizer/schedule/early-stopping contract as bootstrap_train_arrays
    (see its docstring, including the init_state_dict warm-start and the
    policy-target dtype dispatch), for a dataset too large to hold in
    memory as one array: validation shards are loaded ONCE and held in
    memory for the whole run (assumed to comfortably fit -- they're meant
    to be a small fraction of the corpus); training shards are streamed
    one at a time, in a freshly shuffled order EVERY epoch, so a full
    epoch still means a full pass over every training shard -- never more
    than one training shard's worth of rows is held in memory at once.

    Returns: identical shape to bootstrap_train_arrays -- (model, history).
    """
    if not train_shard_paths or not val_shard_paths:
        raise ValueError(
            f"bootstrap_train_sharded needs non-empty train and val shard lists, got "
            f"{len(train_shard_paths)} train / {len(val_shard_paths)} val (see split_shards)"
        )

    torch.manual_seed(seed)
    device = _device()

    print(f"Loading {len(val_shard_paths)} validation shards...", flush=True)
    X_val, policy_val, value_val = _load_and_concat_shards(val_shard_paths)
    X_val_t = torch.from_numpy(X_val).to(device)
    policy_val_t = torch.from_numpy(policy_val).to(device)
    value_val_t = torch.from_numpy(value_val).to(device)
    print(f"Validation set: {X_val.shape[0]} decisions", flush=True)

    model = AZNet(X_val.shape[1], num_players, hidden_sizes=hidden_sizes).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    rng = np.random.default_rng(seed)
    history = []
    best_val_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        shard_order = rng.permutation(len(train_shard_paths))
        loss_sum = value_loss_sum = policy_loss_sum = 0.0
        n_batches = 0

        for shard_i in shard_order:
            X_shard, policy_shard, value_shard = _load_shard(train_shard_paths[shard_i])
            n_shard = X_shard.shape[0]
            perm = rng.permutation(n_shard)
            X_shard_t = torch.from_numpy(X_shard)
            policy_shard_t = torch.from_numpy(policy_shard)
            value_shard_t = torch.from_numpy(value_shard)

            for start in range(0, n_shard, batch_size):
                idx = perm[start:start + batch_size]
                xb = X_shard_t[idx].to(device)
                pb = policy_shard_t[idx].to(device)
                vb = value_shard_t[idx].to(device)

                loss, value_loss, policy_loss = _train_step(model, optimizer, xb, pb, vb, value_loss_weight)
                loss_sum += loss
                value_loss_sum += value_loss
                policy_loss_sum += policy_loss
                n_batches += 1

            del X_shard, policy_shard, value_shard, X_shard_t, policy_shard_t, value_shard_t

        scheduler.step()
        val_loss, val_value_loss, val_policy_loss = _validate(
            model, X_val_t, policy_val_t, value_val_t, value_loss_weight,
        )

        entry = {
            'epoch': epoch,
            'train_loss': loss_sum / n_batches,
            'train_value_loss': value_loss_sum / n_batches,
            'train_policy_loss': policy_loss_sum / n_batches,
            'val_loss': val_loss, 'val_value_loss': val_value_loss, 'val_policy_loss': val_policy_loss,
        }
        history.append(entry)
        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch}: train_loss={entry['train_loss']:.4f} "
                  f"val_loss={val_loss:.4f} (value={val_value_loss:.4f} policy={val_policy_loss:.4f})",
                  flush=True)

        best_val_loss, best_state, epochs_without_improvement, should_stop = _check_early_stopping(
            model, val_loss, epoch, best_val_loss, best_state, epochs_without_improvement, patience,
        )
        if should_stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu()
    model.eval()
    return model, history


def save_checkpoint(model, config, history, run_dir):
    """Writes runs/<run_name>/model.pt (state_dict) and metrics.jsonl (one
    line per epoch) into `run_dir` (as returned by config.save())."""
    import json

    run_dir = Path(run_dir)
    torch.save(model.state_dict(), run_dir / "model.pt")
    with open(run_dir / "metrics.jsonl", "w") as f:
        for entry in history:
            f.write(json.dumps(entry) + "\n")
