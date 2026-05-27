"""Data utilities for o200k Token-Routed pretraining."""

from __future__ import annotations

import logging
import string

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset

from complexity.data.token_shards import TokenShardDataset, token_shard_frequencies
from complexity.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class RandomTokenDataset(IterableDataset):
    def __init__(self, vocab_size: int, seq_len: int, seed: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed)
        while True:
            ids = torch.randint(0, self.vocab_size, (self.seq_len + 1,), generator=gen)
            yield {"input_ids": ids[:-1], "labels": ids[1:]}


class LocalTextDataset(IterableDataset):
    def __init__(self, tokens: list[int], seq_len: int, seed: int):
        if len(tokens) < seq_len + 2:
            raise ValueError(f"Need at least {seq_len + 2} tokens, got {len(tokens)}")
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed)
        high = self.tokens.numel() - self.seq_len - 1
        while True:
            start = torch.randint(0, high + 1, (1,), generator=gen).item()
            chunk = self.tokens[start : start + self.seq_len + 1]
            yield {"input_ids": chunk[:-1], "labels": chunk[1:]}


class FineWebDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len: int, rank: int, world_size: int):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

    def __iter__(self):
        buffer: list[int] = []
        for idx, example in enumerate(self.dataset):
            if idx % self.world_size != self.rank:
                continue
            text = example.get("text", "")
            if not text:
                continue
            buffer.extend(self.tokenizer.encode(text, add_special_tokens=False))
            if self.tokenizer.eos_token_id is not None:
                buffer.append(self.tokenizer.eos_token_id)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }


def load_text_tokens(path: str, tokenizer_path: str) -> list[int]:
    tokenizer = Tokenizer.load(tokenizer_path)
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)
    logger.info(f"Text dataset: {path} ({len(tokens):,} tokens)")
    return tokens


def infer_vocab_size(args) -> int:
    if args.vocab_size is not None:
        return args.vocab_size
    vocab_size = Tokenizer.load(args.tokenizer).vocab_size
    logger.info(f"Tokenizer vocab size: {vocab_size:,} ({args.tokenizer})")
    return vocab_size


def text_token_frequencies(path: str, tokenizer_path: str, vocab_size: int) -> torch.Tensor:
    tokens = load_text_tokens(path, tokenizer_path)
    ids = torch.tensor(tokens, dtype=torch.long)
    ids = ids[(ids >= 0) & (ids < vocab_size)]
    freqs = torch.zeros(vocab_size, dtype=torch.float32)
    if ids.numel() > 0:
        freqs.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
    logger.info(
        f"Zipf routing frequencies: {int(freqs.sum().item()):,} tokens, "
        f"{int((freqs > 0).sum().item()):,} vocab entries"
    )
    return freqs


def _hash_class_window(class_window: torch.Tensor, num_buckets: int) -> torch.Tensor:
    """Polynomial hash of a window of token classes (last dim) into num_buckets.

    Each class digit occupies log2(8) = 3 bits (we have 8 lexical classes), so
    the polynomial base is 8. For K=4 the raw space is 8^4 = 4096, then reduced
    modulo num_buckets to bound the signature space.
    """
    K = class_window.shape[-1]
    weights = torch.tensor(
        [8**i for i in range(K)], dtype=torch.long, device=class_window.device
    )
    sig_raw = (class_window.long() * weights).sum(-1)
    return sig_raw % int(num_buckets)


def text_context_sig_top_n(
    path: str,
    tokenizer_path: str,
    vocab_size: int,
    top_n: int,
    window: int,
    num_buckets: int,
    token_class_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the top-N most frequent (context_signature, cur_id) keys.

    Returns (keys, counts) with keys = sig * vocab_size + cur_id, sorted ascending.
    The first `window` positions of the stream use class 0 padding for any
    missing predecessor (consistent with how the runtime fallback handles
    sequence boundaries).
    """
    tokens = load_text_tokens(path, tokenizer_path)
    ids = torch.tensor(tokens, dtype=torch.long)
    ids = ids[(ids >= 0) & (ids < vocab_size)]
    if ids.numel() < window + 1:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    classes = token_class_table.detach().cpu().long()[ids]  # [T]
    T = ids.numel()
    padded = torch.cat([torch.zeros(window, dtype=torch.long), classes])  # [T+window]
    windows = padded.unfold(0, window, 1)[:T]  # [T, window]: window[t] = classes[t-K..t-1]
    sigs = _hash_class_window(windows, num_buckets)  # [T]
    keys = sigs * vocab_size + ids
    unique_keys, _, counts = torch.unique(
        keys, return_inverse=True, return_counts=True
    )
    n = min(int(top_n), int(unique_keys.numel()))
    top_counts, top_idx = torch.topk(counts, n)
    top_keys = unique_keys[top_idx]
    sort_idx = torch.argsort(top_keys)
    sorted_keys = top_keys[sort_idx]
    sorted_counts = top_counts[sort_idx]
    logger.info(
        f"Context-sig routing: {int(T):,} tokens, "
        f"window=K={window}, buckets={num_buckets}, "
        f"{int(unique_keys.numel()):,} unique (sig, cur) keys, top {n:,} kept "
        f"(coverage: {float(sorted_counts.sum() / counts.sum().clamp_min(1)):.1%} of token occurrences)"
    )
    return sorted_keys.long(), sorted_counts.long()


def build_key_expert_mapping(
    key_counts: torch.Tensor, num_experts: int
) -> torch.Tensor:
    """Greedy bin-packing of routing keys onto experts, balanced by count.

    Iterates keys in descending count order and assigns each to the currently
    least-loaded expert. Returns experts[i] for the i-th key in the input
    order (caller is expected to keep keys and experts aligned). Used by the
    context-signature routing strategy to assign each (sig, cur_id) key.
    """
    n = int(key_counts.numel())
    experts = torch.empty(n, dtype=torch.long)
    if n == 0:
        return experts
    counts = key_counts.detach().cpu().long()
    order = torch.argsort(counts, descending=True)
    loads = [0] * num_experts
    for pos in order.tolist():
        e = min(range(num_experts), key=lambda i: loads[i])
        experts[pos] = e
        loads[e] += int(counts[pos].item())
    return experts


def text_token_cooccurrence_embeddings(
    path: str,
    tokenizer_path: str,
    vocab_size: int,
    target_ids: torch.Tensor,
    window: int = 4,
    dim: int = 64,
    num_features: int = 2000,
    seed: int = 0,
) -> dict[int, "object"]:
    """Distributional (≈ semantic) embeddings for ``target_ids`` from the corpus.

    Builds a dense ``(n_targets, num_features)`` co-occurrence matrix of each
    target token against the ``num_features`` most frequent context tokens
    inside a +/-``window`` neighbourhood, applies PPMI, then truncated SVD to
    ``dim``. Returns ``{token_id -> np.ndarray[dim]}`` (L2-normalised); targets
    that never co-occur get a zero vector. Deterministic given ``seed`` so the
    mapping can be recomputed identically at load time.
    """
    import numpy as np
    from sklearn.decomposition import TruncatedSVD

    tokens = load_text_tokens(path, tokenizer_path)
    ids = torch.tensor(tokens, dtype=torch.long)
    ids = ids[(ids >= 0) & (ids < vocab_size)]
    targets = sorted({int(t) for t in target_ids.tolist()})
    if ids.numel() < 2 or not targets:
        return {t: np.zeros(dim, dtype=np.float32) for t in targets}

    freqs = torch.bincount(ids, minlength=vocab_size)
    feature_ids = torch.topk(freqs, min(num_features, vocab_size)).indices.tolist()
    feature_index = {int(f): j for j, f in enumerate(feature_ids)}
    target_index = {t: i for i, t in enumerate(targets)}

    cooc = np.zeros((len(targets), len(feature_ids)), dtype=np.float64)
    seq = ids.tolist()
    n = len(seq)
    for pos, tok in enumerate(seq):
        ti = target_index.get(tok)
        if ti is None:
            continue
        lo, hi = max(0, pos - window), min(n, pos + window + 1)
        for ctx in range(lo, hi):
            if ctx == pos:
                continue
            fj = feature_index.get(seq[ctx])
            if fj is not None:
                cooc[ti, fj] += 1.0

    total = cooc.sum()
    if total <= 0:
        return {t: np.zeros(dim, dtype=np.float32) for t in targets}
    # PPMI: log( p(t,f) / (p(t) p(f)) ), clipped at 0.
    p_tf = cooc / total
    p_t = p_tf.sum(axis=1, keepdims=True)
    p_f = p_tf.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(p_tf / (p_t * p_f))
    ppmi = np.nan_to_num(np.maximum(pmi, 0.0))

    svd_dim = min(dim, max(1, min(ppmi.shape) - 1))
    emb = TruncatedSVD(n_components=svd_dim, random_state=seed).fit_transform(ppmi)
    if svd_dim < dim:
        emb = np.pad(emb, ((0, 0), (0, dim - svd_dim)))
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-12, None)
    return {t: emb[i].astype(np.float32) for t, i in target_index.items()}


def build_clustered_expert_mapping(
    keys: torch.Tensor,
    key_counts: torch.Tensor,
    num_experts: int,
    embeddings: dict[int, "object"],
    vocab_size: int,
    dim: int,
    balance_slack: float = 1.05,
    seed: int = 0,
) -> torch.Tensor:
    """Distributional clustering of routing keys onto experts (TR, deterministic).

    Drop-in replacement for :func:`build_key_expert_mapping` that puts
    *similar* keys on the *same* expert instead of pure load-balancing. Each key
    is represented by the distributional embedding of its ``cur_id``
    (``key % vocab_size``); k-means (k=``num_experts``, weighted by count) gives
    expert centroids, then keys are assigned greedily by descending count to the
    nearest centroid that still has capacity (``balance_slack`` * mean load),
    falling back to the least-loaded expert. Keeps experts aligned to ``keys``.
    """
    import numpy as np
    from sklearn.cluster import KMeans

    n = int(keys.numel())
    experts = torch.empty(n, dtype=torch.long)
    if n == 0:
        return experts

    counts = key_counts.detach().cpu().long().numpy()
    cur_ids = (keys.detach().cpu().long() % vocab_size).numpy()
    zero = np.zeros(dim, dtype=np.float32)
    X = np.stack([embeddings.get(int(c), zero) for c in cur_ids], axis=0)

    if num_experts <= 1 or not np.any(X):
        return build_key_expert_mapping(key_counts, num_experts)

    km = KMeans(n_clusters=num_experts, random_state=seed, n_init=10)
    km.fit(X, sample_weight=counts.astype(np.float64))
    centroids = km.cluster_centers_

    capacity = balance_slack * counts.sum() / num_experts
    loads = np.zeros(num_experts, dtype=np.float64)
    dist = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)  # (n, E)
    order = np.argsort(-counts)
    for pos in order:
        ranked = np.argsort(dist[pos])
        chosen = next((e for e in ranked if loads[e] + counts[pos] <= capacity), None)
        if chosen is None:
            chosen = int(np.argmin(loads))
        experts[int(pos)] = int(chosen)
        loads[int(chosen)] += counts[pos]
    return experts


def build_ctx_expert_mapping(
    mode: str,
    keys: torch.Tensor,
    counts: torch.Tensor,
    num_experts: int,
    *,
    text_file: str,
    tokenizer_path: str,
    vocab_size: int,
    window: int,
    slack: float = 1.05,
    dim: int = 64,
) -> torch.Tensor:
    """Dispatch (sig, cur) keys onto experts per ``mode``.

    ``balance`` -> load bin-packing (:func:`build_key_expert_mapping`).
    ``distributional`` -> co-occurrence clustering
    (:func:`build_clustered_expert_mapping`). Deterministic, so both the
    training run and any load-time recompute produce the identical mapping.
    """
    if mode == "distributional":
        targets = torch.unique(keys % vocab_size)
        embeddings = text_token_cooccurrence_embeddings(
            text_file, tokenizer_path, vocab_size, targets, window=window, dim=dim
        )
        return build_clustered_expert_mapping(
            keys, counts, num_experts, embeddings, vocab_size, dim, balance_slack=slack
        )
    return build_key_expert_mapping(counts, num_experts)


def tokenizer_token_classes(tokenizer_path: str, vocab_size: int) -> torch.Tensor:
    """Classify each token into coarse lexical buckets for static routing."""

    tokenizer = Tokenizer.load(tokenizer_path)
    classes = torch.zeros(vocab_size, dtype=torch.long)
    encoding = getattr(getattr(tokenizer, "_tokenizer", None), "encoding", None)
    for token_id in range(vocab_size):
        text = _decode_token_for_class(tokenizer, encoding, token_id)
        classes[token_id] = _classify_token_text(text)
    counts = torch.bincount(classes, minlength=8)
    logger.info(
        "Token classes: "
        + ", ".join(f"{idx}={int(count)}" for idx, count in enumerate(counts.tolist()) if count)
    )
    return classes


def _decode_token_for_class(tokenizer: Tokenizer, encoding, token_id: int) -> str:
    try:
        if encoding is not None and hasattr(encoding, "decode_single_token_bytes"):
            return encoding.decode_single_token_bytes(token_id).decode("utf-8", errors="replace")
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        return ""


def _classify_token_text(text: str) -> int:
    if not text:
        return 0
    if text.isspace():
        return 1
    stripped = text.strip()
    if not stripped:
        return 1
    if stripped.isdigit():
        return 2
    if stripped.isalpha() and stripped.isascii():
        return 3
    if stripped.isalnum() and stripped.isascii():
        return 4
    if any(ord(ch) > 127 for ch in stripped):
        return 6
    if all(ch in string.punctuation for ch in stripped):
        return 5
    return 7


def split_tokens(tokens: list[int], eval_ratio: float) -> tuple[list[int], list[int]]:
    n_eval = max(2048, int(len(tokens) * eval_ratio))
    n_eval = min(n_eval, max(1, len(tokens) // 5))
    return tokens[:-n_eval], tokens[-n_eval:]


def build_loaders(args, config, rank: int, world_size: int):
    if args.dataset == "fineweb":
        tokenizer = Tokenizer.load(args.tokenizer)
        if rank == 0:
            logger.info("Dataset: FineWeb-Edu sample-10BT streaming")
        train_ds = FineWebDataset(tokenizer, args.seq_len, rank, world_size)
        eval_ds = FineWebDataset(tokenizer, args.seq_len, rank, world_size) if args.eval_steps > 0 else None
    elif args.dataset == "tokens":
        if not args.tokens_path:
            raise ValueError("--tokens-path is required when --dataset tokens")
        if rank == 0:
            logger.info(f"Dataset: token shard mmap ({args.tokens_path})")
        train_ds = TokenShardDataset(
            args.tokens_path,
            args.seq_len,
            rank=rank,
            world_size=world_size,
            seed=args.seed,
            split="train",
            eval_ratio=args.eval_ratio,
        )
        eval_ds = (
            TokenShardDataset(
                args.tokens_path,
                args.seq_len,
                rank=rank,
                world_size=world_size,
                seed=args.seed + 10_000,
                split="eval",
                eval_ratio=args.eval_ratio,
            )
            if args.eval_steps > 0 else None
        )
    elif args.dataset == "text":
        if not args.text_file:
            raise ValueError("--text-file is required when --dataset text")
        tokens = load_text_tokens(args.text_file, args.tokenizer)
        train_tokens, eval_tokens = split_tokens(tokens, args.eval_ratio)
        train_ds = LocalTextDataset(train_tokens, args.seq_len, args.seed + rank)
        eval_ds = LocalTextDataset(eval_tokens, args.seq_len, args.seed + 10_000 + rank)
    else:
        train_ds = RandomTokenDataset(config.vocab_size, args.seq_len, args.seed + rank)
        eval_ds = RandomTokenDataset(config.vocab_size, args.seq_len, args.seed + 10_000 + rank)

    loader_kwargs = {"batch_size": args.batch_size, "pin_memory": False}
    if args.num_workers > 0:
        loader_kwargs.update(num_workers=args.num_workers, persistent_workers=True)
    eval_loader = DataLoader(eval_ds, **loader_kwargs) if eval_ds is not None else None
    return DataLoader(train_ds, **loader_kwargs), eval_loader


def batch_expert_counts(raw_model, input_ids: torch.Tensor, num_experts: int, distributed: bool) -> torch.Tensor:
    """Return per-expert token counts for the current batch."""

    for module in raw_model.modules():
        if hasattr(module, "token_to_expert"):
            token_to_expert = getattr(module, "topk_token_to_expert", module.token_to_expert)
            if token_to_expert.ndim == 2:
                token_ids = input_ids.clamp(0, token_to_expert.shape[1] - 1)
                expert_ids = token_to_expert[:, token_ids].reshape(-1)
            else:
                token_ids = input_ids.clamp(0, token_to_expert.numel() - 1)
                expert_ids = token_to_expert[token_ids].reshape(-1)
            counts = torch.bincount(expert_ids, minlength=num_experts).to(
                device=input_ids.device,
                dtype=torch.float32,
            )
            if distributed:
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            return counts
    counts = torch.ones(num_experts, device=input_ids.device, dtype=torch.float32)
    if distributed:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return counts


__all__ = [
    "RandomTokenDataset",
    "LocalTextDataset",
    "FineWebDataset",
    "TokenShardDataset",
    "batch_expert_counts",
    "build_loaders",
    "infer_vocab_size",
    "text_token_frequencies",
    "text_context_sig_top_n",
    "build_key_expert_mapping",
    "text_token_cooccurrence_embeddings",
    "build_clustered_expert_mapping",
    "build_ctx_expert_mapping",
    "token_shard_frequencies",
    "tokenizer_token_classes",
]
