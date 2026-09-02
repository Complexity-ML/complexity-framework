"""Interactive 3-D t-SNE of TR-Hash routed-expert contributions.

This script is deliberately narrower than the historical visualization:

* it loads either a historical exported checkpoint or a current
  ``tr_hash_engine`` bundle and preserves its persisted per-layer routes;
* it feeds natural validation text without a chat template;
* it visualizes only the two routed residual contributions, never the shared
  MLP output or the full MLP output;
* every point is one real token contribution from one selected route; and
* it records checkpoint, tokenizer, probe-data, and output hashes.

The embedding is exploratory. t-SNE cluster separation is not evidence that
an expert is functionally specialized or that routing improves model quality.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from complexity.config import ModelConfig
from complexity.core.mlp.tr_hash_engine import TRHashEngineMLP
from complexity.models import ComplexityModel
from complexity.tokenizer import Tokenizer
from complexity.utils.token_routed_conversion import convert_token_routed_checkpoint

DEFAULT_CHECKPOINT = Path(
    "artifacts/remote_runs/tr_hash_moe_500m_20b/hf_base"
)
DEFAULT_PROBE = Path(
    Path.home(),
    ".cache/huggingface/datasets/downloads/extracted/"
    "5df6cfdafae620c63511c300cb6637597d02ac19aa2ca44c027ac428b4352164/"
    "physicaliqa-train-dev/dev.jsonl",
)
DEFAULT_LAYERS = (0, 5, 11, 17, 23)
DEFAULT_MODEL_LABEL = "TR-Hash 500M pretrained"
DEFAULT_ARTIFACT_LABEL = "tr_hash_moe_500m_20b/hf_base"
DEFAULT_SOURCE_TOKEN_EXPOSURE = 19_999_752_192
EXPERT_COLORS = ("#f87171", "#60a5fa", "#34d399", "#fbbf24")
ROUTE_SYMBOLS = ("circle", "diamond")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",")))
    if not layers:
        raise argparse.ArgumentTypeError("at least one layer is required")
    if min(layers) < 0:
        raise argparse.ArgumentTypeError("layer indices must be non-negative")
    return layers


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_current_engine_model(
    state: dict[str, torch.Tensor],
    raw_config: dict,
) -> ComplexityModel:
    config = ModelConfig.from_dict(raw_config)
    config.use_custom_kernels = False
    model = ComplexityModel(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    tolerated_suffixes = (
        "topk_token_to_expert",
        "rotary_emb.inv_freq",
        "pair_hash_route_codes",
        "pair_hash_expert_pairs",
    )
    unexpected_missing = [
        key for key in missing if not key.endswith(tolerated_suffixes)
    ]
    if unexpected_missing or unexpected:
        raise RuntimeError(
            "Checkpoint mismatch: "
            f"missing={unexpected_missing}, unexpected={list(unexpected)}"
        )
    return model


def load_tr_hash_model(checkpoint: Path, device: torch.device) -> ComplexityModel:
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    state = load_file(str(weights_path), device="cpu")
    mlp_type = raw_config.get("mlp_type")
    if mlp_type in {"token_routed", "sort_split", "sort_split_moe"}:
        model = convert_token_routed_checkpoint(state, raw_config)
    elif mlp_type == "tr_hash_engine":
        model = _load_current_engine_model(state, raw_config)
    else:
        raise ValueError(
            "unsupported routed checkpoint layout: "
            f"mlp_type={mlp_type!r}"
        )
    del state
    model.eval().requires_grad_(False).to(device)
    return model


def read_probe_records(path: Path, max_records: int) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            records.append(
                {
                    "goal": str(row["goal"]).strip(),
                    "sol1": str(row["sol1"]).strip(),
                    "sol2": str(row["sol2"]).strip(),
                }
            )
            if len(records) >= max_records:
                break
    if not records:
        raise ValueError(f"probe file has no records: {path}")
    return records


def encode_probe(tokenizer: Tokenizer, records: Iterable[dict[str, str]]) -> list[int]:
    # No assistant/system wrapper: this is a base-pretraining diagnostic.
    text = "\n\n".join(
        "\n".join((record["goal"], record["sol1"], record["sol2"]))
        for record in records
    )
    return [int(token_id) for token_id in tokenizer.encode(text)]


@dataclass
class LayerCapture:
    hidden: list[torch.Tensor]
    residual: list[torch.Tensor]
    token_ids: list[torch.Tensor]


class ContextCollector:
    """Capture contextual MLP inputs for selected layers."""

    def __init__(self, model: ComplexityModel, layer_indices: tuple[int, ...]):
        self.model = model
        self.layer_indices = layer_indices
        self.captures = {
            layer_idx: LayerCapture(hidden=[], residual=[], token_ids=[])
            for layer_idx in layer_indices
        }
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "ContextCollector":
        for layer_idx in self.layer_indices:
            layer = self.model.layers[layer_idx]
            mlp = layer.mlp
            if not isinstance(mlp, TRHashEngineMLP):
                raise TypeError(
                    f"layer {layer_idx} is {type(mlp).__name__}, expected TRHashEngineMLP"
                )
            self.handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(
                    self._residual_hook(layer_idx),
                )
            )
            self.handles.append(
                mlp.register_forward_pre_hook(
                    self._hook(layer_idx),
                    with_kwargs=True,
                )
            )
        return self

    def __exit__(self, *_exc) -> None:
        for handle in self.handles:
            handle.remove()

    def _residual_hook(self, layer_idx: int):
        def hook(_module, args):
            residual = args[0].detach().reshape(-1, args[0].shape[-1])
            self.captures[layer_idx].residual.append(residual.cpu())

        return hook

    def _hook(self, layer_idx: int):
        def hook(_module, args, kwargs):
            hidden = args[0].detach().reshape(-1, args[0].shape[-1])
            token_ids = kwargs.get("token_ids")
            if token_ids is None:
                raise RuntimeError("TR-Hash MLP hook did not receive token_ids")
            self.captures[layer_idx].hidden.append(hidden.cpu())
            self.captures[layer_idx].token_ids.append(token_ids.detach().reshape(-1).cpu())

        return hook


def collect_contexts(
    model: ComplexityModel,
    token_ids: list[int],
    layer_indices: tuple[int, ...],
    *,
    sequence_length: int,
    sequences: int,
    device: torch.device,
) -> dict[
    int,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray],
]:
    needed = sequence_length * sequences
    if len(token_ids) < needed:
        repeats = (needed + len(token_ids) - 1) // len(token_ids)
        token_ids = (token_ids * repeats)[:needed]
    else:
        token_ids = token_ids[:needed]

    sequence_ids = np.repeat(np.arange(sequences, dtype=np.int32), sequence_length)
    positions = np.tile(np.arange(sequence_length, dtype=np.int32), sequences)
    ids = torch.tensor(token_ids, dtype=torch.long).view(sequences, sequence_length)

    with torch.inference_mode(), ContextCollector(model, layer_indices) as collector:
        # Separate sequences prevent attention from crossing artificial probe
        # boundaries while keeping activation collection memory bounded.
        for row in ids:
            model(row.unsqueeze(0).to(device), return_logits=False)

    result = {}
    for layer_idx, capture in collector.captures.items():
        result[layer_idx] = (
            torch.cat(capture.hidden, dim=0),
            torch.cat(capture.residual, dim=0),
            torch.cat(capture.token_ids, dim=0),
            sequence_ids,
            positions,
        )
    return result


def stratified_token_sample(
    route_table: torch.Tensor,
    token_ids: torch.Tensor,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample token positions while keeping every primary expert represented."""

    primary = route_table[0, token_ids].numpy()
    expert_ids = sorted(set(primary.tolist()))
    per_expert = max(1, count // len(expert_ids))
    chosen: list[int] = []
    for expert_id in expert_ids:
        candidates = np.flatnonzero(primary == expert_id)
        take = min(per_expert, len(candidates))
        chosen.extend(rng.choice(candidates, size=take, replace=False).tolist())
    remaining = min(count - len(chosen), len(primary) - len(chosen))
    if remaining > 0:
        pool = np.setdiff1d(np.arange(len(primary)), np.asarray(chosen), assume_unique=False)
        chosen.extend(rng.choice(pool, size=remaining, replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def routed_contributions(
    mlp: TRHashEngineMLP,
    hidden: torch.Tensor,
    token_ids: torch.Tensor,
    sample_indices: np.ndarray,
    device: torch.device,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
]:
    """Return real per-route residual contributions for sampled tokens."""

    engine = mlp.engine
    index = torch.from_numpy(sample_indices)
    sampled_x = hidden[index].to(device=device, dtype=engine.expert_gate.dtype)
    sampled_ids = token_ids[index].to(device)
    routes = engine.route_table[:, sampled_ids].to(device)
    route_weights = torch.tensor(
        engine.config.route_weights,
        device=device,
        dtype=sampled_x.dtype,
    )

    outputs = []
    experts = []
    ranks = []
    source_indices = []
    with torch.inference_mode():
        for route_rank in range(engine.config.top_k):
            expert_ids = routes[route_rank]
            contribution = torch.empty_like(sampled_x)
            for expert_id in range(engine.config.num_experts):
                mask = expert_ids == expert_id
                if not bool(mask.any().item()):
                    continue
                x = sampled_x[mask]
                intermediate = F.silu(x @ engine.expert_gate[expert_id]) * (
                    x @ engine.expert_up[expert_id]
                )
                contribution[mask] = intermediate @ engine.expert_down[expert_id]
            contribution *= (
                float(engine.config.routed_output_scale) * route_weights[route_rank]
            )
            outputs.append(contribution.float().cpu().numpy())
            experts.append(expert_ids.cpu().numpy())
            ranks.append(np.full(len(sample_indices), route_rank, dtype=np.int8))
            source_indices.append(sample_indices.copy())

    vectors = np.concatenate(outputs, axis=0)
    norms = np.linalg.norm(vectors, axis=1)
    unit_vectors = vectors / np.maximum(norms[:, None], 1e-12)
    return (
        unit_vectors,
        np.concatenate(experts),
        np.concatenate(ranks),
        np.concatenate(source_indices),
    ), norms


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "min": float(values.min()),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def audit_full_probe_routing(
    mlp: TRHashEngineMLP,
    hidden: torch.Tensor,
    residual: torch.Tensor,
    token_ids: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int = 512,
) -> dict:
    """Audit routing and routed-branch scale on every collected probe token."""

    if not (len(hidden) == len(residual) == len(token_ids)):
        raise ValueError("hidden, residual, and token_ids must contain the same tokens")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    engine = mlp.engine
    top_k = int(engine.config.top_k)
    num_experts = int(engine.config.num_experts)
    route_counts = np.zeros((top_k, num_experts), dtype=np.int64)
    pair_counts = np.zeros((num_experts, num_experts), dtype=np.int64)
    routed_norms: list[np.ndarray] = []
    residual_norms: list[np.ndarray] = []
    mlp_input_norms: list[np.ndarray] = []
    routed_to_residual: list[np.ndarray] = []
    routed_to_mlp_input: list[np.ndarray] = []
    all_routes: list[np.ndarray] = []
    route_table = engine.route_table.to(device)
    route_weights = torch.tensor(
        engine.config.route_weights,
        device=device,
        dtype=engine.expert_gate.dtype,
    )

    with torch.inference_mode():
        for start in range(0, len(token_ids), batch_size):
            stop = min(start + batch_size, len(token_ids))
            x = hidden[start:stop].to(device=device, dtype=engine.expert_gate.dtype)
            residual_batch = residual[start:stop].to(device=device, dtype=x.dtype)
            ids = token_ids[start:stop].to(device)
            routes = route_table[:, ids]
            routed = torch.zeros_like(x)

            for route_rank in range(top_k):
                expert_ids = routes[route_rank]
                contribution = torch.empty_like(x)
                for expert_id in range(num_experts):
                    mask = expert_ids == expert_id
                    if not bool(mask.any().item()):
                        continue
                    expert_input = x[mask]
                    intermediate = F.silu(
                        expert_input @ engine.expert_gate[expert_id]
                    ) * (expert_input @ engine.expert_up[expert_id])
                    contribution[mask] = intermediate @ engine.expert_down[expert_id]
                routed += contribution * (
                    float(engine.config.routed_output_scale)
                    * route_weights[route_rank]
                )

            routes_cpu = routes.cpu().numpy()
            all_routes.append(routes_cpu)
            for route_rank in range(top_k):
                route_counts[route_rank] += np.bincount(
                    routes_cpu[route_rank], minlength=num_experts
                )
            if top_k >= 2:
                np.add.at(pair_counts, (routes_cpu[0], routes_cpu[1]), 1)

            routed_norm = torch.linalg.vector_norm(routed.float(), dim=-1)
            residual_norm = torch.linalg.vector_norm(residual_batch.float(), dim=-1)
            mlp_input_norm = torch.linalg.vector_norm(x.float(), dim=-1)
            routed_norms.append(routed_norm.cpu().numpy())
            residual_norms.append(residual_norm.cpu().numpy())
            mlp_input_norms.append(mlp_input_norm.cpu().numpy())
            routed_to_residual.append(
                (routed_norm / residual_norm.clamp_min(1e-12)).cpu().numpy()
            )
            routed_to_mlp_input.append(
                (routed_norm / mlp_input_norm.clamp_min(1e-12)).cpu().numpy()
            )

    routes = np.concatenate(all_routes, axis=1)
    combined_counts = route_counts.sum(axis=0)
    token_count = int(len(token_ids))
    repeated_route_count = int(
        sum(
            len(set(routes[:, token_index].tolist())) < top_k
            for token_index in range(token_count)
        )
    )

    return {
        "tokens": token_count,
        "unique_token_ids": int(token_ids.unique().numel()),
        "sampling": "all_collected_probe_tokens",
        "route_counts": route_counts.tolist(),
        "route_fractions": (route_counts / token_count).tolist(),
        "combined_route_counts": combined_counts.tolist(),
        "combined_route_fractions": (combined_counts / (token_count * top_k)).tolist(),
        "primary_secondary_pair_counts": pair_counts.tolist() if top_k >= 2 else None,
        "tokens_with_repeated_expert_across_routes": repeated_route_count,
        "all_experts_observed_per_route": bool(np.all(route_counts > 0)),
        "routed_branch_norm": _distribution_summary(np.concatenate(routed_norms)),
        "residual_stream_norm": _distribution_summary(np.concatenate(residual_norms)),
        "mlp_input_norm": _distribution_summary(np.concatenate(mlp_input_norms)),
        "routed_to_residual_norm_ratio": _distribution_summary(
            np.concatenate(routed_to_residual)
        ),
        "routed_to_mlp_input_norm_ratio": _distribution_summary(
            np.concatenate(routed_to_mlp_input)
        ),
    }


def embed_layer(vectors: np.ndarray, perplexity: float, seed: int) -> tuple[np.ndarray, int, float]:
    pca_dim = min(50, vectors.shape[0] - 1, vectors.shape[1])
    reduced = PCA(n_components=pca_dim, random_state=seed).fit_transform(vectors)
    effective_perplexity = min(float(perplexity), float(vectors.shape[0] - 1))
    tsne = TSNE(
        n_components=3,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1500,
        random_state=seed,
    )
    return tsne.fit_transform(reduced), pca_dim, float(tsne.kl_divergence_)


def token_labels(tokenizer: Tokenizer, ids: np.ndarray) -> list[str]:
    labels = []
    for token_id in ids:
        decoded = tokenizer.decode([int(token_id)])
        labels.append(decoded.replace("\n", "\\n").replace("\t", "\\t") or "<empty>")
    return labels


def write_interactive_html(
    rows: list[dict],
    layers: tuple[int, ...],
    output_path: Path,
    model_label: str,
) -> None:
    import plotly.graph_objects as go

    figure = go.Figure()
    traces_per_layer = 8
    for layer_position, layer_idx in enumerate(layers):
        layer_rows = [row for row in rows if row["layer"] == layer_idx]
        for expert_id in range(4):
            for route_rank in range(2):
                selected = [
                    row
                    for row in layer_rows
                    if row["expert"] == expert_id and row["route_rank"] == route_rank
                ]
                custom = [
                    [
                        row["token"],
                        row["token_id"],
                        row["sequence"],
                        row["position"],
                        row["norm"],
                    ]
                    for row in selected
                ]
                figure.add_trace(
                    go.Scatter3d(
                        x=[row["x"] for row in selected],
                        y=[row["y"] for row in selected],
                        z=[row["z"] for row in selected],
                        mode="markers",
                        name=f"E{expert_id} · route {route_rank + 1}",
                        legendgroup=f"expert-{expert_id}",
                        marker={
                            "size": 3.2,
                            "color": EXPERT_COLORS[expert_id],
                            "symbol": ROUTE_SYMBOLS[route_rank],
                            "opacity": 0.72,
                            "line": {"width": 0},
                        },
                        customdata=custom,
                        hovertemplate=(
                            "token=%{customdata[0]}<br>token_id=%{customdata[1]}"
                            "<br>expert=E" + str(expert_id)
                            + " · route=" + str(route_rank + 1)
                            + "<br>sequence=%{customdata[2]} · position=%{customdata[3]}"
                            "<br>contribution norm=%{customdata[4]:.4f}<extra></extra>"
                        ),
                        visible=layer_position == 0,
                    )
                )

    buttons = []
    total_traces = len(layers) * traces_per_layer
    for layer_position, layer_idx in enumerate(layers):
        visible = [False] * total_traces
        start = layer_position * traces_per_layer
        visible[start : start + traces_per_layer] = [True] * traces_per_layer
        buttons.append(
            {
                "label": f"Layer {layer_idx + 1}",
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title": {
                            "text": (
                                f"{model_label} · routed expert contributions"
                                f" · layer {layer_idx + 1}"
                            )
                        }
                    },
                ],
            }
        )

    figure.update_layout(
        title={
            "text": (
                f"{model_label} · routed expert contributions"
                f" · layer {layers[0] + 1}"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_dark",
        paper_bgcolor="#090b10",
        plot_bgcolor="#090b10",
        font={"color": "#d7dde8", "family": "Inter, system-ui, sans-serif"},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.02,
            "yanchor": "bottom",
        },
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.01,
                "xanchor": "left",
                "y": 1.12,
                "yanchor": "top",
                "showactive": True,
            }
        ],
        annotations=[
            {
                "text": (
                    "PIQA validation probe · no chat template · routed branch only · "
                    "unit-normalized before PCA + t-SNE · exploratory, not a quality claim"
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.08,
                "showarrow": False,
                "font": {"size": 11, "color": "#8993a4"},
            }
        ],
        scene={
            "xaxis_title": "t-SNE 1",
            "yaxis_title": "t-SNE 2",
            "zaxis_title": "t-SNE 3",
            "bgcolor": "#090b10",
            "xaxis": {"gridcolor": "#232936", "zerolinecolor": "#394150"},
            "yaxis": {"gridcolor": "#232936", "zerolinecolor": "#394150"},
            "zaxis": {"gridcolor": "#232936", "zerolinecolor": "#394150"},
        },
        margin={"l": 0, "r": 0, "t": 118, "b": 58},
        height=760,
        autosize=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        str(output_path),
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "displaylogo": False},
        div_id="trhash-routed-expert-tsne",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--points", type=Path)
    parser.add_argument("--layers", type=parse_layers, default=DEFAULT_LAYERS)
    parser.add_argument("--probe-records", type=int, default=128)
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--tokens-per-layer", type=int, default=600)
    parser.add_argument("--perplexity", type=float, default=35.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    parser.add_argument("--artifact-label", default=DEFAULT_ARTIFACT_LABEL)
    parser.add_argument(
        "--source-token-exposure",
        type=int,
        default=DEFAULT_SOURCE_TOKEN_EXPOSURE,
    )
    parser.add_argument("--sft-tokens", type=int, default=0)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    probe = args.probe.resolve()
    output = args.output.resolve()
    metadata_path = (
        args.metadata.resolve()
        if args.metadata
        else output.with_suffix(".metadata.json")
    )
    points_path = (
        args.points.resolve()
        if args.points
        else output.with_suffix(".points.csv.gz")
    )
    device = resolve_device(args.device)
    print(f"device={device}")
    print(f"checkpoint={checkpoint}")
    print(f"probe={probe}")

    records = read_probe_records(probe, args.probe_records)
    tokenizer = Tokenizer.load(str(checkpoint))
    encoded = encode_probe(tokenizer, records)
    model = load_tr_hash_model(checkpoint, device)
    if max(args.layers) >= len(model.layers):
        raise ValueError(
            f"requested layer {max(args.layers)}, model has {len(model.layers)} layers"
        )

    contexts = collect_contexts(
        model,
        encoded,
        args.layers,
        sequence_length=args.sequence_length,
        sequences=args.sequences,
        device=device,
    )
    rng = np.random.default_rng(args.seed)
    all_rows: list[dict] = []
    layer_metrics = {}

    for layer_idx in args.layers:
        hidden, residual, ids, sequence_ids, positions = contexts[layer_idx]
        mlp = model.layers[layer_idx].mlp
        full_probe_audit = audit_full_probe_routing(
            mlp,
            hidden,
            residual,
            ids,
            device,
        )
        sample_indices = stratified_token_sample(
            mlp.engine.route_table.cpu(),
            ids,
            min(args.tokens_per_layer, len(ids)),
            rng,
        )
        packed, norms = routed_contributions(
            mlp,
            hidden,
            ids,
            sample_indices,
            device,
        )
        vectors, experts, route_ranks, source_indices = packed
        embedding, pca_dim, kl_divergence = embed_layer(
            vectors,
            args.perplexity,
            args.seed + layer_idx,
        )
        source_ids = ids[source_indices].numpy()
        labels = token_labels(tokenizer, source_ids)
        for point_idx, point in enumerate(embedding):
            source_index = int(source_indices[point_idx])
            all_rows.append(
                {
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                    "layer": int(layer_idx),
                    "expert": int(experts[point_idx]),
                    "route_rank": int(route_ranks[point_idx]),
                    "token_id": int(source_ids[point_idx]),
                    "token": labels[point_idx],
                    "sequence": int(sequence_ids[source_index]),
                    "position": int(positions[source_index]),
                    "norm": float(norms[point_idx]),
                }
            )
        counts = np.bincount(experts, minlength=mlp.engine.config.num_experts)
        layer_metrics[str(layer_idx)] = {
            "points": int(len(embedding)),
            "sampled_tokens": int(len(sample_indices)),
            "points_per_expert": counts.astype(int).tolist(),
            "pca_dimensions": int(pca_dim),
            "tsne_perplexity": float(min(args.perplexity, len(vectors) - 1)),
            "tsne_kl_divergence": float(kl_divergence),
            "full_probe_route_audit": full_probe_audit,
        }
        print(
            f"layer={layer_idx} points={len(embedding)} "
            f"expert_counts={counts.tolist()} kl={kl_divergence:.4f}"
        )

    write_interactive_html(all_rows, args.layers, output, args.model_label)
    points_path.parent.mkdir(parents=True, exist_ok=True)
    with points_path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            mtime=0,
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
                writer.writeheader()
                writer.writerows(all_rows)

    trainable_parameters = int(
        sum(parameter.numel() for parameter in model.parameters())
    )
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    metadata = {
        "title": f"{args.model_label} routed-expert contribution t-SNE",
        "scope": (
            "Exploratory geometry of routed residual contributions; not a "
            "specialization, quality, or routing-superiority claim."
        ),
        "checkpoint": {
            "artifact": args.artifact_label,
            "weights_sha256": sha256_file(checkpoint / "model.safetensors"),
            "config_sha256": sha256_file(checkpoint / "config.json"),
            "trainable_parameters": trainable_parameters,
            "source_token_exposure": args.source_token_exposure,
            "sft_tokens": args.sft_tokens,
        },
        "probe": {
            "name": "PIQA validation",
            "source": "ybisk/piqa",
            "file": probe.name,
            "sha256": sha256_file(probe),
            "records_used": len(records),
            "encoded_tokens_available": len(encoded),
            "chat_template": False,
        },
        "method": {
            "layers_zero_indexed": list(args.layers),
            "sequences": args.sequences,
            "sequence_length": args.sequence_length,
            "tokens_per_layer": args.tokens_per_layer,
            "route_points_per_token": 2,
            "shared_output_included": False,
            "vector_normalization": "per-point L2",
            "pca_max_dimensions": 50,
            "tsne_dimensions": 3,
            "tsne_seed": args.seed,
            "tsne_max_iter": 1500,
            "layer_metrics": layer_metrics,
        },
        "artifacts": {
            "html": output.name,
            "html_sha256": sha256_file(output),
            "points": points_path.name,
            "points_sha256": sha256_file(points_path),
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
