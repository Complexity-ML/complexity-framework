import torch
import yaml
from pathlib import Path

from mini_wrv import ModelConfig, TinyLanguageModel
from mini_wrv.attention import ContextualWRVAttention, GroupedQueryAttention
from mini_wrv.data import TiktokenO200k


def small_config(attention_type: str) -> ModelConfig:
    return ModelConfig(
        vocab_size=97,
        hidden_size=32,
        num_layers=2,
        num_read_heads=4,
        num_write_heads=2,
        intermediate_size=48,
        lexical_object_rank=8,
        micro_num_experts=2,
        micro_expert_width=4,
        max_sequence_length=64,
        attention_type=attention_type,
        lexical_write_residual=False,
    )


def test_paper_parameter_counts_are_exact() -> None:
    gqa = TinyLanguageModel(ModelConfig.paper(attention_type="gqa"))
    wrv = TinyLanguageModel(ModelConfig.paper(attention_type="wrv"))
    assert gqa.trainable_parameter_count() == 98_179_844
    assert wrv.trainable_parameter_count() == 98_195_204


def test_paper_yaml_configs_rebuild_exact_models() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {"gqa_seed42.yaml": 98_179_844, "wrv_seed42.yaml": 98_195_204}
    for name, parameter_count in expected.items():
        raw = yaml.safe_load((root / "configs" / name).read_text())
        model = TinyLanguageModel(ModelConfig(**raw["model"]))
        assert model.trainable_parameter_count() == parameter_count


def test_o200k_adapter_matches_reported_vocabulary() -> None:
    assert TiktokenO200k().encoding.n_vocab == 200_019


def test_attention_types_are_explicit() -> None:
    assert isinstance(TinyLanguageModel(small_config("gqa")).blocks[0].attention, GroupedQueryAttention)
    assert isinstance(TinyLanguageModel(small_config("wrv")).blocks[0].attention, ContextualWRVAttention)


def test_wrv_lexical_off_is_token_id_invariant() -> None:
    torch.manual_seed(7)
    module = ContextualWRVAttention(small_config("wrv")).eval()
    hidden = torch.randn(2, 7, 32)
    ids_a = torch.randint(0, 97, (2, 7))
    ids_b = torch.randint(0, 97, (2, 7))
    with torch.inference_mode():
        out_a, _ = module(hidden, token_ids=ids_a)
        out_b, _ = module(hidden, token_ids=ids_b)
    torch.testing.assert_close(out_a, out_b)


def test_full_and_incremental_logits_match() -> None:
    for attention_type in ("gqa", "wrv"):
        torch.manual_seed(11)
        model = TinyLanguageModel(small_config(attention_type)).eval()
        ids = torch.randint(0, 97, (1, 10))
        with torch.inference_mode():
            full = model(ids)["logits"]
            cache = None
            pieces = []
            for position in range(ids.shape[1]):
                output = model(ids[:, position : position + 1], past_key_values=cache, use_cache=True)
                pieces.append(output["logits"])
                cache = output["past_key_values"]
        torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=2e-5, rtol=2e-5)


def test_one_training_step_is_finite() -> None:
    torch.manual_seed(13)
    model = TinyLanguageModel(small_config("wrv"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, 97, (2, 12))
    loss = model.loss(tokens)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
