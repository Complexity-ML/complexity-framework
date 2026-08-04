from complexity.training.o200k.data import FineWebDataset


class _Tokenizer:
    eos_token_id = None

    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return [int(value) for value in text.split()]


def test_fineweb_train_and_eval_document_partitions_are_disjoint():
    train = FineWebDataset.__new__(FineWebDataset)
    train.split = "train"
    train.eval_stride = 20
    eval_set = FineWebDataset.__new__(FineWebDataset)
    eval_set.split = "eval"
    eval_set.eval_stride = 20

    for index in range(10_000):
        assert not (
            train._uses_document(index) and eval_set._uses_document(index)
        )

    assert sum(train._uses_document(i) for i in range(10_000)) == 9_500
    assert sum(eval_set._uses_document(i) for i in range(10_000)) == 500


def test_fineweb_partition_rejects_unknown_split():
    dataset = FineWebDataset.__new__(FineWebDataset)
    dataset.split = "test"
    dataset.eval_stride = 20

    try:
        dataset._uses_document(0)
    except ValueError as error:
        assert "split" in str(error)
    else:
        raise AssertionError("unknown split should fail")


def test_fineweb_resume_skips_completed_sequences_per_rank():
    dataset = FineWebDataset.__new__(FineWebDataset)
    dataset.tokenizer = _Tokenizer()
    dataset.seq_len = 3
    dataset.rank = 0
    dataset.world_size = 1
    dataset.split = "train"
    dataset.eval_stride = 100
    dataset.start_sequence = 2
    dataset._examples = lambda: iter([(1, {"text": "0 1 2 3 4 5 6 7 8 9"})])

    first = next(iter(dataset))

    assert first["input_ids"].tolist() == [6, 7, 8]
    assert first["labels"].tolist() == [7, 8, 9]
