from complexity.training.o200k.data import FineWebDataset


def test_fineweb_train_and_eval_document_partitions_are_disjoint():
    train = FineWebDataset.__new__(FineWebDataset)
    train.split = "train"
    train.eval_stride = 1000
    eval_set = FineWebDataset.__new__(FineWebDataset)
    eval_set.split = "eval"
    eval_set.eval_stride = 1000

    for index in range(10_000):
        assert not (
            train._uses_document(index) and eval_set._uses_document(index)
        )

    assert sum(train._uses_document(i) for i in range(10_000)) == 9_990
    assert sum(eval_set._uses_document(i) for i in range(10_000)) == 10


def test_fineweb_partition_rejects_unknown_split():
    dataset = FineWebDataset.__new__(FineWebDataset)
    dataset.split = "test"
    dataset.eval_stride = 1000

    try:
        dataset._uses_document(0)
    except ValueError as error:
        assert "split" in str(error)
    else:
        raise AssertionError("unknown split should fail")
