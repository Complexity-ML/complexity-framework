from complexity.training.tensorboard import TensorBoardMetricWriter


class _FakeSummaryWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.flushed = False
        self.closed = False

    def add_scalar(self, name: str, value: float, step: int) -> None:
        self.scalars.append((name, value, step))

    def flush(self) -> None:
        self.flushed = True

    def close(self) -> None:
        self.closed = True


def test_tensorboard_metric_writer_records_only_finite_numeric_scalars(tmp_path):
    metrics = TensorBoardMetricWriter(tmp_path, enabled=False)
    fake = _FakeSummaryWriter()
    metrics.writer = fake

    metrics.add_scalars(
        "/validation/o2m/",
        {
            "map50": 0.25,
            "examples": 5000,
            "official": True,
            "backend": "pycocotools",
            "nan": float("nan"),
            "inf": float("inf"),
        },
        120,
    )

    assert fake.scalars == [
        ("validation/o2m/map50", 0.25, 120),
        ("validation/o2m/examples", 5000.0, 120),
    ]


def test_tensorboard_metric_writer_flushes_and_closes(tmp_path):
    metrics = TensorBoardMetricWriter(tmp_path, enabled=False)
    fake = _FakeSummaryWriter()
    metrics.writer = fake

    metrics.flush()
    metrics.close()

    assert fake.flushed is True
    assert fake.closed is True


def test_tensorboard_metric_writer_is_noop_when_disabled(tmp_path):
    metrics = TensorBoardMetricWriter(tmp_path, enabled=False)

    metrics.add_scalars("train", {"loss": 1.0}, 1)
    metrics.flush()
    metrics.close()

    assert metrics.enabled is False
