"""Each training stage must validate effects before counting work."""

import hashlib

import pytest

from scripts import pgo_train as P


@pytest.mark.parametrize(
    "stage,factory",
    [
        ("first-cache", "first"),
        ("tf-live", "live"),
        ("append", "append"),
        ("streams", "stream"),
        ("range", "append"),
    ],
)
def test_training_rejects_noop_factory(monkeypatch, stage, factory):
    from test import bench_workloads as W, bench_first_cache as F, bench_tf_live as T

    monkeypatch.setattr(P, "TRAIN_PASSES", 1)
    module, name = (
        (F, "build")
        if factory == "first"
        else (T, "build")
        if factory == "live"
        else (W, factory)
    )
    original = getattr(module, name)

    def noop(*a, **k):
        _, setup = original(*a, **k)
        return (lambda df: df), setup

    monkeypatch.setattr(module, name, noop)
    with pytest.raises(AssertionError):
        P.train(stage)


def test_training_boundaries_rejects_wrong_values(monkeypatch):
    from test import bench_workloads as W
    import numpy as np

    monkeypatch.setattr(P, "TRAIN_PASSES", 1)
    original = W.interop

    def incorrect(n):
        factories = original(n)
        factories["dlpack_export"] = lambda: np.zeros(n)
        return factories

    monkeypatch.setattr(W, "interop", incorrect)
    with pytest.raises(AssertionError):
        P.train("boundaries")


@pytest.mark.parametrize("stage", list(P.STAGES))
def test_training_effect_feeds_digest(monkeypatch, stage):
    monkeypatch.setattr(P, "TRAIN_PASSES", 1)
    result = P.train(stage)
    assert result["result_sha256"] != hashlib.sha256().hexdigest()


@pytest.mark.parametrize("stage", ["batch", "exceptional"])
@pytest.mark.parametrize("damage", ["values", "mask"])
def test_training_rejects_wrong_batch_results(monkeypatch, stage, damage):
    import numpy as np
    import volas

    original = volas.DataFrame

    class WrongResult:
        def __init__(self, *args, **kw):
            self.frame = original(*args, **kw)

        def __getitem__(self, key):
            return self.frame[key]

        def exec(self, *args, **kw):
            values = self.frame.exec(*args, **kw).copy()
            values[-1] = np.nan if damage == "mask" else 12345678.0
            return values

    monkeypatch.setattr(volas, "DataFrame", WrongResult)
    monkeypatch.setattr(P, "TRAIN_PASSES", 1)
    with pytest.raises(AssertionError):
        P.train(stage)
