from inter_active_learning import compute


def test_compute():
    assert compute(["a", "bc", "abc"]) == "abc"
