from subforge.core.utils.download_progress import DownloadEstimator


def test_warmup_and_constant_speed():
    estimator = DownloadEstimator()
    for t in range(4):
        assert estimator.update(t * 100, 10000, t * 100, t)["eta_seconds"] is None
    data = estimator.update(400, 10000, 400, 4)
    assert data["speed_bps"] == 100
    assert data["eta_seconds"] == 96


def test_cached_bytes_never_inflate_speed():
    estimator = DownloadEstimator()
    estimator.update(9000, 10000, 0, 0)
    data = estimator.update(9400, 10000, 400, 4)
    assert data["speed_bps"] == 100
    assert data["eta_seconds"] == 6


def test_unknown_size_reports_speed_but_no_percentage_or_eta():
    estimator = DownloadEstimator()
    estimator.update(0, None, 0, 0)
    data = estimator.update(400, None, 400, 4)
    assert data["progress"] is None
    assert data["eta_seconds"] is None
    assert data["speed_bps"] == 100


def test_stall_and_recovery_require_new_samples():
    estimator = DownloadEstimator()
    for t in range(6):
        estimator.update(t * 100, 10000, t * 100, t)
    stalled = estimator.update(500, 10000, 500, 15)
    assert stalled["phase"] == "waiting"
    assert stalled["eta_seconds"] is None
    recovered = estimator.update(600, 10000, 600, 16)
    assert recovered["eta_seconds"] is None
    assert estimator.update(1000, 10000, 1000, 20)["speed_bps"] == 100


def test_retry_rollback_and_completion_do_not_emit_zero_eta():
    estimator = DownloadEstimator()
    estimator.update(500, 1000, 0, 0)
    estimator.update(900, 1000, 400, 4)
    data = estimator.update(500, 1000, 400, 5)
    assert data["phase"] == "retrying"
    assert data["eta_seconds"] is None
    data = estimator.update(1000, 1000, 900, 10)
    assert data["progress"] == 99
    assert data["eta_seconds"] is None


def test_slowdown_adapts_without_single_sample_jump():
    estimator = DownloadEstimator()
    for t in range(21):
        estimator.update(t * 1000, 1000000, t * 1000, t)
    data = estimator.update(20100, 1000000, 20100, 21)
    assert 100 < data["speed_bps"] < 1000
    for t in range(22, 65):
        data = estimator.update(20000 + (t - 20) * 100, 1000000, 20000 + (t - 20) * 100, t)
    assert 95 <= data["speed_bps"] <= 105
