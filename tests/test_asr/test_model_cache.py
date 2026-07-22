import threading
import time

from subforge.core.asr.model_cache import SingleEntryModelCache


def test_single_entry_model_cache_reuses_same_key_and_reloads_new_key():
    cache = SingleEntryModelCache()
    loads = []

    def loader():
        value = object()
        loads.append(value)
        return value

    with cache.acquire(("english", "mps"), loader) as first:
        pass
    with cache.acquire(("english", "mps"), loader) as second:
        pass
    with cache.acquire(("korean", "mps"), loader) as third:
        pass

    assert first is second
    assert third is not first
    assert len(loads) == 2


def test_single_entry_model_cache_serializes_inference_access():
    cache = SingleEntryModelCache()
    active = 0
    peak = 0
    state_lock = threading.Lock()

    def run():
        nonlocal active, peak
        with cache.acquire(("model",), object):
            with state_lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.01)
            with state_lock:
                active -= 1

    threads = [threading.Thread(target=run) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert peak == 1
