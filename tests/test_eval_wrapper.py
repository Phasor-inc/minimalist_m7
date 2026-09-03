"""Only tests the parsing half without invoking the real (slow, GPU-bound)
eval pipeline -- the pipeline itself is exercised for real in Step 6/7
below, not under pytest."""
import json
from code.eval_wrapper import run_eval


def test_run_eval_raises_when_no_summary_produced(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    try:
        run_eval(checkpoint_path="/fake", output_dir=str(tmp_path))
        assert False, "expected FileNotFoundError when no summary.json exists"
    except FileNotFoundError:
        pass
