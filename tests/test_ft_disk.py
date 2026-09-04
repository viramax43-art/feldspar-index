import importlib.util
from pathlib import Path

_FT = Path(__file__).resolve().parents[1] / "scripts" / "finetune_xtts.py"
_spec = importlib.util.spec_from_file_location("finetune_xtts", _FT)
assert _spec and _spec.loader
_ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ft)

_lean_payload = _ft._lean_payload
prune_training_junk = _ft.prune_training_junk


def test_prune_training_junk_keeps_lean_best(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    keep = run / "best_model.pth"
    keep.write_bytes(b"lean")
    (run / "best_model_709.pth").write_bytes(b"dup")
    (run / "checkpoint_100.pth").write_bytes(b"ckpt")
    (run / "events.out.tfevents.1").write_bytes(b"tb")
    (run / "config.json").write_text("{}", encoding="utf-8")

    removed = prune_training_junk(tmp_path, preserve={keep})
    assert removed > 0
    assert keep.exists()
    assert (run / "config.json").exists()
    assert not (run / "best_model_709.pth").exists()
    assert not (run / "checkpoint_100.pth").exists()


def test_lean_payload_drops_optimizer() -> None:
    state = {
        "model": {"w": 1},
        "optimizer": {"huge": True},
        "scheduler": {},
        "config": {"lr": 1e-5},
        "step": 10,
        "epoch": 1,
        "date": "today",
        "model_loss": 0.1,
    }
    lean = _lean_payload(state)
    assert "optimizer" not in lean
    assert lean["model"] == {"w": 1}
    assert lean["step"] == 10
