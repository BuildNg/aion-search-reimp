"""The full run must refuse to start (and must not create a results/<run_id>
directory) unless a passing, current preflight report exists at the path
config.run.preflight_report names.

scripts/run_phase6_probes_cluster.py is cluster-only (it streams live data
and loads live model checkpoints in its non-gating code paths), so this
suite only exercises the two pure, file-IO-only gate functions -- no
network, no model weights, matching this repo's existing convention of not
unit-testing cluster entrypoints end to end (see architecture.md's compute
boundary).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from spec_probes.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_phase6_probes_cluster.py"
CONFIG_PATH = ROOT / "configs" / "phase6_probes.yaml"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_phase6_probes_cluster", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script_module():
    return _load_script_module()


def _config(tmp_path: Path, report_name: str = "preflight/phase6_probes_v1.json"):
    config = load_config(CONFIG_PATH)
    config["run"] = dict(config["run"])
    config["run"]["preflight_report"] = str(tmp_path / report_name)
    return config


def _passing_report(script_module, config):
    contract = script_module._preflight_contract(config)
    return {
        "status": "pass",
        "contract": contract,
        "contract_fingerprint": script_module._canonical_fingerprint(contract),
        "checks": {},
    }


def test_require_passing_preflight_raises_when_report_missing(tmp_path, script_module) -> None:
    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match="No preflight report"):
        script_module._require_passing_preflight(config)


def test_preflight_sample_covers_pca_and_neural_batch_requirements(tmp_path, script_module) -> None:
    config = _config(tmp_path)
    assert script_module._preflight_sample_size(config) == 128


def test_require_passing_preflight_raises_when_report_failed(tmp_path, script_module) -> None:
    config = _config(tmp_path)
    report_path = Path(config["run"]["preflight_report"])
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps({"status": "fail"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not report status 'pass'"):
        script_module._require_passing_preflight(config)


def test_require_passing_preflight_accepts_a_passing_report(tmp_path, script_module) -> None:
    config = _config(tmp_path)
    report_path = Path(config["run"]["preflight_report"])
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(_passing_report(script_module, config)), encoding="utf-8")
    script_module._require_passing_preflight(config)  # must not raise


def test_require_passing_preflight_rejects_a_stale_contract(tmp_path, script_module) -> None:
    config = _config(tmp_path)
    report_path = Path(config["run"]["preflight_report"])
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(_passing_report(script_module, config)), encoding="utf-8")
    config["run"]["seed"] += 1
    with pytest.raises(RuntimeError, match="different config, code version, package environment, or device"):
        script_module._require_passing_preflight(config)


def test_write_preflight_report_never_touches_results_directory(tmp_path, script_module) -> None:
    config = _config(tmp_path)
    report = {"status": "pass", "checks": {}}
    report_path = script_module._write_preflight_report(config, report)
    assert report_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    # No results/<run_id> directory anywhere under tmp_path.
    assert not any(path.name == "results" for path in tmp_path.rglob("*") if path.is_dir())


def test_write_preflight_report_overwrites_a_previous_attempt(tmp_path, script_module) -> None:
    config = _config(tmp_path)
    script_module._write_preflight_report(config, {"status": "fail", "checks": {}})
    report_path = script_module._write_preflight_report(config, {"status": "pass", "checks": {}})
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "pass"
