"""The suite must not read production configuration.

Every strategy knob is resolved once at import time from the environment, and
the hourly workflow exported all of them to *every* step -- including the unit
tests. Setting the repo variable EXECUTION_MODE=maker for production therefore
rewrote the round trip from 0.30% to 0.08% underneath the assertions and turned
12 tests red on run #50. Nothing was wrong with the code. The tests had been
measuring the environment.

These pin the invariant: a repo variable can change what the bot does, never
what the tests conclude.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import STRATEGY_ENV_KNOBS

ROOT = Path(__file__).resolve().parents[1]


def test_conftest_clears_every_knob_the_workflow_exports():
    """If a knob is added to the workflow but not here, it leaks back in."""
    workflow = (ROOT / ".github" / "workflows" / "hourly_pipeline.yml").read_text()
    risk = (ROOT / ".github" / "workflows" / "risk_monitor.yml").read_text()

    exported = set()
    for text in (workflow, risk):
        for line in text.splitlines():
            line = line.strip()
            if "${{ vars." in line and ":" in line:
                exported.add(line.split(":", 1)[0].strip())

    # Deployment/plumbing values are supposed to reach the process.
    exported -= {"SUPABASE_URL", "SUPABASE_KEY", "EXCHANGE_ID"}
    missing = sorted(exported - set(STRATEGY_ENV_KNOBS))
    assert not missing, (
        f"the workflow exports {missing} but tests/conftest.py does not clear them, "
        "so a repo variable can change what the tests conclude"
    )


def test_no_strategy_knob_survives_into_the_test_process():
    for knob in STRATEGY_ENV_KNOBS:
        assert knob not in os.environ, f"{knob} leaked into the test environment"


def test_the_economics_the_suite_asserts_on_are_the_documented_defaults():
    from src.strategy.economics import EXECUTION_MODE, round_trip_cost_pct

    assert EXECUTION_MODE == "taker"
    assert round_trip_cost_pct() == pytest.approx(0.003)


@pytest.mark.parametrize("hostile", [
    {"EXECUTION_MODE": "maker"},
    {"TAKER_FEE": "0.05", "SLIPPAGE_BPS": "999"},
    {"MIN_EDGE_MARGIN": "0", "MIN_HORIZON_FRACTION": "0.99"},
])
def test_a_hostile_environment_cannot_change_the_verdict(hostile):
    """Runs a representative slice in a subprocess with the knob exported.

    Subprocess rather than monkeypatch, because the leak happens at *import*
    time -- which is exactly what an in-process fixture cannot reproduce.
    """
    env = {**os.environ, **hostile}
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header",
         "tests/test_economics.py", "tests/test_horizon.py"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert r.returncode == 0, f"{hostile} changed the outcome:\n{r.stdout[-2500:]}"
