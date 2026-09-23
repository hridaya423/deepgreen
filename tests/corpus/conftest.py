import json
import os
import subprocess
import sys

import pytest

from tests.corpus import REPO_ROOT

LANDING_DIR = REPO_ROOT / "data" / "sources" / "openwordle-v1"


def run_repo_tool(script, *args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / script), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


@pytest.fixture(scope="session")
def landed_sources():
    result = run_repo_tool("tools/corpus/fetch.py")
    assert result.returncode == 0, f"fetch failed:\n{result.stdout}\n{result.stderr}"
    return LANDING_DIR


@pytest.fixture(scope="session")
def candidate(tmp_path_factory, landed_sources):
    output = tmp_path_factory.mktemp("candidate-build") / "candidate"
    result = run_repo_tool("tools/corpus/build.py", "--output", str(output))
    assert result.returncode == 0, f"build failed:\n{result.stdout}\n{result.stderr}"
    return output


def read_candidate_json(candidate, name):
    return json.loads((candidate / name).read_text(encoding="utf-8"))


def read_words(candidate, rel):
    return (candidate / rel).read_text(encoding="utf-8").splitlines()
