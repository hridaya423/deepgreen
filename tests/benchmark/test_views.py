"""Comparison-keyed views: every tier/track/mode cell exists, digests bind, bundles refuse outsiders."""

from __future__ import annotations

import json

import pytest

from tests.benchmark import repo_path
from tests.benchmark.test_candidate_manifest import CANDIDATE_DIR

PROTOCOL_PATH = repo_path("benchmarks", "openwordle-v1", "protocol.toml")

TIER_SIZES = (512, 1024, 2048, 2095)
TRACKS = ("full-compiled", "answer-held-out-ah-catalog-exposed")
MODES = ("normal", "hard")
SEALED_TEST_FOLD_SIZE = 209

RELEASE_CELL_IDS = (
    "openwordle-v1.uniform.normal.full-compiled",
    "openwordle-v1.uniform.hard.full-compiled",
    "openwordle-v1-ah-test-209.uniform.normal.answer-held-out-ah-catalog-exposed",
    "openwordle-v1-ah-test-209.uniform.hard.answer-held-out-ah-catalog-exposed",
)


@pytest.fixture(scope="module")
def views():
    from deep_green.benchmark import build_views, load_protocol

    return build_views(load_protocol(PROTOCOL_PATH), CANDIDATE_DIR)


def view_mapping(record):
    return record if isinstance(record, dict) else vars(record)


def views_by_id(views) -> dict[str, dict]:
    if isinstance(views, dict):
        return {mapping["view_id"]: mapping for mapping in (view_mapping(v) for v in views.values())}
    return {view_mapping(record)["view_id"]: view_mapping(record) for record in views}


def test_release_cells_exist_verbatim(views):
    ids = set(views_by_id(views))
    for cell in RELEASE_CELL_IDS:
        assert cell in ids


def test_every_tier_track_mode_combination_has_a_view(views):
    ids = set(views_by_id(views))
    for size in TIER_SIZES:
        for track in TRACKS:
            answer_base = (
                "openwordle-v1" if track == "full-compiled" else f"openwordle-v1-ah-test-{SEALED_TEST_FOLD_SIZE}"
            )
            for mode in MODES:
                expected = f"{answer_base}.uniform.{mode}.{track}"
                assert expected in ids, f"missing view {expected}"


def test_track_identifiers_are_lowercase_with_the_contract_spelling(views):
    ids = set(views_by_id(views))
    assert any(".answer-held-out-ah-catalog-exposed" in vid for vid in ids)
    for track in TRACKS:
        assert track == track.lower()
        assert "AH" not in track and "Ah" not in track


def test_views_carry_comparison_keys_and_digest_bindings(views):
    candidate_manifest = json.loads((CANDIDATE_DIR / "manifest.json").read_text(encoding="utf-8"))
    accepted_digest = candidate_manifest["files"]["words/accepted-guesses.txt"]

    by_id = views_by_id(views)
    full_view = by_id[RELEASE_CELL_IDS[0]]
    hard_view = by_id[RELEASE_CELL_IDS[1]]
    ah_normal = by_id[RELEASE_CELL_IDS[2]]
    ah_hard = by_id[RELEASE_CELL_IDS[3]]

    for mapping in (full_view, hard_view, ah_normal, ah_hard):
        assert mapping["action_space_sha256"] == accepted_digest
        assert mapping["prior_id"] == "uniform-v1"
        assert mapping["horizon_turns"] == 6
        assert mapping["tie_order"] == "lowest_canonical_action_index_wins"

    assert full_view["answers_file"] == "tiers/tier-2095.txt"
    assert hard_view["answers_file"] == "tiers/tier-2095.txt"
    assert ah_normal["answers_file"] == "splits/ah-test.txt"
    assert full_view["legality_profile_id"] is None
    assert hard_view["legality_profile_id"] == "nyt-standard-hard"
    assert ah_normal["legality_profile_id"] is None
    assert (
        hard_view["rules_sha256"]
        == ah_normal["rules_sha256"]
        == full_view["rules_sha256"]
    )


def test_ah_views_bind_only_the_sealed_test_fold(views):
    by_id = views_by_id(views)
    ah_views = [mapping for mapping in by_id.values() if "ah-test" in mapping.get("answers_file", "")]
    assert len(ah_views) >= 2
    for mapping in ah_views:
        assert mapping["answers_file"] == "splits/ah-test.txt"
        assert f"-{SEALED_TEST_FOLD_SIZE}." in mapping["view_id"]
    assert not [mapping for mapping in by_id.values() if mapping.get("answers_file") in ("splits/ah-dev.txt", "splits/ah-train.txt")]


def test_load_frozen_benchmark_rejects_a_cross_bundle_view(tmp_path):
    from deep_green.benchmark import (
        BundleError,
        build_views,
        load_frozen_benchmark,
        load_protocol,
        write_bundle_files,
    )

    protocol = load_protocol(PROTOCOL_PATH)
    views = build_views(protocol, CANDIDATE_DIR)
    by_id = views_by_id(views)

    bundle_dir = tmp_path / "bundle-full"
    write_bundle_files(bundle_dir, protocol, CANDIDATE_DIR, [by_id[RELEASE_CELL_IDS[0]]])
    foreign = tmp_path / "foreign-view.json"
    foreign.write_text(json.dumps(by_id[RELEASE_CELL_IDS[2]]), encoding="utf-8")

    with pytest.raises(BundleError, match="not part of this bundle"):
        load_frozen_benchmark(bundle_dir / "manifest.json", extra_view_paths=[foreign])


def test_load_frozen_benchmark_rejects_digest_drift(tmp_path):
    from deep_green.benchmark import (
        BundleError,
        build_views,
        load_frozen_benchmark,
        load_protocol,
        write_bundle_files,
    )

    protocol = load_protocol(PROTOCOL_PATH)
    views = build_views(protocol, CANDIDATE_DIR)
    record = views_by_id(views)[RELEASE_CELL_IDS[0]]
    drifted = dict(record)
    drifted["answers_sha256"] = "0" * 64

    bundle_dir = tmp_path / "bundle-drift"
    write_bundle_files(bundle_dir, protocol, CANDIDATE_DIR, [drifted])
    with pytest.raises(BundleError, match="digest|sha256"):
        load_frozen_benchmark(bundle_dir / "manifest.json")
