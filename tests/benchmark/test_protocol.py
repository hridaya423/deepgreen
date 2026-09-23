"""Protocol pinning: every comparison dimension recorded, parsing fails closed."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import pytest

from tests.benchmark import repo_path

PROTOCOL_PATH = repo_path("benchmarks", "openwordle-v1", "protocol.toml")
RULES_PATH = repo_path("benchmarks", "rules", "openwordle-rules-v1.json")
PRIOR_PATH = repo_path("benchmarks", "openwordle-v1", "priors", "uniform-v1.json")

REQUIRED_SECTIONS = (
    "identity",
    "sources",
    "corpus",
    "action_space",
    "tiers",
    "ah_split",
    "rules",
    "prior",
    "harness",
    "model",
    "campaign",
    "clean_room",
)

REQUIRED_FIELDS = {
    "sources": ("scowl_v1_commit", "scowl_v1_sha256", "wordnet_3_version", "wordnet_3_sha256"),
    "corpus": (
        "salt_hex",
        "accepted_guesses_count",
        "primary_answer_count",
        "broad_answer_count",
    ),
    "action_space": ("kind", "count"),
    "tiers": ("sizes",),
    "ah_split": ("counts",),
    "rules": ("document", "ruleset_id", "legality_profiles", "sha256"),
    "prior": ("document", "prior_id", "distribution", "support", "sha256"),
    "harness": ("horizon_turns", "loss_scoring", "objective", "tie_order"),
    "model": ("ladder_widths",),
    "campaign": ("seeds", "local_time_limit_hours"),
    "clean_room": ("build_identity",),
}


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{key} = {_toml_value(item)}" for key, item in value.items()) + " }"
    return str(value)


def write_mutated_protocol(tmp_path: Path, mutate) -> Path:
    raw = tomllib.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    lines: list[str] = []
    for section, fields in raw.items():
        lines.append(f"[{section}]")
        for key, value in fields.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    lines.append(f"{key}.{sub_key} = {_toml_value(sub_value)}")
            else:
                lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    path = tmp_path / "protocol-mutant.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_every_pinned_dimension_is_recorded():
    raw = tomllib.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert set(raw) == set(REQUIRED_SECTIONS)
    for section in REQUIRED_SECTIONS:
        for field in REQUIRED_FIELDS.get(section, ()):
            assert field in raw[section], f"protocol.{section}.{field} is not pinned"


def test_load_protocol_returns_typed_pins():
    from deep_green.benchmark import load_protocol

    protocol = load_protocol(PROTOCOL_PATH)
    assert protocol.identity.benchmark_id == "openwordle-v1"
    assert protocol.action_space.count == 5022 == protocol.corpus.accepted_guesses_count
    assert protocol.corpus.primary_answer_count == 2095
    assert protocol.corpus.broad_answer_count == 3048
    assert protocol.tiers.sizes == (512, 1024, 2048, 2095)
    assert protocol.ah_split.counts == {"train": 1677, "dev": 209, "test": 209}
    assert protocol.harness.horizon_turns == 6


def test_source_pins_match_the_ledger_documents():
    from corpus_ledger import load_corpus_spec, load_source_specs

    from deep_green.benchmark import load_protocol

    protocol = load_protocol(PROTOCOL_PATH)
    sources = load_source_specs(repo_path("benchmarks", "openwordle-v1", "sources.toml"))
    spec = load_corpus_spec(repo_path("benchmarks", "openwordle-v1", "corpus.toml"))
    assert protocol.sources.scowl_v1_commit == sources["scowl_v1"].commit
    assert protocol.sources.scowl_v1_sha256 == sources["scowl_v1"].sha256
    assert protocol.sources.wordnet_3_sha256 == sources["wordnet_3"].sha256
    assert protocol.corpus.salt_hex == spec.splits["ah"].salt_hex
    assert bytes.fromhex(protocol.corpus.salt_hex) == b"openwordle-v1"


def test_rules_and_prior_digests_bind_the_actual_documents():
    from deep_green.benchmark import load_protocol

    protocol = load_protocol(PROTOCOL_PATH)
    rules_bytes = RULES_PATH.read_bytes()
    prior_bytes = PRIOR_PATH.read_bytes()
    assert hashlib.sha256(rules_bytes).hexdigest() == protocol.rules.sha256
    assert hashlib.sha256(prior_bytes).hexdigest() == protocol.prior.sha256
    rules_doc = json.loads(rules_bytes)
    prior_doc = json.loads(prior_bytes)
    assert rules_doc["legality_profiles"]["nyt-standard-hard"]["preserve_greens"] is True
    assert rules_doc["legality_profiles"]["normal"]["candidate_consistency_minus_u"] is False
    assert prior_doc["prior_id"] == "uniform-v1"
    assert prior_doc["distribution"] == "uniform"


def test_corpus_manifest_pin_matches_the_candidate_manifest_bytes():
    import hashlib

    manifest_path = repo_path(
        "artifacts", "corpus", "openwordle-v1-candidate", "manifest.json"
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    from deep_green.benchmark import load_protocol

    protocol = load_protocol(PROTOCOL_PATH)
    assert protocol.identity.corpus_manifest_sha256 == manifest_sha
    accepted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (
        protocol.identity.corpus_manifest_sha256
        != accepted_manifest["files"]["words/accepted-guesses.txt"]
    )


def test_model_ladder_seeds_and_campaign_limits_are_pinned():
    from deep_green.benchmark import load_protocol

    protocol = load_protocol(PROTOCOL_PATH)
    assert protocol.model.ladder_widths == (64, 96, 128, 192, 256, 320, 384)
    assert protocol.campaign.seeds == (17, 29, 43, 71, 113)
    assert protocol.campaign.local_time_limit_hours == 72
    assert protocol.clean_room.build_identity == "pinned-linux-container-scowl-build"
    assert protocol.harness.loss_scoring == "turns_to_solve_unsolved_scores_seven"
    assert protocol.harness.objective.startswith(
        "lexicographic_minimize_losses_then_maximize_score7_total"
    )
    assert protocol.harness.tie_order == "lowest_canonical_action_index_wins"


def test_hard_mode_is_identified_by_legality_profile_and_rules_hash_together():
    from deep_green.benchmark import load_protocol

    protocol = load_protocol(PROTOCOL_PATH)
    hard_mode = protocol.modes["hard"]
    assert hard_mode.legality_profile_id == "nyt-standard-hard"
    assert hard_mode.ruleset_id == protocol.rules.ruleset_id
    assert hard_mode.rules_sha256 == protocol.rules.sha256
    normal_mode = protocol.modes["normal"]
    assert normal_mode.legality_profile_id is None
    assert normal_mode.rules_sha256 == protocol.rules.sha256


@pytest.mark.parametrize("bad_mode", ["hard_mode", "HARD", "nyt-hard", "expert"])
def test_mode_values_outside_normal_and_hard_fail_parsing(tmp_path, bad_mode):
    from deep_green.benchmark import ProtocolError, load_protocol

    def poison(raw):
        raw["rules"]["legality_profiles"] = ["normal", bad_mode]

    mutant = write_mutated_protocol(tmp_path, poison)
    with pytest.raises(ProtocolError, match="mode"):
        load_protocol(mutant)


def test_removing_a_required_field_fails_parsing(tmp_path):
    from deep_green.benchmark import ProtocolError, load_protocol

    def cut(raw):
        del raw["campaign"]["local_time_limit_hours"]

    mutant = write_mutated_protocol(tmp_path, cut)
    with pytest.raises(ProtocolError, match="local_time_limit_hours"):
        load_protocol(mutant)


def test_unknown_field_fails_parsing(tmp_path):
    from deep_green.benchmark import ProtocolError, load_protocol

    def add(raw):
        raw["harness"]["entropy_bonus"] = 0.01

    mutant = write_mutated_protocol(tmp_path, add)
    with pytest.raises(ProtocolError, match="entropy_bonus|unknown"):
        load_protocol(mutant)


def test_unknown_section_fails_parsing(tmp_path):
    from deep_green.benchmark import ProtocolError, load_protocol

    def add(raw):
        raw["rewards"] = {"shaping": "none"}

    mutant = write_mutated_protocol(tmp_path, add)
    with pytest.raises(ProtocolError, match="rewards|unknown"):
        load_protocol(mutant)


def test_hard_mode_resolvable_by_name_alone_is_refused(tmp_path):
    from deep_green.benchmark import ProtocolError, load_protocol

    def strip_profile(raw):
        del raw["rules"]["legality_profiles"]

    mutant = write_mutated_protocol(tmp_path, strip_profile)
    with pytest.raises(ProtocolError, match="legality"):
        load_protocol(mutant)


def test_drifted_rules_digest_fails_parsing(tmp_path):
    from deep_green.benchmark import ProtocolError, load_protocol

    def drift(raw):
        raw["rules"]["sha256"] = "0" * 64

    mutant = write_mutated_protocol(tmp_path, drift)
    with pytest.raises(ProtocolError, match="sha256|digest"):
        load_protocol(mutant)


def test_legacy_objective_identifier_orders_lower_score_as_better():
    from deep_green.benchmark import PINNED_OBJECTIVE, objective_order_key
    better = objective_order_key(PINNED_OBJECTIVE, losses=0, score7_total=100, worst_score=4)
    worse = objective_order_key(PINNED_OBJECTIVE, losses=0, score7_total=101, worst_score=4)
    assert better < worse


def test_unknown_objective_identifier_fails_closed():
    from deep_green.benchmark import ProtocolError, objective_order_key
    with pytest.raises(ProtocolError, match="unknown objective"):
        objective_order_key("minimize_everything", losses=0, score7_total=0, worst_score=0)
