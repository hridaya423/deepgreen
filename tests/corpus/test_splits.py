import hashlib

from tests.corpus.conftest import read_words

SALT_HEX = "6f70656e776f72646c652d7631"
TIER_SIZES = (512, 1024, 2048, 2095)
FOLD_SIZES = {"train": 1677, "dev": 209, "test": 209}


def _ranked(primary, domain_salt):
    return sorted(primary, key=lambda word: hashlib.sha256(b"tier" + domain_salt + word.encode("utf-8")).hexdigest())


def test_tiers_are_strictly_nested_hash_ranks(candidate):
    primary = read_words(candidate, "answers/primary-2095.txt")
    tier_paths = {size: f"tiers/tier-{size}.txt" for size in TIER_SIZES}
    tiers = {size: read_words(candidate, rel) for size, rel in tier_paths.items()}
    for size in TIER_SIZES:
        assert len(tiers[size]) == size
    assert set(tiers[512]) < set(tiers[1024])
    assert set(tiers[1024]) < set(tiers[2048])
    assert set(tiers[2048]) < set(primary)
    assert set(tiers[2095]) == set(primary)
    ranked = _ranked(primary, b"openwordle-v1")
    for size in TIER_SIZES:
        assert tiers[size] == ranked[:size], f"tier-{size} not the hash-ranked prefix"


def test_ah_folds_partition_primary_answers_exactly(candidate):
    primary = set(read_words(candidate, "answers/primary-2095.txt"))
    folds = {
        name: read_words(candidate, f"splits/ah-{name}.txt")
        for name in ("train", "dev", "test")
    }
    for name, words in folds.items():
        assert len(words) == FOLD_SIZES[name], f"{name}: {len(words)}"
    train, dev, test = (set(folds[name]) for name in ("train", "dev", "test"))
    assert not (train & dev) and not (train & test) and not (dev & test)
    assert train | dev | test == primary


def test_fold_assignment_reproduces_from_preregistered_bytes(candidate):
    import sys

    from tests.corpus import REPO_ROOT

    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    from deep_green.corpus import assign_answer_held_out_folds

    primary = read_words(candidate, "answers/primary-2095.txt")
    salt = bytes.fromhex(SALT_HEX)
    assignment = assign_answer_held_out_folds(primary, salt)
    for name in ("train", "dev", "test"):
        produced = sorted(getattr(assignment, name))
        stored = read_words(candidate, f"splits/ah-{name}.txt")
        assert produced == stored


def test_divergent_salt_interpretation_fails_verification(tmp_path, candidate):
    from tests.corpus.conftest import run_repo_tool

    verify = run_repo_tool("tools/corpus/verify_candidate.py", str(candidate))
    assert verify.returncode == 0
    tampered = tmp_path / "candidate-copy"
    tampered.mkdir()
    for item in candidate.rglob("*"):
        rel = item.relative_to(candidate)
        target = tampered / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            data = bytearray(item.read_bytes())
            data[-1] ^= 0x20
            target.write_bytes(bytes(data))
    verify_bad = run_repo_tool("tools/corpus/verify_candidate.py", str(tampered))
    assert verify_bad.returncode != 0
