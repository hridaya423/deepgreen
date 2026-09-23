"""Firewall tests: the strict IPC allowlist rejects every leak class."""

from __future__ import annotations

import json

import pytest

from deep_green.protocol import (
    PROTOCOL_SCHEMA,
    PolicyRequest,
    ProtocolError,
    parse_request,
    serialize_request,
)
from deep_green.rules import ActionSpace

WORDS = ("audio", "crane", "pouty", "slate")
ACTION_DIGEST = ActionSpace.parse(WORDS).digest


def _request(**overrides):
    fields = {
        "mode": "normal",
        "turn": 1,
        "played_rows": (),
        "action_space_sha256": ACTION_DIGEST,
        "legality_mask": (True, True, True, True),
    }
    fields.update(overrides)
    return PolicyRequest(**fields)


def _base_line() -> str:
    request = _request()
    return serialize_request(request, action_words=WORDS)


def test_round_trip_preserves_public_fields():
    raw = parse_request(_base_line())
    assert raw["schema"] == PROTOCOL_SCHEMA
    assert raw["mode"] == "normal"
    assert raw["turn"] == 1


def test_serializer_rejects_mask_length_mismatch():
    with pytest.raises(ProtocolError):
        serialize_request(_request(legality_mask=(True,)), action_words=WORDS)


@pytest.mark.parametrize(
    "forbidden_field,payload",
    [
        ("answer_id", "slate"),
        ("candidate_count", 12972),
        ("posterior", [0.1, 0.2]),
        ("entropy", 3.25),
        ("diagnostics", {"x": 1}),
        ("oracle_value", {"losses": 0}),
        ("secret_target", "pouty"),
        ("daily_id", 552),
        ("prior_weight", 0.5),
        ("extras", {}),
        ("fold_id", "train"),
        ("exposure_track", "full-compiled"),
    ],
)
def test_forbidden_field_classes_fail_at_parser(forbidden_field, payload):
    body = json.loads(_base_line())
    body[forbidden_field] = payload
    with pytest.raises(ProtocolError) as excinfo:
        parse_request(json.dumps(body))
    message = str(excinfo.value)
    assert "forbidden field" in message or "unknown fields" in message


def test_unknown_benign_field_still_fails_closed():
    body = json.loads(_base_line())
    body["weather"] = "sunny"
    with pytest.raises(ProtocolError, match="unknown fields"):
        parse_request(json.dumps(body))


def test_missing_required_field_fails():
    body = json.loads(_base_line())
    del body["legality_mask"]
    with pytest.raises(ProtocolError, match="missing required"):
        parse_request(json.dumps(body))


def test_wrong_schema_fails():
    body = json.loads(_base_line())
    body["schema"] = "other/v9"
    with pytest.raises(ProtocolError, match="unknown IPC schema"):
        parse_request(json.dumps(body))


def test_empty_mask_is_rejected():
    from deep_green.protocol import validate_parsed_request

    raw = parse_request(_base_line())
    raw["legality_mask"] = [0, 0, 0, 0]
    with pytest.raises(ProtocolError, match="empty legality mask"):
        validate_parsed_request(raw, action_words=WORDS)


def test_row_turn_mismatch_is_rejected():
    from deep_green.protocol import validate_parsed_request

    request = _request(turn=2, played_rows=(("crane", "bbybb"),))
    raw = json.loads(serialize_request(request, action_words=WORDS))
    raw["played_rows"] = []
    with pytest.raises(ProtocolError, match="played rows"):
        validate_parsed_request(raw, action_words=WORDS)


def test_request_hash_binds_all_public_bytes():
    a = _request()
    b = _request(legality_mask=(True, True, False, True))
    ha = a.request_hash(action_words=WORDS)
    hb = b.request_hash(action_words=WORDS)
    assert ha != hb
    assert ha == a.request_hash(action_words=WORDS)


def test_tampered_request_hash_detected():
    from deep_green.protocol import validate_parsed_request

    raw = json.loads(_base_line())
    raw["request_hash"] = "0" * 64
    with pytest.raises(ProtocolError, match="request hash mismatch"):
        validate_parsed_request(raw, action_words=WORDS)


def test_action_space_digest_drift_is_rejected():
    from deep_green.protocol import validate_parsed_request

    raw = json.loads(_base_line())
    raw["action_space_sha256"] = "0" * 64
    with pytest.raises(ProtocolError, match="action-space digest mismatch"):
        validate_parsed_request(raw, action_words=WORDS)


def test_invalid_marks_rejected():
    from deep_green.protocol import validate_parsed_request

    request = _request(turn=2, played_rows=(("crane", "bbxbb"),))
    line = serialize_request.__wrapped__ if False else None
    del line
    raw = {
        "schema": PROTOCOL_SCHEMA,
        "mode": "normal",
        "turn": 2,
        "played_rows": [["crane", "bbxbb"]],
        "action_space_sha256": ACTION_DIGEST,
        "action_count": len(WORDS),
        "legality_mask": [1, 1, 1, 1],
        "request_hash": request.request_hash(action_words=WORDS),
    }
    with pytest.raises(ProtocolError, match="invalid marks"):
        validate_parsed_request(raw, action_words=WORDS)


def _validated_raw(**overrides):
    raw = json.loads(_base_line())
    raw.update(overrides)
    return raw


@pytest.mark.parametrize("bad_turn", [True, "1", 1.0, 0, 7, None, [1]])
def test_turn_is_never_coerced(bad_turn):
    from deep_green.protocol import validate_parsed_request

    with pytest.raises(ProtocolError, match="turn"):
        validate_parsed_request(_validated_raw(turn=bad_turn), action_words=WORDS)


@pytest.mark.parametrize("bad_count", [True, "4", 4.0, 3, 5, None])
def test_action_count_is_never_coerced(bad_count):
    from deep_green.protocol import validate_parsed_request

    with pytest.raises(ProtocolError, match="action_count|action count"):
        validate_parsed_request(
            _validated_raw(action_count=bad_count), action_words=WORDS
        )


@pytest.mark.parametrize(
    "bad_mask",
    [
        [True, True, True, True],
        [1, 1, 1, "1"],
        [1, 1, 1, 0.5],
        [1, 1, 1, 2],
        [1, 1, 1, None],
        "1111",
        [1, 1, 1],
    ],
)
def test_legality_mask_values_are_never_coerced(bad_mask):
    from deep_green.protocol import validate_parsed_request

    with pytest.raises(ProtocolError):
        validate_parsed_request(
            _validated_raw(legality_mask=bad_mask), action_words=WORDS
        )


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_constants_rejected(constant):
    line = _base_line().replace('"turn":1', f'"turn":{constant}')
    with pytest.raises(ProtocolError):
        parse_request(line)


def test_oversized_request_line_rejected():
    body = json.loads(_base_line())
    body["schema"] = PROTOCOL_SCHEMA
    pad = "x" * (65536 + 1)
    body["played_rows"] = []
    line = json.dumps(body)[:-1] + f',"pad":"{pad}"' + "}"
    with pytest.raises(ProtocolError, match="bytes|size|oversized"):
        parse_request(line)


def test_bytes_input_decoded_strictly():
    with pytest.raises(ProtocolError):
        parse_request(b"\xff\xfe{}\n")
    assert parse_request(_base_line().encode("utf-8"))["mode"] == "normal"


@pytest.mark.parametrize(
    "rows",
    [
        "not-a-list",
        ["crane"],
        [["crane"]],
        [["crane", "bbbbb", "extra"]],
        [[12345, "bbbbb"]],
        [["CRANE", "bbbbb"]],
        [["crane", "BBBBB"]],
    ],
)
def test_malformed_row_shapes_and_characters(rows):
    from deep_green.protocol import validate_parsed_request

    raw = _validated_raw(turn=2, played_rows=rows)
    with pytest.raises(ProtocolError):
        validate_parsed_request(raw, action_words=WORDS)


def test_solved_row_continuation_rejected():
    from deep_green.protocol import validate_parsed_request

    raw = _validated_raw(
        turn=3, played_rows=[["crane", "ggggg"], ["slate", "bbbbb"]]
    )
    with pytest.raises(ProtocolError):
        validate_parsed_request(raw, action_words=WORDS)


@pytest.mark.parametrize("bad_hash", ["A" * 64, "0" * 63, 123, None])
def test_request_hash_must_be_lowercase_sha256(bad_hash):
    from deep_green.protocol import validate_parsed_request

    raw = _validated_raw(request_hash=bad_hash)
    with pytest.raises(ProtocolError):
        validate_parsed_request(raw, action_words=WORDS)


def test_action_digest_must_be_lowercase_sha256():
    from deep_green.protocol import validate_parsed_request

    raw = _validated_raw(action_space_sha256=ACTION_DIGEST.upper())
    with pytest.raises(ProtocolError):
        validate_parsed_request(raw, action_words=WORDS)


def test_serializer_rejects_non_bool_mask():
    with pytest.raises(ProtocolError):
        serialize_request(
            _request(legality_mask=(1, 1, 1, 1)), action_words=WORDS
        )


def test_serializer_rejects_noncanonical_action_words():
    with pytest.raises(ProtocolError):
        serialize_request(_request(), action_words=("slate", "crane", "audio", "pouty"))


def test_serializer_rejects_bad_request_field_types():
    with pytest.raises(ProtocolError):
        serialize_request(_request(turn="1"), action_words=WORDS)
    with pytest.raises(ProtocolError):
        serialize_request(_request(mode="easy"), action_words=WORDS)
