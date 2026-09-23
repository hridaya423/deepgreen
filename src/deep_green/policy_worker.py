"""Policy worker process: scores the full action vocabulary, fails closed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESPONSE_SCHEMA = "wordle.policy-response/v1"


def worker_main() -> int:
    """Read requests on stdin, write responses on stdout; any fault exits."""
    if len(sys.argv) != 2:
        print("usage: policy_worker <release-dir>", file=sys.stderr)
        return 2
    release_dir = Path(sys.argv[1])
    from deep_green.release import ReleaseError, load_release

    try:
        payload = load_release(release_dir)
    except (ReleaseError, OSError) as exc:
        print(f"release load failed: {exc}", file=sys.stderr)
        return 3
    words = payload.action_words

    import numpy as np

    from deep_green.numeric_policy import encode_lexicon
    from deep_green.protocol import (
        MAX_REQUEST_BYTES,
        ProtocolError,
        parse_request,
        validate_parsed_request,
    )

    letter_codes = {chr(ord("a") + i): i for i in range(26)}
    mark_codes = {"b": 0, "y": 1, "g": 2}
    mode_codes = {"normal": [1.0, 0.0], "hard": [0.0, 1.0]}
    word_letters, word_positions = encode_lexicon(list(words))

    def emit(response: dict) -> None:
        line = json.dumps(response, sort_keys=True, separators=(",", ":"))
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def fail(message: str) -> int:
        emit(
            {
                "schema": RESPONSE_SCHEMA,
                "status": "FAIL",
                "reason": message[:200],
                "request_hash": None,
                "scored_word_count": 0,
                "argmax": None,
                "artifact_sha256": payload.weights_sha256,
            }
        )
        return 8

    while True:
        line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
        if not line:
            return 0
        if len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
            return fail("request exceeds the bounded line length")
        try:
            raw = parse_request(line)
            request = validate_parsed_request(raw, action_words=words)
        except ProtocolError as exc:
            return fail(f"protocol violation: {exc}")
        try:
            row_count = len(request.played_rows)
            slot_count = max(row_count, 1)
            batch_rows = np.zeros((1, slot_count, 5), dtype=np.int64)
            batch_marks = np.zeros((1, slot_count, 5), dtype=np.int64)
            row_mask = np.zeros((1, slot_count), dtype=bool)
            row_mask[0, :row_count] = True
            for index, (guess_text, feedback_text) in enumerate(request.played_rows):
                batch_rows[0, index] = [letter_codes[c] for c in guess_text]
                batch_marks[0, index] = [mark_codes[c] for c in feedback_text]
            mask = np.array(request.legality_mask, dtype=bool)
        except (KeyError, TypeError, ValueError) as exc:
            return fail(f"malformed request body: {exc}")

        from deep_green.numeric_policy import score_logits, select_action

        logits = score_logits(
            tensors=payload.tensors,
            row_letters=batch_rows,
            row_marks=batch_marks,
            row_positions=np.broadcast_to(np.arange(5), batch_rows.shape).copy(),
            row_mask=row_mask,
            turn=np.array([float(request.turn)], dtype=np.float32),
            mode=np.array([mode_codes[request.mode]], dtype=np.float32),
            word_letters=word_letters,
            word_positions=word_positions,
        )
        if not np.isfinite(logits).all():
            return fail("nonfinite logits end the episode")
        argmax_index = select_action(logits[0], mask)
        emit(
            {
                "schema": RESPONSE_SCHEMA,
                "status": "OK",
                "request_hash": raw["request_hash"],
                "scored_word_count": len(words),
                "argmax": argmax_index,
                "artifact_sha256": payload.weights_sha256,
            }
        )


if __name__ == "__main__":
    sys.exit(worker_main())
