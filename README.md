# Deep Green

A pure-neural Wordle policy: one width-320 float32 network — 1,550,400
parameters, 6.2 MB — that replays a certified exact oracle
decision-for-decision on the frozen OpenWordle-v1 benchmark, in normal
and hard mode. The policy sees only the public board; the oracle tree is
used to train and to score, never to play.

## Playing

    uv sync --extra dev
    cd web && npm ci && npm run build && cd ..
    uv run wordle play serve --port 3101

- **Challenge** takes any word from the 5,022-word vocabulary and shows
  the certified policy's exact play against it. The certified tier covers
  the entire vocabulary, so every challenge is `CERTIFIED` whenever the
  public proof verifies; `EXHIBITION` is reserved for anything outside
  the frozen corpus.
- **Assist** is always `EXHIBITION`: you record feedback from a real
  external game whose answer is unknowable here, so nothing about it can
  be certified.
