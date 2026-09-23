from dataclasses import FrozenInstanceError
from hashlib import sha256
from itertools import product

import pytest

from deep_green.rules import (
    ActionSpace,
    Board,
    Feedback,
    GuessRow,
    Mode,
    Tile,
    Word5,
    board_from_public_state,
    is_legal_normal,
    is_legal_nyt_standard_hard,
    legal_actions,
    legality_digest,
    score_guess,
)


def make_row(word: str, marks: str) -> GuessRow:
    return GuessRow(Word5.parse(word), Feedback.parse(marks))


def make_board(mode, *pairs):
    rows = tuple(make_row(word, marks) for word, marks in pairs)
    return Board.create(mode, rows, len(rows) + 1)


def reference_score(guess: str, answer: str) -> str:
    marks = ["b"] * 5
    remaining = {}
    for letter in answer:
        remaining[letter] = remaining.get(letter, 0) + 1
    for index, letter in enumerate(guess):
        if answer[index] == letter:
            marks[index] = "g"
            remaining[letter] -= 1
    for index, letter in enumerate(guess):
        if marks[index] != "g" and remaining.get(letter, 0) > 0:
            marks[index] = "y"
            remaining[letter] -= 1
    return "".join(marks)


def test_word5_accepts_exactly_five_lowercase_ascii_letters():
    assert Word5.parse("crane").value == "crane"
    assert Word5.parse("zebra").value == "zebra"


@pytest.mark.parametrize(
    "text",
    [
        "CRANE",
        "Crane",
        "cRANE",
        "cran",
        "cranes",
        "",
        "cra1e",
        "cra e",
        "crânè",
    ],
)
def test_word5_rejects_invalid_text(text):
    with pytest.raises(ValueError):
        Word5.parse(text)


def test_feedback_maps_five_marks_to_tiles_in_order():
    assert Feedback.parse("bgbyy").tiles == (
        Tile.ABSENT,
        Tile.CORRECT,
        Tile.ABSENT,
        Tile.PRESENT,
        Tile.PRESENT,
    )


@pytest.mark.parametrize(
    "marks",
    [
        "bbbb",
        "bbbbbb",
        "bbbxg",
        "BBGGG",
        "bbb g",
        "bbbÿg",
        "",
    ],
)
def test_feedback_rejects_unknown_or_wrong_length_marks(marks):
    with pytest.raises(ValueError):
        Feedback.parse(marks)


def test_tile_canonical_order_is_absent_present_correct():
    assert Tile.ABSENT < Tile.PRESENT < Tile.CORRECT


def test_tile_sort_matches_absent_present_correct_order():
    assert sorted([Tile.CORRECT, Tile.ABSENT, Tile.PRESENT]) == [
        Tile.ABSENT,
        Tile.PRESENT,
        Tile.CORRECT,
    ]


def test_tile_comparison_with_plain_string_is_not_implemented():
    assert Tile.ABSENT.__lt__("a") is NotImplemented


@pytest.mark.parametrize(("mode", "wire"), [(Mode.NORMAL, "normal"), (Mode.HARD, "hard")])
def test_mode_wire_values_are_pinned(mode, wire):
    assert mode.value == wire


def test_guess_row_pairs_word_and_feedback():
    guess = Word5.parse("slate")
    feedback = Feedback.parse("ggggb")
    row = GuessRow(guess, feedback)
    assert row.guess is guess
    assert row.feedback is feedback


def test_board_holds_mode_and_rows():
    rows = (make_row("crane", "bbbyy"), make_row("slate", "ggggb"))
    board = Board.create(Mode.HARD, rows, next_turn=3)
    assert board.mode is Mode.HARD
    assert board.rows == rows


def test_board_accepts_empty_board_at_first_turn():
    assert Board.create(Mode.NORMAL, (), next_turn=1).rows == ()


def test_public_board_parser_owns_pending_turn_and_legality_dispatch():
    board = board_from_public_state(
        mode="hard", turn=2, rows=[["allee", "yybbb"]]
    )
    actions = ActionSpace.parse(("alley", "crane", "llama"))
    mask = legal_actions(board, actions)
    assert mask == [True, False, True]
    assert legality_digest(mask) == sha256(b"101").hexdigest()
    with pytest.raises(ValueError, match="pending turn"):
        board_from_public_state(mode="normal", turn=7, rows=[])
    with pytest.raises(ValueError, match="pending turn"):
        board_from_public_state(mode="normal", turn=True, rows=[])


def test_board_coerces_row_input_to_a_tuple():
    rows = [make_row("crane", "bbbyy"), make_row("slate", "bbbbb")]
    board = Board.create(Mode.NORMAL, rows, next_turn=3)
    assert isinstance(board.rows, tuple)
    assert board.rows == tuple(rows)


def test_board_rejects_more_than_six_rows():
    rows = tuple(make_row("crane", "bbbby") for _ in range(7))
    with pytest.raises(ValueError):
        Board.create(Mode.NORMAL, rows, next_turn=8)


def test_board_rejects_rows_after_a_solved_row():
    rows = (make_row("crane", "ggggg"), make_row("slate", "bbbbb"))
    with pytest.raises(ValueError):
        Board.create(Mode.NORMAL, rows, next_turn=3)


def test_board_accepts_a_final_solved_row():
    rows = (make_row("crane", "bbbyy"), make_row("slate", "ggggg"))
    board = Board.create(Mode.NORMAL, rows, next_turn=3)
    assert board.rows[-1].feedback.tiles == (Tile.CORRECT,) * 5


def test_board_rejects_next_turn_disagreeing_with_row_count():
    rows = (make_row("crane", "bbbby"),)
    with pytest.raises(ValueError):
        Board.create(Mode.NORMAL, rows, next_turn=1)
    with pytest.raises(ValueError):
        Board.create(Mode.NORMAL, rows, next_turn=3)


def test_score_guess_all_green_and_all_absent():
    assert score_guess(Word5.parse("crane"), Word5.parse("crane")).tiles == (Tile.CORRECT,) * 5
    assert score_guess(Word5.parse("crane"), Word5.parse("boost")).tiles == (Tile.ABSENT,) * 5


def test_score_guess_allee_against_apple_consumes_multiplicity_green_first():
    assert score_guess(Word5.parse("allee"), Word5.parse("apple")).tiles == Feedback.parse("gybbg").tiles


def test_score_guess_belle_against_label_assigns_yellows_left_to_right():
    assert score_guess(Word5.parse("belle"), Word5.parse("label")).tiles == Feedback.parse("yyyyb").tiles


def test_score_guess_greens_are_claimed_before_yellows_left_to_right():
    assert score_guess(Word5.parse("treat"), Word5.parse("crate")).tiles == Feedback.parse("ygyyb").tiles
    assert score_guess(Word5.parse("roars"), Word5.parse("zebra")).tiles == Feedback.parse("bbygb").tiles


def test_score_guess_green_steals_multiplicity_from_another_position():
    assert score_guess(Word5.parse("about"), Word5.parse("aaaaa")).tiles == Feedback.parse("gbbbb").tiles


def test_score_guess_matches_reference_over_toy_alphabet():
    words = ["".join(letters) for letters in product("ab", repeat=5)]
    seen = set()
    for guess in words:
        for answer in words:
            feedback = score_guess(Word5.parse(guess), Word5.parse(answer))
            marks = "".join(tile.value for tile in feedback.tiles)
            assert marks == reference_score(guess, answer)
            seen.add(marks)
    assert len(words) * len(words) == 1024
    assert len(seen) == 89


def test_trusted_values_are_frozen():
    word = Word5.parse("crane")
    feedback = Feedback.parse("bgbyg")
    row = GuessRow(word, feedback)
    board = Board.create(Mode.NORMAL, (row,), next_turn=2)
    with pytest.raises(FrozenInstanceError):
        word.value = "apple"
    with pytest.raises(FrozenInstanceError):
        feedback.tiles = ()
    with pytest.raises(FrozenInstanceError):
        row.guess = word
    with pytest.raises(FrozenInstanceError):
        board.rows = ()


def test_action_space_sorts_words_once_into_canonical_order():
    words = ["slate", "crane", "about"]
    space = ActionSpace.parse(words)
    assert space.words == ("about", "crane", "slate")
    assert all(isinstance(word, str) for word in space.words)
    again = ActionSpace.parse(list(space.words))
    assert again.words == space.words


def test_action_space_digest_is_stable_across_orderings_and_repeat_calls():
    words = ["slate", "crane", "about", "irate", "treat", "eerie"]
    space = ActionSpace.parse(words)
    shuffled = ActionSpace.parse(["eerie", "treat", "slate", "about", "irate", "crane"])
    assert shuffled.words == space.words
    assert shuffled.digest == space.digest
    first = space.digest
    second = space.digest
    assert first is second
    expected = sha256(b"about\ncrane\neerie\nirate\nslate\ntreat\n").hexdigest()
    assert first == expected


def test_action_space_digest_matches_canonical_file_bytes():
    assert ActionSpace.parse(("slate", "crane")).digest == sha256(b"crane\nslate\n").hexdigest()


def test_action_space_rejects_duplicates_even_after_case_normalization():
    with pytest.raises(ValueError):
        ActionSpace.parse(["crane", "slate", "crane"])
    with pytest.raises(ValueError):
        ActionSpace.parse(["CRANE", "crane"])


def test_action_space_rejects_words_that_fail_word5_validity():
    with pytest.raises(ValueError):
        ActionSpace.parse(["crane", "cran"])


def test_mask_round_trips_through_space_and_bytes():
    words = ["crane", "slate"]
    mask = [True, False]
    space = ActionSpace.parse(words)
    encoded = bytes([1, 0])
    restored = space.mask(encoded)
    assert list(restored) == mask
    assert len(restored) == len(space.words)
    assert all(isinstance(flag, bool) for flag in restored)


def test_mask_rejects_wrong_length():
    space = ActionSpace.parse(["crane", "slate"])
    for bad in ([True], [True, False, True]):
        with pytest.raises(ValueError):
            space.mask(bad)


def test_mask_rejects_digest_mismatch_with_well_lengthed_input():
    space = ActionSpace.parse(["crane", "slate", "about"])
    forged_digest = "0" * 64
    with pytest.raises(ValueError):
        space.mask(bytes([1, 0, 1]), action_space_digest=forged_digest)


def test_normal_legality_is_accepted_membership_on_any_board():
    words = ["crane", "slate", "truck"]
    space = ActionSpace.parse(words)
    empty = make_board(Mode.NORMAL)
    restrictive = make_board(Mode.NORMAL, ("trace", "ggbbb"))
    assert set(is_legal_normal(empty, space)) == {True}
    normal_on_restrictive = set(is_legal_normal(restrictive, space))
    hard_on_restrictive = {
        word
        for word, legal in zip(space.words, is_legal_nyt_standard_hard(restrictive, space))
        if legal
    }
    assert normal_on_restrictive == {True}
    assert len(hard_on_restrictive) < len(words)


def test_hard_legality_preserves_greens():
    board = make_board(Mode.HARD, ("trace", "ggbbb"))
    verdicts = dict(
        zip(sorted(["truck", "crane", "irate"]), is_legal_nyt_standard_hard(board, ActionSpace.parse(["truck", "crane", "irate"])))
    )
    assert verdicts["truck"] is True
    assert verdicts["crane"] is False
    assert verdicts["irate"] is False


def test_hard_legality_enforces_revealed_letter_multiplicities():
    board = make_board(Mode.HARD, ("elite", "ggbby"))
    verdicts = dict(zip(("elate", "steel", "least"), is_legal_nyt_standard_hard(board, ActionSpace.parse(["elate", "steel", "least"]))))
    assert verdicts["elate"] is True
    assert verdicts["steel"] is False
    assert verdicts["least"] is False

    belle_board = make_board(Mode.HARD, ("belle", "yyyyb"))
    verdicts = dict(zip(sorted(["label", "legal", "aloee", "skill"]), is_legal_nyt_standard_hard(belle_board, ActionSpace.parse(["label", "legal", "aloee", "skill"]))))
    assert verdicts["label"] is True
    assert verdicts["legal"] is False
    assert verdicts["aloee"] is False
    assert verdicts["skill"] is False


def test_hard_legality_takes_the_maximum_demand_across_rows():
    board = make_board(Mode.HARD, ("crane", "bbbby"), ("tepee", "bybgg"))
    verdicts = dict(
        zip(("tepee", "typee"), is_legal_nyt_standard_hard(board, ActionSpace.parse(["tepee", "typee"])))
    )
    assert verdicts["tepee"] is True
    assert verdicts["typee"] is False


def test_hard_legality_permits_gray_reuse_and_yellow_in_old_position():
    gray_board = make_board(Mode.HARD, ("crane", "bbbby"))
    assert is_legal_nyt_standard_hard(gray_board, ActionSpace.parse(["creed"])) == [True]

    yellow_board = make_board(Mode.HARD, ("store", "ybbyy"))
    verdicts = dict(zip(sorted(["rebus", "stare", "rebut"]), is_legal_nyt_standard_hard(yellow_board, ActionSpace.parse(["rebus", "stare", "rebut"]))))
    assert verdicts["rebus"] is True
    assert verdicts["stare"] is True
    assert verdicts["rebut"] is False


def test_scored_legality_never_filters_by_candidate_consistency():
    board = make_board(Mode.HARD, ("crane", "bbbby"))
    assert is_legal_nyt_standard_hard(board, ActionSpace.parse(["eerie"])) == [True]
    inconsistent_board = make_board(Mode.HARD, ("eerie", "gybbb"))
    assert is_legal_nyt_standard_hard(inconsistent_board, ActionSpace.parse(["eerie"])) == [True]
