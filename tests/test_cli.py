from deep_green.cli import main


def test_help_lists_the_domain_commands(capsys):
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "benchmark" in output
    assert "oracle" in output
    assert "train" in output
    assert "evaluate" in output
    assert "play" in output
