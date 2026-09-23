import argparse
import sys

COMMAND_GROUPS = ("benchmark", "oracle", "runs", "train", "evaluate", "play", "release")

_BENCHMARK_SUBCOMMANDS = {
    "preflight": ("--candidate",),
    "freeze": ("--candidate", "--protocol", "--output"),
    "verify": ("--frozen",),
}


def _stub_handler(group_name):
    def run(_args):
        print(f"error: '{group_name}' is not implemented yet", file=sys.stderr)
        return 2

    return run


def _rebuild_index_handler():
    def run(args):
        from deep_green.oracle_commands import _run_rebuild_index

        return _run_rebuild_index(args)

    return run


def _benchmark_handler():
    def run(args):
        from deep_green.benchmark_commands import (
            run_benchmark_freeze,
            run_benchmark_preflight,
            run_benchmark_verify,
        )

        handlers = {
            "preflight": run_benchmark_preflight,
            "freeze": run_benchmark_freeze,
            "verify": run_benchmark_verify,
        }
        return handlers[args.benchmark_command](args)

    return run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="wordle",
        description="Train and evaluate a neural Wordle policy against an exact oracle.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command", required=True)
    for group_name in COMMAND_GROUPS:
        group_parser = subparsers.add_parser(group_name, help=f"{group_name} command group")
        if group_name == "benchmark":
            benchmark_subparsers = group_parser.add_subparsers(
                dest="benchmark_command", metavar="benchmark_command", required=True
            )
            for name in ("preflight", "freeze", "verify"):
                command_parser = benchmark_subparsers.add_parser(name, help=f"benchmark {name}")
                if name == "preflight":
                    command_parser.add_argument("--candidate", required=True)
                elif name == "freeze":
                    command_parser.add_argument("--candidate", required=True)
                    command_parser.add_argument("--protocol", required=True)
                    command_parser.add_argument("--output", required=True)
                else:
                    command_parser.add_argument("--frozen", required=True, dest="frozen")
                command_parser.set_defaults(handler=_benchmark_handler())
        elif group_name == "oracle":
            from deep_green.oracle_commands import register_oracle_subcommands

            register_oracle_subcommands(group_parser)
        elif group_name == "evaluate":
            from deep_green.evaluate_commands import register_evaluate_subcommands

            register_evaluate_subcommands(group_parser)
        elif group_name == "train":
            from deep_green.evaluate_commands import register_train_subcommands

            register_train_subcommands(group_parser)
        elif group_name == "play":
            from deep_green.evaluate_commands import register_play_subcommands

            register_play_subcommands(group_parser)
        elif group_name == "release":
            from deep_green.release_commands import register_release_subcommands

            register_release_subcommands(group_parser)
        elif group_name == "runs":
            runs_parser = group_parser.add_subparsers(
                dest="runs_command", metavar="runs_command", required=True
            )
            rebuild = runs_parser.add_parser(
                "rebuild-index", help="rebuild the SQLite index from run evidence"
            )
            rebuild.add_argument("--runs-root", default="runs")
            rebuild.set_defaults(handler=_rebuild_index_handler())
        else:
            group_parser.set_defaults(handler=_stub_handler(group_name))

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
