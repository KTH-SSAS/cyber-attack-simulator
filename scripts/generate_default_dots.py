#!/usr/bin/env python3

if __name__ == "__main__":
    from argparse import ArgumentParser

    from attack_simulator.graph import save_all_default_graphviz

    parser = ArgumentParser(description="Dot generator.")
    parser.add_argument(
        "-i",
        "--indexed",
        type=bool,
        default=False,
        help="Include indices in front of attack step names.",
    )
    args = parser.parse_args()

    save_all_default_graphviz(None, indexed=args.indexed)
