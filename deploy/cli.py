"""Command line interface for deployment helpers.

The CLI intentionally keeps dependencies minimal so it can run from a bare
Python installation on freshly provisioned hosts.  Each sub-command mirrors a
previous ad-hoc helper:

* ``pull`` replaces the ``git-auto-pull`` checkout refresh script.
* ``bootstrap`` installs Python dependencies as done by historical bootstrap
  notes.
* ``check`` wraps common pre-deployment validation such as the pytest suite.

Additional utilities can be registered in the same module to avoid scattering
shell snippets throughout the repository.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from . import get_repo_root


@dataclass(slots=True)
class CommandRunner:
    """Execute subprocess commands rooted at the repository."""

    repo_root: Path
    dry_run: bool = False
    verbose: bool = False

    def run(
        self,
        command: Sequence[str],
        *,
        check: bool = True,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        """Execute ``command`` relative to the repository root."""

        display = "$ " + " ".join(shlex.quote(part) for part in command)
        if self.verbose or self.dry_run:
            print(display)
        if self.dry_run:
            return subprocess.CompletedProcess(command, returncode=0)

        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        return subprocess.run(
            command,
            cwd=str(self.repo_root),
            env=process_env,
            check=check,
        )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Echo commands before execution.",
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deploy",
        description="Deployment and maintenance helpers for the CAD Quoting Tool.",
    )
    _add_common_arguments(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)

    pull_parser = subparsers.add_parser(
        "pull", help="Fast-forward the current checkout from the remote.",
    )
    pull_parser.add_argument(
        "branch",
        nargs="?",
        help="Optional branch to pull (defaults to the upstream tracking branch).",
    )
    pull_parser.add_argument(
        "--remote",
        default="origin",
        help="Remote to pull from (default: origin).",
    )
    pull_parser.set_defaults(handler=handle_pull)

    bootstrap_parser = subparsers.add_parser(
        "bootstrap", help="Install Python dependencies required by the app.",
    )
    bootstrap_parser.add_argument(
        "--requirements",
        default="requirements.txt",
        help="Path to the requirements file (default: requirements.txt).",
    )
    bootstrap_parser.add_argument(
        "--upgrade-pip",
        action="store_true",
        help="Upgrade pip before installing dependencies.",
    )
    bootstrap_parser.set_defaults(handler=handle_bootstrap)

    check_parser = subparsers.add_parser(
        "check",
        help="Run the default validation suite prior to deployment.",
    )
    check_parser.add_argument(
        "--tests",
        nargs="*",
        default=["tests"],
        help="Targets to pass to pytest (default: tests).",
    )
    check_parser.set_defaults(handler=handle_check)

    env_parser = subparsers.add_parser(
        "print-env", help="Dump the runtime configuration via appV5.py --print-env.",
    )
    env_parser.set_defaults(handler=handle_print_env)

    return parser


def _build_runner(args: argparse.Namespace) -> CommandRunner:
    repo_root = get_repo_root()
    return CommandRunner(repo_root=repo_root, dry_run=args.dry_run, verbose=args.verbose)


def handle_pull(args: argparse.Namespace) -> int:
    runner = _build_runner(args)

    fetch_cmd: list[str] = ["git", "fetch", args.remote]
    runner.run(fetch_cmd)

    pull_cmd: list[str] = ["git", "pull", "--ff-only"]
    if args.branch:
        pull_cmd.extend([args.remote, args.branch])
    runner.run(pull_cmd)
    return 0


def handle_bootstrap(args: argparse.Namespace) -> int:
    runner = _build_runner(args)

    if args.upgrade_pip:
        runner.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    requirements_path = Path(args.requirements)
    if not requirements_path.is_absolute():
        requirements_path = runner.repo_root / requirements_path

    runner.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--requirement",
            str(requirements_path),
        ]
    )
    return 0


def handle_check(args: argparse.Namespace) -> int:
    runner = _build_runner(args)

    pytest_args: list[str] = ["pytest", *args.tests]
    runner.run(pytest_args)
    return 0


def handle_print_env(args: argparse.Namespace) -> int:
    runner = _build_runner(args)
    runner.run([sys.executable, "appV5.py", "--print-env"])
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handler = args.handler
    return handler(args)


if __name__ == "__main__":  # pragma: no cover - exercised via module execution
    raise SystemExit(main())
