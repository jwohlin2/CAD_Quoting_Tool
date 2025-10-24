"""Project automation sessions configured to target the src layout."""
from __future__ import annotations

import pathlib

import nox

PYTHON_VERSION = "3.11"
SRC_DIR = "src"
TESTS_DIR = "tests"
PACKAGE_IMPORT = "cad_quoter"

nox.options.sessions = ("lint", "typecheck", "tests")


def install_project(session: nox.Session) -> None:
    """Install the project in editable mode with tooling dependencies."""
    session.install("-e", ".")


@nox.session(python=PYTHON_VERSION)
def lint(session: nox.Session) -> None:
    """Run Ruff against the source and tests packages."""
    session.install("ruff")
    session.run("ruff", "check", SRC_DIR, TESTS_DIR)


@nox.session(python=PYTHON_VERSION)
def typecheck(session: nox.Session) -> None:
    """Run static type analysis with Pyright and Mypy."""
    install_project(session)
    session.install("pyright", "mypy")
    session.run("pyright")
    session.run("mypy", SRC_DIR)


@nox.session(python=PYTHON_VERSION)
def tests(session: nox.Session) -> None:
    """Execute the unit tests under coverage configured for the src layout."""
    install_project(session)
    session.install("pytest", "coverage")
    coverage_source = pathlib.Path(SRC_DIR, PACKAGE_IMPORT)
    session.run(
        "coverage",
        "run",
        "--source",
        str(coverage_source),
        "-m",
        "pytest",
        external=True,
    )
    session.run("coverage", "report")
