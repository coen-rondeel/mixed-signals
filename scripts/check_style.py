"""Check the repository style rules that ruff cannot enforce.

Rules:
    STY001: no one-letter names for variables, parameters, attributes,
        functions, classes or type parameters; ``_`` stays allowed.
    STY002: decimal number literals with five or more consecutive digits use
        ``_`` separators, for example ``10_000`` and ``0.000_01``.
    STY003: no em-dashes in code or documentation.
    STY004: no emojis in code or documentation.
    STY005: comment blocks in Python files are at most two lines long.

A Python line containing ``# style: allow`` is exempt from STY001 and STY002.

Usage:
    uv run python scripts/check_style.py [paths ...]

Without paths, the script checks every file under the current directory that
git tracks or would track.
"""

import argparse
import ast
import io
import re
import subprocess
import tokenize
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = frozenset(
    {".py", ".md", ".rst", ".toml", ".yaml", ".yml", ".txt", ".cfg", ".ini"}
)
ALLOW_MARKER = "# style: allow"
EM_DASH = chr(0x2014)
# * Pictographs, emoticons and flags, then miscellaneous symbols and dingbats.
EMOJI_PATTERN = re.compile(
    f"[{chr(0x1F000)}-{chr(0x1FAFF)}{chr(0x2600)}-{chr(0x27BF)}{chr(0xFE0F)}]"
)
LONG_DIGIT_RUN = re.compile(r"\d{5,}")
MAX_COMMENT_BLOCK_LINES = 2
TOOL_PRAGMA = re.compile(r"#\s*(noqa|type:|fmt:|pragma|pyright:|mypy:|ruff:|isort:)")


@dataclass(frozen=True)
class Finding:
    """One rule violation at a file position."""

    path: Path
    line: int
    column: int
    code: str
    message: str

    def __str__(self) -> str:
        """Format the finding as ``path:line:column: code message``."""
        return f"{self.path}:{self.line}:{self.column}: {self.code} {self.message}"


def bound_names(tree: ast.AST) -> Iterator[tuple[str, int, int]]:
    """Yield each name a module binds, with its line and zero-based column."""
    for node in ast.walk(tree):
        match node:
            case (
                ast.Name(id=name, ctx=ast.Store())
                | ast.Attribute(attr=name, ctx=ast.Store())
                | ast.arg(arg=name)
                | ast.FunctionDef(name=name)
                | ast.AsyncFunctionDef(name=name)
                | ast.ClassDef(name=name)
                | ast.ExceptHandler(name=str() as name)
                | ast.alias(asname=str() as name)
                | ast.MatchAs(name=str() as name)
                | ast.MatchStar(name=str() as name)
                | ast.MatchMapping(rest=str() as name)
                | ast.TypeVar(name=name)
                | ast.ParamSpec(name=name)
                | ast.TypeVarTuple(name=name)
            ):
                yield name, node.lineno, node.col_offset


def has_long_digit_run(number_literal: str) -> bool:
    """Return whether a decimal literal has five or more digits in a row."""
    lowered = number_literal.lower()
    if lowered.startswith(("0x", "0o", "0b")):
        return False
    mantissa = lowered.rstrip("j").split("e", maxsplit=1)[0]
    return LONG_DIGIT_RUN.search(mantissa) is not None


def check_text(path: Path, lines: Sequence[str]) -> Iterator[Finding]:
    """Check every line of a text file for em-dashes and emojis."""
    for line_number, line in enumerate(lines, start=1):
        if (column := line.find(EM_DASH)) >= 0:
            yield Finding(path, line_number, column + 1, "STY003", "em-dash")
        if emoji_match := EMOJI_PATTERN.search(line):
            yield Finding(path, line_number, emoji_match.start() + 1, "STY004", "emoji")


def check_python(path: Path, source: str) -> Iterator[Finding]:
    """Check names, number literals and comment blocks in Python source."""
    try:
        tree = ast.parse(source, filename=str(path))
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (SyntaxError, tokenize.TokenError) as error:
        yield Finding(path, 1, 1, "STY000", f"could not parse: {error}")
        return

    allowed_lines = {
        token.start[0]
        for token in tokens
        if token.type == tokenize.COMMENT and ALLOW_MARKER in token.string
    }
    for name, line_number, column in bound_names(tree):
        if len(name) == 1 and name != "_" and line_number not in allowed_lines:
            yield Finding(
                path, line_number, column + 1, "STY001", f"one-letter name {name!r}"
            )

    comment_block_lines: list[int] = []
    for token in tokens:
        line_number, column = token.start
        if (
            token.type == tokenize.NUMBER
            and line_number not in allowed_lines
            and has_long_digit_run(token.string)
        ):
            yield Finding(
                path,
                line_number,
                column + 1,
                "STY002",
                f"number {token.string} needs _ digit separators",
            )
        is_block_comment = (
            token.type == tokenize.COMMENT
            and token.line.lstrip().startswith("#")
            and not TOOL_PRAGMA.match(token.string)
            and not (line_number == 1 and token.string.startswith("#!/"))
        )
        if not is_block_comment:
            continue
        if comment_block_lines and line_number != comment_block_lines[-1] + 1:
            yield from _long_comment_block(path, comment_block_lines)
            comment_block_lines = []
        comment_block_lines.append(line_number)
    yield from _long_comment_block(path, comment_block_lines)


def _long_comment_block(path: Path, block_lines: Sequence[int]) -> Iterator[Finding]:
    if len(block_lines) > MAX_COMMENT_BLOCK_LINES:
        yield Finding(
            path,
            block_lines[0],
            1,
            "STY005",
            f"comment block of {len(block_lines)} lines; move long notes to the docs",
        )


def check_file(path: Path) -> Iterator[Finding]:
    """Run every rule that applies to one file."""
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    yield from check_text(path, source.splitlines())
    if path.suffix == ".py":
        yield from check_python(path, source)


def tracked_files() -> list[Path]:
    """List files under the current directory that git tracks or would track."""
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [Path(name) for name in listing.split("\0") if name]


def main(argv: Sequence[str] | None = None) -> int:
    """Check the given files, or all tracked files, and return the exit code."""
    parser = argparse.ArgumentParser(
        description="Check the repository style rules that ruff cannot enforce."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="files to check")
    arguments = parser.parse_args(argv)
    candidate_paths: list[Path] = arguments.paths or tracked_files()
    findings = sorted(
        (
            finding
            for path in candidate_paths
            if path.suffix in TEXT_SUFFIXES and path.is_file()
            for finding in check_file(path)
        ),
        key=lambda finding: (str(finding.path), finding.line, finding.column),
    )
    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
