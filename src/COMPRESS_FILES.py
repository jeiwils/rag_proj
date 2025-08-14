


"""Utility script to compress project artifacts.

This module provides a small command line interface for bundling project
files into a ``.tar.gz`` archive.  It relies solely on the Python standard
library and is intentionally lightweight so it can be run in environments
without additional dependencies.

Example
-------
Compress the ``src`` and ``tests`` directories into ``archive.tar.gz``::

    python src/COMPRESS_FILES.py archive.tar.gz src tests

"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from typing import Iterable


def _add_paths_to_tar(
    tar: tarfile.TarFile, paths: Iterable[Path], base_dir: Path
) -> None:
    """Add ``paths`` to an open ``tar`` archive.

    Parameters
    ----------
    tar:
        An open :class:`tarfile.TarFile` object in write mode.
    paths:
        Iterable of filesystem paths to include.
    base_dir:
        Base directory used to compute relative paths inside the archive.
    """

    for path in paths:
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        arcname = path.relative_to(base_dir)
        tar.add(path, arcname=arcname)


def compress(output: Path, paths: Iterable[str]) -> Path:
    """Create a ``.tar.gz`` archive containing ``paths``.

    Parameters
    ----------
    output:
        Destination path for the resulting archive.  Parent directories are
        created if necessary.
    paths:
        Sequence of file or directory paths to include in the archive.

    Returns
    -------
    Path
        The path to the created archive.
    """

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    base_dir = Path.cwd().resolve()

    with tarfile.open(output, "w:gz") as tar:
        _add_paths_to_tar(tar, [Path(p) for p in paths], base_dir)

    return output


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Compress project artifacts")
    parser.add_argument(
        "output",
        type=Path,
        help="Destination .tar.gz file",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to include in the archive",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for command line execution."""

    args = _parse_args(argv)
    archive = compress(args.output, args.paths)
    print(f"Created archive {archive}")


if __name__ == "__main__":
    main()