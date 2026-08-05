"""Points 6 and 13: build the downloadable artefact with one command.

    python scripts/build_release_artifact.py --version v1.0.0

Produces, under release/:
  * a wheel and a source distribution
  * a .sha256 next to each, so a download can be verified

Deliberately not a frozen executable. This project depends on torch and
transformers; a PyInstaller bundle would be several gigabytes, take most of an
hour to build on a hosted runner, and still need the user to supply model
weights. A wheel is the artefact that actually helps someone: `pip install` it
and the package and its console entry point are there.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RELEASE = ROOT / "release"
DIST = ROOT / "dist"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_distributions() -> list[Path]:
    probe = subprocess.run(
        [sys.executable, "-m", "build", "--version"], capture_output=True, text=True, check=False
    )
    if probe.returncode != 0:
        raise SystemExit('the build backend is missing. Run: pip install -e ".[release]"')
    print(probe.stdout.strip())

    if DIST.exists():
        shutil.rmtree(DIST)
    subprocess.run([sys.executable, "-m", "build"], check=True, cwd=ROOT)

    produced = sorted(DIST.glob("*.whl")) + sorted(DIST.glob("*.tar.gz"))
    if not produced:
        raise SystemExit("python -m build produced neither a wheel nor an sdist")
    return produced


def verify_wheel(wheel: Path) -> None:
    """Install the wheel into a throwaway environment and import the package.

    A wheel that builds but cannot be imported is worse than no wheel: it fails
    for the person who downloaded it rather than for the person who shipped it.
    """
    venv = ROOT / "build" / "verify-venv"
    if venv.exists():
        shutil.rmtree(venv)
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    python = venv / ("Scripts" if sys.platform == "win32" else "bin") / "python"

    # --no-deps: torch alone is over two gigabytes, and the question here is
    # whether the archive is well-formed, not whether PyPI is reachable.
    subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", "--no-deps", str(wheel)], check=True
    )
    result = subprocess.run(
        [str(python), "-c", "import xray_classifier; print(xray_classifier.__name__)"],
        capture_output=True,
        text=True,
        check=False,
    )
    shutil.rmtree(venv, ignore_errors=True)
    if result.returncode != 0:
        raise SystemExit(f"the wheel installs but does not import:\n{result.stderr.strip()}")
    print(f"the wheel installs and imports as {result.stdout.strip()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default="v0.0.0-dev", help="version tag, for the log only")
    parser.add_argument("--skip-verify", action="store_true", help="do not install the wheel")
    args = parser.parse_args()

    print(f"building {args.version}")
    if RELEASE.exists():
        shutil.rmtree(RELEASE)
    RELEASE.mkdir(parents=True)

    produced: list[Path] = []
    for artefact in build_distributions():
        target = RELEASE / artefact.name
        shutil.copy(artefact, target)
        produced.append(target)

    wheels = [p for p in produced if p.suffix == ".whl"]
    if wheels and not args.skip_verify:
        verify_wheel(wheels[0])

    for path in list(produced):
        checksum = RELEASE / f"{path.name}.sha256"
        checksum.write_text(f"{sha256_of(path)}  {path.name}\n", encoding="utf-8")
        produced.append(checksum)

    print("\nrelease/")
    for path in sorted(RELEASE.iterdir()):
        print(f"  {path.name:<52} {path.stat().st_size / 1024:>9.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
