"""Install/check the exact Homebrew bottles in environment.lock.json (macOS ARM64)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TAP = "proof-bridge/locked"
sys.path.insert(0, str(ROOT))
from verifier import Rejection, check_environment, sha256_file  # noqa: E402


def fetch_bottle(package: dict, cache: Path) -> Path:
    destination = cache / package["filename"]
    if not destination.exists():
        # Public GHCR bearer token: anonymous package-download scope, no user credentials.
        query = urllib.parse.urlencode({
            "service": "ghcr.io", "scope": f"repository:homebrew/core/{package['name']}:pull",
        })
        with urllib.request.urlopen("https://ghcr.io/token?" + query, timeout=30) as response:
            token = json.load(response)["token"]
        request = urllib.request.Request(package["url"], headers={"Authorization": "Bearer " + token})
        partial = destination.with_suffix(".partial")
        try:
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            if sha256_file(partial) != package["sha256"]:
                raise RuntimeError(f"Bottle checksum mismatch: {package['name']}")
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
    if sha256_file(destination) != package["sha256"]:
        raise RuntimeError(f"Bottle checksum mismatch: {destination}")
    return destination


def prepare_tap(lock: dict, brew: str, environment: dict) -> None:
    """Use a named local tap, respecting Homebrew's ban on file-path installs."""
    repository = Path(subprocess.check_output([brew, "--repository"], text=True, env=environment).strip())
    tap_directory = repository / "Library/Taps/proof-bridge/homebrew-locked"
    # A named local tap is a directory under Taps. Creating it directly avoids
    # `brew tap-new`, which persistently turns on Homebrew developer mode.
    tap_directory.mkdir(parents=True, exist_ok=True)
    formula_directory = tap_directory / "Formula"
    formula_directory.mkdir(exist_ok=True)
    for package in lock["packages"]:
        name = package["name"]
        class_name = "".join(part.capitalize() for part in name.split("-"))
        version, _, revision = package["version"].partition("_")
        rebuild = re.search(r"\.bottle\.(\d+)\.tar", package["filename"])
        recipe = (
            f"class {class_name} < Formula\n"
            f'  desc "Locked Proof Bridge runtime: {name}"\n'
            '  homepage "https://rocq-prover.org/"\n'
            f'  url "{package["url"]}"\n'
            f'  version "{version}"\n'
            f'  sha256 "{package["sha256"]}"\n'
        )
        if revision:
            recipe += f"  revision {int(revision)}\n"
        recipe += '  bottle do\n    root_url "https://ghcr.io/v2/homebrew/core"\n'
        if rebuild:
            recipe += f"    rebuild {int(rebuild.group(1))}\n"
        recipe += f'    sha256 cellar: "{lock["homebrew_prefix"]}/Cellar", arm64_sequoia: "{package["sha256"]}"\n  end\n'
        for dependency in package["dependencies"]:
            recipe += f'  depends_on "{TAP}/{dependency}"\n'
        recipe += '  def install\n    odie "Only the locked binary bottle is supported."\n  end\nend\n'
        destination = formula_directory / f"{name}.rb"
        if destination.exists() and destination.read_text() != recipe:
            raise RuntimeError(f"Refusing to overwrite a different local formula: {destination}")
        destination.write_text(recipe)
    print(f"OK prepared named tap {TAP} with five immutable bottle recipes", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check installed versions and runtime without installing")
    parser.add_argument("--fetch-only", action="store_true", help="Download and verify all locked bottles without installing")
    args = parser.parse_args()
    if args.check and args.fetch_only:
        parser.error("choose either --check or --fetch-only")
    lock = json.loads((ROOT / "environment.lock.json").read_text())
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("This tested lock supports macOS ARM64 only.")
    if int(platform.mac_ver()[0].split(".")[0]) < 15:
        raise RuntimeError("The locked bottles require macOS 15 or newer.")
    brew = shutil.which("brew")
    if brew is None:
        raise RuntimeError("Homebrew at /opt/homebrew is required.")
    environment = dict(os.environ)
    environment.update({
        "HOMEBREW_NO_AUTO_UPDATE": "1", "HOMEBREW_NO_INSTALL_CLEANUP": "1",
        "HOMEBREW_NO_INSTALLED_DEPENDENTS_CHECK": "1", "HOMEBREW_NO_INSTALL_UPGRADE": "1",
    })
    prefix = Path(subprocess.check_output([brew, "--prefix"], text=True, env=environment).strip())
    if str(prefix) != lock["homebrew_prefix"]:
        raise RuntimeError("Homebrew prefix differs from the locked bottle relocation prefix.")
    cache = ROOT / ".cache/bottles"
    if not args.check:
        cache.mkdir(parents=True, exist_ok=True)
    if not args.check and not args.fetch_only:
        prepare_tap(lock, brew, environment)
    for package in lock["packages"]:
        if args.fetch_only:
            print(f"Verified {fetch_bottle(package, cache).name}", flush=True)
            continue
        keg = prefix / "Cellar" / package["name"] / package["version"]
        linked = prefix / "opt" / package["name"]
        if not keg.is_dir():
            if args.check:
                raise RuntimeError(f"Missing {package['name']} {package['version']}")
            if linked.exists():
                raise RuntimeError(f"Different {package['name']} already installed; refusing to replace it.")
            fetch_bottle(package, cache)
            # The named tap fixes both bottle digests and dependency recipes.
            subprocess.run([brew, "install", "--force-bottle", f"{TAP}/{package['name']}"], check=True, env=environment)
        if not linked.exists() or linked.resolve() != keg.resolve():
            raise RuntimeError(f"Expected {package['name']} {package['version']} to be linked; no automatic relinking.")
        if not args.check:
            subprocess.run([brew, "pin", package["name"]], check=True, env=environment)
        print(f"OK {package['name']} {package['version']}", flush=True)
    if not args.fetch_only:
        with tempfile.TemporaryDirectory(prefix="proof-bridge-setup-") as temporary:
            check_environment(Path(temporary))
        print("OK compiler, prelude, and runtime fingerprints match environment.lock.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, Rejection, subprocess.SubprocessError) as error:
        print(f"Setup failed: {error}", file=sys.stderr)
        raise SystemExit(1)
