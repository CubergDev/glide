"""Packaging and supply-chain checks on pyproject.toml and uv.lock. Offline: both files are only parsed."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
LOCK = tomllib.loads((ROOT / "uv.lock").read_text())

# Pillow releases before 12.3.0 carry published advisories (OSV: 40 records, 14 HIGH; all fixed in 12.3.0).
PILLOW_FIRST_SAFE = Version("12.3.0")
# Releases PyPI marks as yanked. A yanked pin still resolves for people who ask for exactly it, so it is easy to miss:
# add a version here when PyPI yanks it. (The offline suite cannot ask PyPI; the lock refresh in a release can.)
YANKED_BUILD_BACKENDS = {"hatchling": {"1.30.0", "1.32.1"}}
EXTRAS = PROJECT["project"]["optional-dependencies"]


def requirements(lines):
    return [Requirement(line) for line in lines]


def pinned_version(requirement):
    specs = list(requirement.specifier)
    assert len(specs) == 1 and specs[0].operator == "==", f"{requirement} must be pinned with =="
    return specs[0].version


def test_pillow_range_admits_no_vulnerable_release():
    """F1: the range used to be >=11.0,<13, which admits eight vulnerable releases."""
    (pillow,) = [r for r in requirements(PROJECT["project"]["dependencies"]) if r.name == "pillow"]
    for vulnerable in ("11.0.0", "11.3.0", "12.0.0", "12.2.0"):
        assert not pillow.specifier.contains(vulnerable), f"pillow {vulnerable} must be excluded by {pillow.specifier}"
    assert pillow.specifier.contains(PILLOW_FIRST_SAFE)
    assert not pillow.specifier.contains("13.0.0"), "the next major stays excluded until someone checks it"


def test_locked_pillow_is_a_fixed_release():
    (pillow,) = [p for p in LOCK["package"] if p["name"] == "pillow"]
    assert Version(pillow["version"]) >= PILLOW_FIRST_SAFE


def test_build_backend_pin_is_not_a_yanked_release():
    """F2: hatchling==1.32.1 was yanked on PyPI."""
    for requirement in requirements(PROJECT["build-system"]["requires"]):
        assert pinned_version(requirement) not in YANKED_BUILD_BACKENDS.get(requirement.name, set()), (
            f"{requirement} is yanked on PyPI"
        )


def test_wheel_still_ships_exactly_the_glide_package():
    assert PROJECT["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["glide"]
