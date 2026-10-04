"""Packaging and supply-chain checks on pyproject.toml and uv.lock. Offline: both files are only parsed."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
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


def test_sdist_ships_only_the_source_and_its_metadata():
    """F3: the sdist used to be the whole repository (tests, handoff notes, a private path, the lock, CI files)."""
    sdist = PROJECT["tool"]["hatch"]["build"]["targets"]["sdist"]
    allowed = {"/glide", "/pyproject.toml", "/glide.toml.example", "/README.md"}
    assert set(sdist["include"]) <= allowed
    assert {"/glide", "/pyproject.toml"} <= set(sdist["include"])
    for entry in sdist["include"]:
        assert (ROOT / entry.lstrip("/")).exists(), f"{entry} is listed for the sdist but missing from the tree"
    if (ROOT / "README.md").exists():
        assert "/README.md" in sdist["include"]


def test_wheel_still_ships_exactly_the_glide_package():
    assert PROJECT["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["glide"]


def test_metadata_has_classifiers_and_blocks_an_accidental_upload():
    """F4: the wheel metadata had no classifiers. "Private :: Do Not Upload" makes PyPI reject an accidental upload."""
    classifiers = set(PROJECT["project"]["classifiers"])
    assert "Private :: Do Not Upload" in classifiers
    assert {"Programming Language :: Python :: 3.12", "Programming Language :: Python :: 3.13"} <= classifiers
    assert "Programming Language :: Python :: 3 :: Only" in classifiers
    # The team decides the license; none is invented here.
    assert not any(c.startswith("License ::") for c in classifiers)
    assert "license" not in PROJECT["project"]


def test_readme_is_declared_once_it_exists():
    if (ROOT / "README.md").exists():
        assert PROJECT["project"]["readme"] == "README.md"
    else:
        assert "readme" not in PROJECT["project"]


def test_extras_name_only_features_with_code_and_all_covers_them():
    """F5: `ocr` (rapidocr, which drags in opencv and fetches models from a third-party host) had no code at all."""
    assert "ocr" not in EXTRAS
    names = {r.name for lines in EXTRAS.values() for r in requirements(lines)}
    assert "rapidocr" not in names
    (everything,) = [r for r in requirements(EXTRAS["all"])]
    assert set(everything.extras) == set(EXTRAS) - {"all"}


def test_speech_extra_lists_no_package_nothing_imports():
    """F5: `websockets` was listed, but the code speaks websocket-client (a core dependency)."""
    assert "websockets" not in {r.name for r in requirements(EXTRAS["speech"])}
    imported = "\n".join(p.read_text() for p in (ROOT / "glide").rglob("*.py"))
    assert "import websockets" not in imported and "from websockets" not in imported


def test_lock_records_the_same_requirements_as_pyproject():
    """`uv lock --check` is the real check; this one runs offline in every test run."""
    (glide,) = [p for p in LOCK["package"] if p["name"] == "glide"]
    locked = glide["metadata"]["requires-dist"]
    locked_names = {canonicalize_name(r["name"]) for r in locked}
    wanted = {canonicalize_name(r.name) for r in requirements(PROJECT["project"]["dependencies"])}
    wanted |= {canonicalize_name(r.name) for lines in EXTRAS.values() for r in requirements(lines)}
    assert locked_names == wanted
    assert set(glide["metadata"]["provides-extras"]) == set(EXTRAS)


def test_version_is_not_allowed_to_drift_from_pyproject():
    """F7: glide/computer/__init__.py repeats the version. Until that line is deleted, keep the two equal."""
    from glide import computer

    version = getattr(computer, "__version__", None)
    if version is not None:
        assert version == PROJECT["project"]["version"]
