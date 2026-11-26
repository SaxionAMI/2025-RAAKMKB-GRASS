"""Repository consistency checks (no framework needed)."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _framework_tag() -> str:
    m = re.search(r'AudioOpsFramework\.git@([^"]+)"', (REPO / "pyproject.toml").read_text())
    assert m, "pyproject.toml should pin the framework to a git tag"
    return m.group(1)


def test_framework_version_is_the_same_everywhere():
    tag = _framework_tag()
    dockerfile = (REPO / "Dockerfile").read_text()
    compose = (REPO / "docker-compose.yml").read_text()
    assert f"ARG FRAMEWORK_REF={tag}" in dockerfile
    assert set(re.findall(r"FRAMEWORK_REF:-([^}]+)\}", compose)) == {tag}


def test_scripts_are_executable():
    for script in (REPO / "scripts").glob("*.sh"):
        assert script.stat().st_mode & 0o111, f"{script.name} is not executable (chmod +x)"


def test_env_example_has_no_container_hostnames():
    """Local runs read .env too; container addresses belong in docker-compose.yml."""
    env = (REPO / ".env.example").read_text()
    for host in ("http://influxdb:", "http://minio:", "http://mediamtx:"):
        assert host not in env
