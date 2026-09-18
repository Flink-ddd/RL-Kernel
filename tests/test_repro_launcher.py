from __future__ import annotations

import json
from pathlib import Path

from rl_engine.repro import (
    _arm_config,
    _resolved_paths,
    _runner_command,
    build_parser,
    canonical_arm,
)


def _profile() -> dict:
    profile_path = (
        Path(__file__).parents[1]
        / "examples/vime_qwen3_8b_tp4_cp2_200/profiles/qwen3-8b-tp4-cp2.json"
    )
    return json.loads(profile_path.read_text(encoding="utf-8"))


def _args(tmp_path: Path, mode: str):
    return build_parser().parse_args(
        [
            "plan",
            "--workspace",
            str(tmp_path),
            "--te-root",
            str(tmp_path / "te"),
            "--mode",
            mode,
        ]
    )


def test_user_modes_map_to_operator_arms():
    assert canonical_arm("native") == "G00"
    assert canonical_arm("consistency") == "G01"
    assert _arm_config(_profile(), "native")["te_root_env"] == "TE218_ROOT"
    assert _arm_config(_profile(), "consistency")["te_root_env"] == "TE218_ROOT"


def test_consistency_command_does_not_enable_rollout_logprob_reuse(tmp_path: Path):
    profile = _profile()
    args = _args(tmp_path, "consistency")
    paths = _resolved_paths(profile, args)
    command = _runner_command(paths, profile, args)

    assert command[command.index("--group") + 1] == "G01"
    assert "--use-rollout-logprobs" not in command


def test_native_command_uses_production_operator_route(tmp_path: Path):
    profile = _profile()
    args = _args(tmp_path, "native")
    paths = _resolved_paths(profile, args)
    command = _runner_command(paths, profile, args)

    assert command[command.index("--group") + 1] == "G00"
    assert "--use-rollout-logprobs" not in command
