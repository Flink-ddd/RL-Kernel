# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""User-facing launcher for the Qwen3 VIME TP4/CP2 reproduction.

The experiment itself remains implemented by the example runners.  This module
only resolves the profile, performs actionable preflight checks, and delegates
to those runners so the long-form reproduction runbook stays an audit escape
hatch rather than the normal user interface.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


EXAMPLE_RELATIVE = Path("examples/vime_qwen3_8b_tp4_cp2_200")
DEFAULT_PROFILE = EXAMPLE_RELATIVE / "profiles/qwen3-8b-tp4-cp2.json"
ARM_ALIASES = {
    "g00": "native",
    "g01": "consistency",
    "native": "native",
    "consistency": "consistency",
}


class ReproError(RuntimeError):
    """An actionable user-facing reproduction error."""


def canonical_arm(value: str) -> str:
    """Map a user mode to the descriptive run name."""
    canonical = ARM_ALIASES.get(value.strip().lower())
    if canonical is None:
        raise ReproError(f"unknown mode {value!r}; choose native or consistency")
    return canonical


@dataclass(frozen=True)
class Paths:
    workspace: Path
    rl_kernel_root: Path
    vime_root: Path
    megatron_root: Path
    runtime_root: Path
    cuda_runtime_root: Path
    runtime_site: Path
    cuda_python_site: Path
    te_root: Path
    data_root: Path
    model_root: Path
    ref_load: Path
    prompt_data: Path
    output_root: Path
    python: Path
    ray: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_profile(value: str | None) -> Path:
    if value:
        candidate = Path(value).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        repo_candidate = _repo_root() / candidate
        if repo_candidate.is_file():
            return repo_candidate.resolve()
        raise ReproError(f"profile does not exist: {candidate}")
    candidate = _repo_root() / DEFAULT_PROFILE
    if not candidate.is_file():
        raise ReproError(f"default profile is missing: {candidate}")
    return candidate


def _load_profile(value: str | None) -> tuple[dict[str, Any], Path]:
    path = _find_profile(value)
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReproError(f"invalid JSON profile {path}: {exc}") from exc
    if not isinstance(profile, dict):
        raise ReproError(f"profile must be a JSON object: {path}")
    if profile.get("schema_version") != "rlkernel.repro.profile.v1":
        raise ReproError(f"unsupported profile schema in {path}")
    return profile, path


def _path(value: str | Path | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value).expanduser().resolve()


def _profile_path(profile: dict[str, Any], key: str) -> str | None:
    paths = profile.get("paths", {})
    value = paths.get(key) if isinstance(paths, dict) else None
    return str(value) if value else None


def _override_or_env(value: str | None, env_name: str) -> str | None:
    return value or os.environ.get(env_name)


def _required_path(value: Path | None, label: str) -> Path:
    if value is None:
        raise ReproError(
            f"{label} is not configured; pass the corresponding --{label.replace('_', '-')} "
            "option or set the documented RLK_REPRO_* environment variable"
        )
    return value


def resolve_paths(
    profile: dict[str, Any],
    *,
    workspace: str | None,
    arm: str,
    rl_kernel_root: str | None = None,
    vime_root: str | None = None,
    megatron_root: str | None = None,
    runtime_root: str | None = None,
    cuda_runtime_root: str | None = None,
    runtime_site: str | None = None,
    cuda_python_site: str | None = None,
    te_root: str | None = None,
    data_root: str | None = None,
    model_root: str | None = None,
    ref_load: str | None = None,
    prompt_data: str | None = None,
    output_root: str | None = None,
    python: str | None = None,
    ray: str | None = None,
) -> Paths:
    arm = canonical_arm(arm)
    workspace_value = _override_or_env(
        workspace or _profile_path(profile, "workspace"), "RLK_REPRO_WORKSPACE"
    )
    workspace_path = _required_path(_path(workspace_value), "workspace")

    def choose(value: str | None, profile_key: str, env_name: str, default: Path | None) -> Path:
        selected = _override_or_env(value or _profile_path(profile, profile_key), env_name)
        return _path(selected) or _required_path(default, profile_key)

    runtime_path = choose(
        runtime_root, "runtime_root", "RLK_REPRO_RUNTIME_ROOT", workspace_path / "runtime"
    )
    data_path = choose(data_root, "data_root", "RLK_REPRO_DATA_ROOT", workspace_path / "data")
    output_path = choose(
        output_root,
        "output_root",
        "RLK_REPRO_OUTPUT_ROOT",
        data_path / "runs" / "convergence",
    )
    paths = Paths(
        workspace=workspace_path,
        rl_kernel_root=choose(
            rl_kernel_root,
            "rl_kernel_root",
            "RLK_REPRO_RL_KERNEL_ROOT",
            workspace_path / "RL-Kernel",
        ),
        vime_root=choose(vime_root, "vime_root", "RLK_REPRO_VIME_ROOT", workspace_path / "vime"),
        megatron_root=choose(
            megatron_root,
            "megatron_root",
            "RLK_REPRO_MEGATRON_ROOT",
            workspace_path / "Megatron-LM",
        ),
        runtime_root=runtime_path,
        cuda_runtime_root=choose(
            cuda_runtime_root,
            "cuda_runtime_root",
            "RLK_REPRO_CUDA_RUNTIME_ROOT",
            runtime_path / "lib/python3.11/site-packages/nvidia/cuda_runtime",
        ),
        runtime_site=choose(
            runtime_site,
            "runtime_site",
            "RLK_REPRO_RUNTIME_SITE",
            workspace_path / ".runtime-site-packages",
        ),
        cuda_python_site=choose(
            cuda_python_site,
            "cuda_python_site",
            "RLK_REPRO_CUDA_PYTHON_SITE",
            workspace_path / ".cuda-python-12.8-site-packages",
        ),
        te_root=(
            _path(_override_or_env(te_root, "RLK_REPRO_TE_ROOT"))
            or _path(_profile_path(profile, f"te_root_{arm.lower()}"))
            or _path(
                os.environ.get(str(profile.get("modes", {}).get(arm, {}).get("te_root_env", "")))
            )
            or workspace_path / f".te-root-{arm.lower()}"
        ),
        data_root=data_path,
        model_root=choose(
            model_root,
            "model_root",
            "RLK_REPRO_MODEL_ROOT",
            workspace_path / "checkpoints/Qwen3-8B_vime_rlkernel_tp2_cp2",
        ),
        ref_load=choose(
            ref_load,
            "ref_load",
            "RLK_REPRO_REF_LOAD",
            workspace_path / "checkpoints/Qwen3-8B_torch_dist",
        ),
        prompt_data=choose(
            prompt_data,
            "prompt_data",
            "RLK_REPRO_PROMPT_DATA",
            data_path / "datasets/dapo-math-17k.vime.jsonl",
        ),
        output_root=output_path,
        python=choose(python, "python", "RLK_REPRO_PYTHON", runtime_path / "bin/python3.11"),
        ray=choose(ray, "ray", "RLK_REPRO_RAY", runtime_path / "bin/ray"),
    )
    return paths


def _run(
    command: Sequence[str],
    *,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, env=env)


def _git_clean(path: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode == 0 and not result.stdout.strip()


def _gpu_count() -> int | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    result = subprocess.run(
        [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return None
    return len([line for line in result.stdout.splitlines() if line.strip()])


def _python_versions(python: Path) -> dict[str, str]:
    script = (
        "import importlib.metadata as m, platform, json; "
        "names=('torch','vllm','ray','sympy','pylatexenc'); "
        "out={'python': platform.python_version()}; "
        "out.update({n: (m.version(n) if m.packages_distributions().get(n) else 'missing') "
        "for n in names}); "
        "print(json.dumps(out, sort_keys=True))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", script], check=False, capture_output=True, text=True
        )
    except OSError as exc:
        return {"error": str(exc)}
    if result.returncode:
        return {"error": result.stderr.strip() or f"exit {result.returncode}"}
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": result.stdout.strip()}
    return value if isinstance(value, dict) else {"error": "invalid version output"}


def doctor(paths: Paths, profile: dict[str, Any], *, as_json: bool = False) -> int:
    expected = int(profile.get("requirements", {}).get("gpus", 8))
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    gpu_count = _gpu_count()
    add(
        "gpu",
        gpu_count == expected,
        f"found {gpu_count if gpu_count is not None else 'unavailable'}, expected {expected}",
    )
    for name, path in asdict(paths).items():
        if name in {"workspace", "data_root", "output_root"}:
            continue
        exists = path.exists()
        add(name, exists, str(path) if exists else f"missing: {path}")
    for name, path in (
        ("rl_kernel_git", paths.rl_kernel_root),
        ("vime_git", paths.vime_root),
        ("megatron_git", paths.megatron_root),
    ):
        add(name, (path / ".git").exists(), f"{path}/.git")
    for name, path in (
        ("model", paths.model_root),
        ("reference_checkpoint", paths.ref_load),
        ("prompt_data", paths.prompt_data),
    ):
        add(name, path.exists(), str(path) if path.exists() else f"missing: {path}")
    add(
        "runtime_versions",
        paths.python.is_file(),
        json.dumps(_python_versions(paths.python), sort_keys=True),
    )

    payload = {
        "profile": profile.get("name"),
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Profile: {payload['profile']}")
        for item in checks:
            marker = "PASS" if item["passed"] else "FAIL"
            print(f"[{marker}] {item['name']}: {item['detail']}")
        print("Doctor: PASS" if payload["passed"] else "Doctor: FAIL")
    return 0 if payload["passed"] else 1


def _arm_config(profile: dict[str, Any], arm: str) -> dict[str, Any]:
    arm = canonical_arm(arm)
    arms = profile.get("modes", {})
    config = arms.get(arm)
    if not isinstance(config, dict):
        raise ReproError(f"unsupported mode {arm!r}; choose one of {', '.join(sorted(arms))}")
    return config


def _example_root(rl_kernel_root: Path | None = None) -> Path:
    root = (rl_kernel_root or _repo_root()) / EXAMPLE_RELATIVE
    if not root.is_dir():
        raise ReproError(f"reproduction example is missing: {root}")
    return root


def _example_root_for_run(run_dir: Path) -> Path:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            root_value = manifest.get("paths", {}).get("rl_kernel_root")
            if root_value:
                return _example_root(Path(root_value))
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    return _example_root()


def _runner_command(paths: Paths, profile: dict[str, Any], args: argparse.Namespace) -> list[str]:
    arm = canonical_arm(args.arm)
    _arm_config(profile, arm)
    if (paths.rl_kernel_root / EXAMPLE_RELATIVE).is_dir():
        example_root = _example_root(paths.rl_kernel_root)
    elif getattr(args, "command", None) == "plan":
        example_root = _example_root()
    else:
        example_root = _example_root(paths.rl_kernel_root)
    command = [
        str(paths.python),
        str(example_root / "run_arm.py"),
        "--group",
        arm,
        "--num-rollout",
        str(args.rollouts),
        "--seed",
        str(args.seed),
        "--rollout-seed",
        str(args.rollout_seed),
        "--output-root",
        str(paths.output_root),
        "--rl-kernel-root",
        str(paths.rl_kernel_root),
        "--vime-root",
        str(paths.vime_root),
        "--megatron-root",
        str(paths.megatron_root),
        "--model-root",
        str(paths.model_root),
        "--ref-load",
        str(paths.ref_load),
        "--prompt-data",
        str(paths.prompt_data),
        "--python",
        str(paths.python),
        "--ray-bin",
        str(paths.ray),
        "--extra-pythonpath",
        str(paths.runtime_site),
        "--extra-pythonpath",
        str(paths.cuda_python_site),
        "--extra-pythonpath",
        str(paths.te_root),
        "--ld-library-path",
        f"{paths.cuda_runtime_root / 'lib'}:{paths.te_root / 'transformer_engine/wheel_lib'}",
    ]
    if args.run_id:
        command.extend(["--run-id", args.run_id])
    if args.wait:
        command.append("--wait")
    if args.allow_dirty:
        command.append("--allow-dirty")
    if args.dry_run:
        command.append("--dry-run")
    for item in profile.get("runner_args", []):
        command.extend([str(part) for part in item])
    return command


def _print_plan(paths: Paths, profile: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    command = _runner_command(paths, profile, args)
    plan = {
        "schema_version": "rlkernel.repro.plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile.get("name"),
        "mode": canonical_arm(args.arm),
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "runner_command": command,
    }
    print(json.dumps(plan, indent=2))
    return plan


def _prepare_repositories(paths: Paths, profile: dict[str, Any], arm: str) -> None:
    repos = profile.get("repositories", {})
    mode = canonical_arm(arm)
    for key, target in (
        ("rl_kernel", paths.rl_kernel_root),
        ("vime", paths.vime_root),
        ("megatron", paths.megatron_root),
    ):
        config = repos.get(key, {})
        url = config.get("url")
        if not target.exists():
            if not url:
                raise ReproError(f"cannot clone {key}: profile has no repository URL")
            target.parent.mkdir(parents=True, exist_ok=True)
            _run(["git", "clone", str(url), str(target)])
        if not (target / ".git").exists():
            raise ReproError(f"{key} is not a git checkout: {target}")
        if not _git_clean(target):
            raise ReproError(
                f"{key} has local changes; commit or clean it before prepare: {target}"
            )
        revision = config.get("revisions", {}).get(mode)
        if revision:
            _run(["git", "-C", str(target), "fetch", "origin"])
            _run(["git", "-C", str(target), "checkout", "--detach", str(revision)])


def _convert_checkpoint(paths: Paths) -> None:
    if paths.ref_load.exists():
        print(f"Reference checkpoint already exists: {paths.ref_load}")
        return
    model_script = paths.vime_root / "scripts/models/qwen3-8B.sh"
    converter = paths.vime_root / "tools/convert_hf_to_torch_dist.py"
    if not model_script.is_file():
        raise ReproError(f"VIME model argument script is missing: {model_script}")
    if not converter.is_file():
        raise ReproError(f"checkpoint converter is missing: {converter}")
    if not paths.model_root.is_dir():
        raise ReproError(f"HF model root is missing: {paths.model_root}")
    paths.ref_load.parent.mkdir(parents=True, exist_ok=True)
    bash = shutil.which("bash")
    if not bash:
        raise ReproError("bash is required for checkpoint conversion on the training host")
    command = (
        "set -euo pipefail; "
        f"source {shlex.quote(str(model_script))}; "
        f"PYTHONPATH={shlex.quote(str(paths.megatron_root))} "
        f"{shlex.quote(str(paths.python))} {shlex.quote(str(converter))} "
        '"${MODEL_ARGS[@]}" '
        f"--hf-checkpoint {shlex.quote(str(paths.model_root))} "
        f"--save {shlex.quote(str(paths.ref_load))}"
    )
    _run([bash, "-lc", command])


def prepare(paths: Paths, profile: dict[str, Any], args: argparse.Namespace) -> int:
    paths.data_root.mkdir(parents=True, exist_ok=True)
    paths.output_root.mkdir(parents=True, exist_ok=True)
    _prepare_repositories(paths, profile, canonical_arm(args.arm))
    if args.download_data:
        source = paths.data_root / "datasets/dapo-math-17k.parquet"
        paths.prompt_data.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                str(paths.python),
                str(_example_root(paths.rl_kernel_root) / "prepare_dapo_data.py"),
                "--download",
                "--source",
                str(source),
                "--output",
                str(paths.prompt_data),
            ]
        )
    if args.convert_checkpoint:
        _convert_checkpoint(paths)
    print(f"Prepared workspace: {paths.workspace}")
    print(f"Next: rlk-repro doctor --workspace {paths.workspace}")
    return 0


def _add_path_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default=None, help="profile JSON path")
    parser.add_argument("--workspace", default=None, help="reproduction workspace root")
    for name, label in (
        ("runtime-root", "runtime environment root"),
        ("cuda-runtime-root", "CUDA runtime package root"),
        ("runtime-site", "additional runtime site-packages"),
        ("cuda-python-site", "CUDA Python site-packages"),
        ("te-root", "Transformer Engine site-packages root"),
        ("data-root", "data root"),
        ("model-root", "HF model root"),
        ("ref-load", "Megatron torch-dist reference checkpoint"),
        ("prompt-data", "prepared prompt JSONL"),
        ("output-root", "append-only run output root"),
        ("python", "experiment Python executable"),
        ("ray", "Ray executable"),
        ("rl-kernel-root", "RL-Kernel checkout"),
        ("vime-root", "VIME checkout"),
        ("megatron-root", "Megatron-LM checkout"),
    ):
        parser.add_argument(f"--{name}", default=None, help=label)


def _resolved_paths(profile: dict[str, Any], args: argparse.Namespace) -> Paths:
    values = vars(args)
    return resolve_paths(
        profile,
        workspace=values.get("workspace"),
        arm=canonical_arm(str(values.get("arm", "native"))),
        **{key.replace("-", "_"): values.get(key.replace("-", "_")) for key in (
            "rl-kernel-root",
            "vime-root",
            "megatron-root",
            "runtime-root",
            "cuda-runtime-root",
            "runtime-site",
            "cuda-python-site",
            "te-root",
            "data-root",
            "model-root",
            "ref-load",
            "prompt-data",
            "output-root",
            "python",
            "ray",
        )},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlk-repro", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor", help="check host, runtime, assets, and topology"
    )
    _add_path_options(doctor_parser)
    doctor_parser.add_argument(
        "--mode",
        default="native",
        dest="arm",
        metavar="MODE",
        help="native (no rollout-logprob reuse) or consistency (RL-Kernel operators)",
    )
    doctor_parser.add_argument(
        "--arm", dest="arm", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    doctor_parser.add_argument("--json", action="store_true", dest="as_json")

    prepare_parser = subparsers.add_parser(
        "prepare", help="clone/pin sources and optionally prepare data"
    )
    _add_path_options(prepare_parser)
    prepare_parser.add_argument(
        "--mode",
        default="native",
        dest="arm",
        metavar="MODE",
        help="native (no rollout-logprob reuse) or consistency (RL-Kernel operators)",
    )
    prepare_parser.add_argument(
        "--arm", dest="arm", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    prepare_parser.add_argument("--download-data", action="store_true")
    prepare_parser.add_argument("--convert-checkpoint", action="store_true")

    for command in ("plan", "run"):
        command_parser = subparsers.add_parser(command, help=f"{command} one reproduction mode")
        _add_path_options(command_parser)
        command_parser.add_argument(
            "--mode",
            default="native",
            dest="arm",
            metavar="MODE",
            help="native (no rollout-logprob reuse) or consistency (RL-Kernel operators)",
        )
        command_parser.add_argument(
            "--arm", dest="arm", default=argparse.SUPPRESS, help=argparse.SUPPRESS
        )
        command_parser.add_argument("--rollouts", type=int, default=200)
        command_parser.add_argument("--seed", type=int, default=1234)
        command_parser.add_argument("--rollout-seed", type=int, default=1234)
        command_parser.add_argument("--run-id", default=None)
        command_parser.add_argument("--wait", action="store_true")
        command_parser.add_argument("--allow-dirty", action="store_true")
        command_parser.add_argument("--dry-run", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="validate one completed run")
    validate_parser.add_argument("--run-dir", type=Path, required=True)
    validate_parser.add_argument("--seal", action="store_true")
    validate_parser.add_argument("--profile", default=None)

    report_parser = subparsers.add_parser("report", help="collect and plot validated runs")
    _add_path_options(report_parser)
    report_parser.add_argument("--phase", default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        profile, _profile_path_value = _load_profile(getattr(args, "profile", None))
        if args.command == "validate":
            run_dir = args.run_dir.expanduser().resolve()
            command = [
                sys.executable,
                str(_example_root_for_run(run_dir) / "validate_run.py"),
                "--run-dir",
                str(run_dir),
            ]
            if args.seal:
                command.append("--seal")
            return _run(command).returncode

        paths = _resolved_paths(profile, args)
        if args.command == "doctor":
            return doctor(paths, profile, as_json=args.as_json)
        if args.command == "prepare":
            return prepare(paths, profile, args)
        if args.command == "plan":
            _print_plan(paths, profile, args)
            return 0
        if args.command == "run":
            preflight = doctor(paths, profile)
            if preflight:
                print("Run aborted: fix the failed doctor checks first.", file=sys.stderr)
                return preflight
            command = _runner_command(paths, profile, args)
            print("Executing:", " ".join(command))
            env = os.environ.copy()
            env["CUDNN_FRONTEND_CUDART_LIB_NAME"] = str(
                paths.cuda_runtime_root / "lib/libcudart.so.12"
            )
            return _run(command, env=env).returncode
        if args.command == "report":
            results_root = paths.data_root / "results"
            _run(
                [
                    str(paths.python),
                    str(_example_root(paths.rl_kernel_root) / "collect_results.py"),
                    "--runs-root",
                    str(paths.data_root / "runs"),
                    "--output-dir",
                    str(results_root),
                ]
            )
            plot_command = [
                str(paths.python),
                str(_example_root(paths.rl_kernel_root) / "plot_results.py"),
                "--rounds-csv",
                str(results_root / "rounds.csv"),
                "--output-dir",
                str(results_root / "figures"),
            ]
            if args.phase:
                plot_command.extend(["--phase", args.phase])
            _run(plot_command)
            print(f"Report written to {results_root}")
            return 0
    except (OSError, ReproError, subprocess.CalledProcessError) as exc:
        print(f"rlk-repro: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
