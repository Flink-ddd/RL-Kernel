"""Run one P/P or R/R arm of the PR377 Qwen3-8B TP4/CP2 workload on ROCm.

The P/P arm selects production attention, FFN and logp on both frameworks and
uses Vime's rollout-logprob consistency mode.  The R/R arm selects the strict
RL-Kernel route on both frameworks and validates bitwise train/rollout
agreement.  Paths and the round count are CLI arguments so the same script
serves every machine layout; nothing is hard-coded.

Example::

    python -m examples.vime_rocm_attention_ablation.run_pr377_workload \
      --case R/R --num-rollout 3 \
      --run-dir /app/model/vime-runs/mfma-rr-3round \
      --rl-kernel-root /work/RL-Kernel --vime-root /work/vime \
      --megatron-root /work/Megatron-LM-vime
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from examples.vime_rocm_attention_ablation.run import (
    MatrixConfig,
    _canonical_fingerprint,
    _prepare_run_dir,
    build_arm_environment,
    frozen_input_manifest,
    public_arm_environment,
)
from examples.vime_rocm_attention_ablation.validate_artifacts import (
    CASE_IMPLEMENTATIONS,
    compare_train_rollout_logps,
    load_readbacks,
    load_rollout_identity,
    validate_arm,
    write_report,
)

# Strict-path knobs forwarded into the Ray runtime environment when set.
FORWARDED_STRICT_ENVIRONMENT = (
    "RL_KERNEL_DET_GEMM_BACKEND",
    "RL_KERNEL_ROCM_FIXED_PAGED_TILE",
    "RL_KERNEL_ROCM_PAGED_KV_MAX_TOKENS",
    "RL_KERNEL_ROCM_ATTENTION_BACKEND",
)


class WorkloadConfig(MatrixConfig):
    case_id: str = "R/R"

    def frozen_parameters(self):
        value = super().frozen_parameters()
        value["ffn_case"] = self.case_id
        value["logp_case"] = self.case_id
        if self.case_id == "P/P":
            value["framework_consistency"] = {
                "use_rollout_logprobs": True,
                "get_mismatch_metrics": True,
                "custom_tis_function": (
                    "vime_rocm_attention_ablation.tis_metrics.metrics_only_tis"
                ),
            }
        return value


def sealed_manifest(config: MatrixConfig):
    value = frozen_input_manifest(config)
    value["fingerprint"] = _canonical_fingerprint(
        {key: item for key, item in value.items() if key != "fingerprint"}
    )
    return value


def validate_native_readbacks(readback_dir: Path):
    errors = []
    frameworks = set()
    paths = []
    for record in load_readbacks(readback_dir):
        paths.append(record["_path"])
        framework = record.get("framework")
        if framework in {"megatron", "vllm"}:
            frameworks.add(framework)
        if record.get("fallbacks"):
            errors.append(f"{record['_path']}: unexpected adapter fallback")
        operators = record.get("operators", {})
        for module in ("attention", "ffn", "logp"):
            operator = operators.get(module)
            if not isinstance(operator, dict):
                errors.append(f"{record['_path']}: missing {module} readback")
                continue
            if operator.get("case_id") != "P/P":
                errors.append(f"{record['_path']}: {module} case is {operator.get('case_id')!r}")
            if operator.get("implementation") != "production":
                errors.append(
                    f"{record['_path']}: {module} implementation is "
                    f"{operator.get('implementation')!r}"
                )
    if frameworks != {"megatron", "vllm"}:
        errors.append(f"readback frameworks are {sorted(frameworks)!r}")
    return {"passed": not errors, "errors": errors, "paths": paths}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["P/P", "R/R"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-rollout", type=int, default=3)
    parser.add_argument("--rl-kernel-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--vime-root", type=Path, default=Path(os.environ.get("VIME_ROOT", "/work/vime"))
    )
    parser.add_argument(
        "--megatron-root",
        type=Path,
        default=Path(os.environ.get("MEGATRON_ROOT", "/work/Megatron-LM-vime")),
    )
    parser.add_argument("--model-root", type=Path, default=Path("/app/model/Qwen3-8B"))
    parser.add_argument(
        "--reference-checkpoint", type=Path, default=Path("/app/model/Qwen3-8B_torch_dist")
    )
    parser.add_argument(
        "--prompt-data", type=Path, default=Path("/app/model/dapo-math-17k/dapo-math-17k.jsonl")
    )
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--max-response-length", type=int, default=7168)
    parser.add_argument("--max-tokens-per-gpu", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rollout-seed", type=int, default=1234)
    parser.add_argument("--fixed-paged-tile", default="128")
    parser.add_argument("--paged-kv-max-tokens", default="8192")
    parser.add_argument("--vllm-gpu-memory-utilization", default="0.38")
    parser.add_argument("--ray-port", type=int, default=6385)
    parser.add_argument("--ray-dashboard-port", type=int, default=28265)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    case_id = args.case
    root = args.rl_kernel_root.resolve()
    config = WorkloadConfig(
        vime_root=args.vime_root,
        rl_kernel_root=root,
        megatron_root=args.megatron_root,
        model_root=args.model_root,
        reference_checkpoint=args.reference_checkpoint,
        prompt_data=args.prompt_data,
        run_dir=args.run_dir,
        launcher=root / "examples/vime_rocm_attention_ablation/launch_arm.sh",
        num_rollout=args.num_rollout,
        rollout_batch_size=1,
        samples_per_prompt=args.samples_per_prompt,
        global_batch_size=args.global_batch_size,
        max_response_length=args.max_response_length,
        max_tokens_per_gpu=args.max_tokens_per_gpu,
        seed=args.seed,
        rollout_seed=args.rollout_seed,
        ray_port=args.ray_port,
        ray_dashboard_port=args.ray_dashboard_port,
    )
    config.case_id = case_id
    config.validate(require_paths=True)
    _prepare_run_dir(args.run_dir)
    frozen_before = sealed_manifest(config)
    write_report(args.run_dir / "frozen-inputs.before.json", frozen_before)

    arm_slug = case_id.lower().replace("/", "-")
    arm_dir = args.run_dir / "arms" / arm_slug
    for directory in (
        arm_dir / "readbacks",
        arm_dir / "dump",
        arm_dir / "checkpoint",
        arm_dir / "mismatch_sidecars",
    ):
        directory.mkdir(parents=True, exist_ok=False)

    environment = build_arm_environment(
        config, case_id, arm_dir, arm_index=0 if case_id == "P/P" else 3
    )
    environment.update(
        {
            "RL_KERNEL_ATTENTION_CASE": case_id,
            "RL_KERNEL_FFN_CASE": case_id,
            "RL_KERNEL_LOGP_CASE": case_id,
            "RL_KERNEL_ROCM_FIXED_PAGED_TILE": args.fixed_paged_tile,
            "RL_KERNEL_ROCM_PAGED_KV_MAX_TOKENS": args.paged_kv_max_tokens,
            "VLLM_GPU_MEMORY_UTILIZATION": args.vllm_gpu_memory_utilization,
        }
    )
    if case_id == "P/P":
        environment["RLK_ABLATION_USE_ROLLOUT_LOGPROBS"] = "1"
    launch = {
        "schema_version": "rlkernel.vime_rocm_attention_arm_launch.v1",
        "case_id": case_id,
        "expected_implementations": CASE_IMPLEMENTATIONS[case_id],
        "framework_consistency": (
            {"use_rollout_logprobs": True, "mismatch_metrics_recompute": True}
            if case_id == "P/P"
            else {"use_rollout_logprobs": False, "strict_linear_logp": True}
        ),
        "strict_environment": {
            name: environment[name] for name in FORWARDED_STRICT_ENVIRONMENT if name in environment
        },
        "frozen_input_fingerprint": frozen_before["fingerprint"],
        "command": ["bash", str(config.launcher.resolve())],
        "environment": public_arm_environment(environment),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_report(arm_dir / "launch.json", launch)

    with (arm_dir / "launcher.log").open("w", encoding="utf-8") as log_handle:
        process = subprocess.run(
            ["bash", str(config.launcher.resolve())],
            cwd=config.rl_kernel_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if case_id == "R/R":
        report = validate_arm(arm_dir, case_id, launcher_returncode=process.returncode)
    else:
        errors = []
        if process.returncode:
            errors.append(f"Vime launcher exited with status {process.returncode}")
        try:
            readbacks = validate_native_readbacks(arm_dir / "readbacks")
        except Exception as exc:  # pragma: no cover - runtime evidence failure
            readbacks = {"passed": False, "errors": [str(exc)], "paths": []}
        errors.extend(readbacks["errors"])
        rollout_identity = load_rollout_identity(arm_dir / "dump" / "rollout_data")
        errors.extend(rollout_identity["errors"])
        try:
            metrics = compare_train_rollout_logps(
                arm_dir / "mismatch_sidecars",
                require_exact=False,
                tensor_parallel_size=config.tensor_parallel_size,
                context_parallel_size=config.context_parallel_size,
            )
        except Exception as exc:  # pragma: no cover - runtime evidence failure
            metrics = {"passed": False, "errors": [str(exc)]}
        errors.extend(metrics["errors"])
        report = {
            "case_id": case_id,
            "launcher_returncode": process.returncode,
            "passed": not errors,
            "errors": errors,
            "readbacks": readbacks,
            "rollout_identity": rollout_identity,
            "metrics": metrics,
        }
    frozen_after = sealed_manifest(config)
    write_report(args.run_dir / "frozen-inputs.after.json", frozen_after)
    frozen_match = frozen_before["fingerprint"] == frozen_after["fingerprint"]
    if not frozen_match:
        report["errors"] = list(report.get("errors", [])) + [
            "frozen source fingerprint changed during the run"
        ]
        report["passed"] = False
    report["frozen_sources_match"] = frozen_match
    report["run_dir"] = str(args.run_dir)
    report["num_rollout"] = args.num_rollout
    write_report(arm_dir / "validation.json", report)
    summary = {
        "run_dir": str(args.run_dir),
        "case_id": case_id,
        "num_rollout": args.num_rollout,
        "launcher_returncode": process.returncode,
        "passed": report["passed"],
        "errors": report["errors"],
        "metrics": report.get("metrics"),
        "strict_environment": launch["strict_environment"],
        "frozen_sources_match": frozen_match,
    }
    write_report(args.run_dir / "single-arm-summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
