#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-${HOME}/isaacsim}"
ISAAC_PYTHON="${ISAAC_SIM_ROOT}/python.sh"
EXPECTED_GIT_SHA="${EXPECTED_GIT_SHA:-}"
CURRENT_GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
EVIDENCE_ROOT="${V2_STABILITY_ROOT:-${REPO_ROOT}/artifacts/v2/stability-${CURRENT_GIT_SHA:0:8}-${TIMESTAMP}}"
SUMMARY_FILE="${EVIDENCE_ROOT}/restart-summary.tsv"

if [[ -z "${EXPECTED_GIT_SHA}" || "${EXPECTED_GIT_SHA}" != "${CURRENT_GIT_SHA}" ]]; then
  printf 'ERROR: EXPECTED_GIT_SHA must equal current HEAD %s.\n' "${CURRENT_GIT_SHA}" >&2
  exit 2
fi
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  printf 'ERROR: V2 stability evidence requires a clean worktree.\n' >&2
  git -C "${REPO_ROOT}" status --short >&2
  exit 2
fi
if [[ ! -x "${ISAAC_PYTHON}" ]]; then
  printf 'ERROR: Isaac Sim python is not executable: %s\n' "${ISAAC_PYTHON}" >&2
  exit 2
fi

mkdir -p "${EVIDENCE_ROOT}"
{
  printf 'captured_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'repository=%s\n' "${REPO_ROOT}"
  printf 'git_sha=%s\n' "${CURRENT_GIT_SHA}"
  printf 'git_branch=%s\n' "$(git -C "${REPO_ROOT}" branch --show-current)"
  printf 'worktree_clean=yes\n'
  printf '\n[os-release]\n'
  cat /etc/os-release
  printf '\n[gpu]\n'
  nvidia-smi
} >"${EVIDENCE_ROOT}/platform-inventory.txt" 2>&1

printf 'run\tstarted_at\texit_code\tpurpose\n' >"${SUMMARY_FILE}"
overall_exit=0
for run_number in 1 2 3; do
  run_dir="${EVIDENCE_ROOT}/restart-${run_number}"
  mkdir -p "${run_dir}"
  started_at="$(date --iso-8601=seconds)"
  if [[ "${run_number}" -eq 1 ]]; then
    purpose="V2 1000 steps + 20 resets + 3 cameras"
    extra_args=(--steps 1000 --resets 20 --capture-cameras)
  else
    purpose="V2 independent cold restart smoke"
    extra_args=(--steps 20 --resets 0 --no-capture-cameras)
  fi
  "${ISAAC_PYTHON}" "${REPO_ROOT}/simulation/run_v2_stability_acceptance.py" \
    --evidence-dir "${run_dir}" \
    --output-scene "${run_dir}/single_bin_scene_v2.usda" \
    --reset-settle-steps 120 "${extra_args[@]}" \
    >"${run_dir}/console.log" 2>&1
  exit_code=$?
  result_file="${run_dir}/run_result.json"
  if [[ ! -s "${result_file}" ]]; then
    exit_code=10
  elif ! grep -Eq '"status"[[:space:]]*:[[:space:]]*"PASS"' "${result_file}"; then
    exit_code=11
  fi
  if [[ "${run_number}" -eq 1 ]]; then
    for required in reset_report.json step_checks.json camera_manifest.json \
      cameras/CAM_A_TOP.ppm cameras/CAM_HANDOFF.ppm cameras/CAM_B_TOP.ppm; do
      if [[ ! -s "${run_dir}/${required}" ]]; then
        printf 'Missing required evidence: %s\n' "${run_dir}/${required}" >>"${run_dir}/console.log"
        exit_code=12
      fi
    done
  fi
  printf '%s\t%s\t%s\t%s\n' "${run_number}" "${started_at}" "${exit_code}" "${purpose}" >>"${SUMMARY_FILE}"
  if [[ "${exit_code}" -ne 0 ]]; then
    overall_exit=1
    printf 'Restart %s FAILED: %s\n' "${run_number}" "${run_dir}/console.log" >&2
  else
    printf 'Restart %s PASSED.\n' "${run_number}"
  fi
done

(
  cd "${EVIDENCE_ROOT}" || exit 1
  find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum
) >"${EVIDENCE_ROOT}/SHA256SUMS.txt" || overall_exit=1
(
  cd "${EVIDENCE_ROOT}" || exit 1
  sha256sum -c SHA256SUMS.txt
) >/dev/null || overall_exit=1

printf 'V2_STABILITY_ROOT=%s\n' "${EVIDENCE_ROOT}"
printf 'SUMMARY=%s\n' "${SUMMARY_FILE}"
exit "${overall_exit}"
