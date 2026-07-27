#!/usr/bin/env bash
#
# Member-B G0 wrapper for an Isaac Sim 5.1 Linux workstation.
#
# Usage:
#   ISAAC_SIM_ROOT="$HOME/isaacsim" bash scripts/run_g0_linux.sh
#
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-${HOME}/isaacsim}"
ISAAC_PYTHON="${ISAAC_SIM_ROOT}/python.sh"
TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
EVIDENCE_ROOT="${G0_EVIDENCE_ROOT:-${REPO_ROOT}/artifacts/g0/${TIMESTAMP}}"
SCENE_OUTPUT="${REPO_ROOT}/simulation/generated/single_bin_scene_v1.usda"
SUMMARY_FILE="${EVIDENCE_ROOT}/restart-summary.tsv"

mkdir -p "${EVIDENCE_ROOT}"

if [[ ! -x "${ISAAC_PYTHON}" ]]; then
  printf 'ERROR: Isaac Sim python.sh is not executable: %s\n' "${ISAAC_PYTHON}" >&2
  printf 'Set the correct path, for example:\n' >&2
  printf '  ISAAC_SIM_ROOT=/opt/isaacsim bash scripts/run_g0_linux.sh\n' >&2
  exit 2
fi

{
  printf 'captured_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'repository=%s\n' "${REPO_ROOT}"
  printf 'isaac_sim_root=%s\n' "${ISAAC_SIM_ROOT}"
  printf '\n[uname]\n'
  uname -a
  printf '\n[os-release]\n'
  cat /etc/os-release
  printf '\n[cpu]\n'
  lscpu
  printf '\n[memory]\n'
  free -h
  printf '\n[disk]\n'
  df -h "${REPO_ROOT}" "${ISAAC_SIM_ROOT}"
  printf '\n[gpu]\n'
  nvidia-smi
  printf '\n[git]\n'
  git -C "${REPO_ROOT}" rev-parse HEAD
  git -C "${REPO_ROOT}" status --short
  printf '\n[isaac-files]\n'
  ls -l "${ISAAC_SIM_ROOT}/python.sh" "${ISAAC_SIM_ROOT}/isaac-sim.sh"
} >"${EVIDENCE_ROOT}/platform-inventory.txt" 2>&1

printf 'run\tstarted_at\texit_code\tpurpose\n' >"${SUMMARY_FILE}"
overall_exit=0

for run_number in 1 2 3; do
  run_dir="${EVIDENCE_ROOT}/restart-${run_number}"
  mkdir -p "${run_dir}"
  started_at="$(date --iso-8601=seconds)"

  if [[ "${run_number}" -eq 1 ]]; then
    purpose="1000 steps + 20 resets + 3 cameras"
    command=(
      "${ISAAC_PYTHON}"
      "${REPO_ROOT}/simulation/run_g0_acceptance.py"
      --evidence-dir "${run_dir}"
      --output-scene "${SCENE_OUTPUT}"
      --steps 1000
      --resets 20
      --capture-cameras
    )
  else
    purpose="independent cold restart smoke"
    command=(
      "${ISAAC_PYTHON}"
      "${REPO_ROOT}/simulation/run_g0_acceptance.py"
      --evidence-dir "${run_dir}"
      --output-scene "${SCENE_OUTPUT}"
      --steps 20
      --resets 0
      --no-capture-cameras
    )
  fi

  printf '\n=== Isaac Sim restart %s/3: %s ===\n' "${run_number}" "${purpose}"
  printf 'Command:'
  printf ' %q' "${command[@]}"
  printf '\n'

  "${command[@]}" >"${run_dir}/console.log" 2>&1
  process_exit_code=$?
  exit_code="${process_exit_code}"
  result_file="${run_dir}/run_result.json"

  # A Kit process can exit with code 0 even if Python never completed the
  # requested acceptance work. The evidence file and its explicit PASS status
  # are therefore mandatory; process exit code alone is not a pass criterion.
  if [[ ! -f "${result_file}" ]]; then
    printf 'Missing required evidence: %s\n' "${result_file}" \
      >>"${run_dir}/console.log"
    exit_code=10
  elif ! grep -Eq '"status"[[:space:]]*:[[:space:]]*"PASS"' "${result_file}"; then
    printf 'run_result.json does not contain status PASS.\n' \
      >>"${run_dir}/console.log"
    exit_code=11
  elif [[ "${run_number}" -eq 1 ]]; then
    required_files=(
      "${run_dir}/reset_report.json"
      "${run_dir}/robot_observation.json"
      "${run_dir}/camera_manifest.json"
      "${run_dir}/cameras/CAM_A_TOP.ppm"
      "${run_dir}/cameras/CAM_HANDOFF.ppm"
      "${run_dir}/cameras/CAM_B_TOP.ppm"
    )
    for required_file in "${required_files[@]}"; do
      if [[ ! -s "${required_file}" ]]; then
        printf 'Missing or empty required evidence: %s\n' "${required_file}" \
          >>"${run_dir}/console.log"
        exit_code=12
      fi
    done
  fi

  printf '%s\t%s\t%s\t%s\n' \
    "${run_number}" "${started_at}" "${exit_code}" "${purpose}" >>"${SUMMARY_FILE}"

  if [[ "${exit_code}" -ne 0 ]]; then
    overall_exit=1
    printf 'Restart %s FAILED. Read %s\n' \
      "${run_number}" "${run_dir}/console.log" >&2
  else
    printf 'Restart %s PASSED.\n' "${run_number}"
  fi
done

if command -v sha256sum >/dev/null 2>&1; then
  find "${EVIDENCE_ROOT}" -type f ! -name 'SHA256SUMS.txt' -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >"${EVIDENCE_ROOT}/SHA256SUMS.txt"
fi

printf '\nEvidence directory: %s\n' "${EVIDENCE_ROOT}"
printf 'Restart summary: %s\n' "${SUMMARY_FILE}"
if [[ "${overall_exit}" -eq 0 ]]; then
  printf 'G0 AUTOMATED CHECKS PASSED.\n'
else
  printf 'G0 AUTOMATED CHECKS FAILED. Do not mark the gate as passed.\n' >&2
fi
exit "${overall_exit}"
