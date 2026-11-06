#!/bin/bash -l
#SBATCH --job-name=flagsparse-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=flagsparse-%j.out
#SBATCH --error=flagsparse-%j.err

# Submit from the repository root, or provide --repo explicitly.
# Scheduler options belong BEFORE this filename; test options follow --.
set -euo pipefail

usage() {
    cat <<'HELP'
sbatch [Slurm options] tools/run_cuda_tests.sbatch [wrapper options] -- [test arguments]

Wrapper options:
  --repo PATH       Repository (default: FLAGSPARSE_REPO, current repo, submission directory)
  --image PATH      Image (default: $HOME/run/images/flagtree.sif)
  --venv PATH       Venv (default: /data/home/scwb320/run/gcx/vpy)
  --libcuda PATH    Triton libcuda directory (default: /usr/local/cuda/compat/lib.real)
  --entry PATH      Python entry relative to repo (default: run_flagsparse_pytest.py)
  --help           Show this help; no container or tests started

All arguments after -- are passed verbatim to the Python entry, without eval.
The runner gets a job-specific --results-dir unless one was supplied explicitly.
Use --entry tests/test_spmv_csr.py (etc.) to run a benchmark directly.
HELP
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
# Interactive allocations retain their original SLURM_SUBMIT_DIR even after cd.
# Prefer a repository in the current directory when no explicit override exists.
if [[ -n "${FLAGSPARSE_REPO:-}" ]]; then
    repo=$FLAGSPARSE_REPO
elif [[ -f "$PWD/run_flagsparse_pytest.py" ]]; then
    repo=$PWD
else
    repo=${SLURM_SUBMIT_DIR:-$PWD}
fi
image=${FLAGSPARSE_IMAGE:-$HOME/run/images/flagtree.sif}
venv=${FLAGSPARSE_VENV:-/data/home/scwb320/run/gcx/vpy}
libcuda=${FLAGSPARSE_LIBCUDA:-/usr/local/cuda/compat/lib.real}
entry=run_flagsparse_pytest.py
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --) shift; break ;;
        --repo|--image|--venv|--libcuda|--entry)
            (($# >= 2)) || die "Missing value for $1"
            case "$1" in
                --repo) repo=$2 ;;
                --image) image=$2 ;;
                --venv) venv=$2 ;;
                --libcuda) libcuda=$2 ;;
                --entry) entry=$2 ;;
            esac
            shift 2 ;;
        *) die "Unknown wrapper option $1; put Python test arguments after --" ;;
    esac
done

[[ -d "$repo" ]] || die "Repository does not exist: $repo"
repo=$(cd "$repo" && pwd -P)
[[ -f "$repo/run_flagsparse_pytest.py" ]] || die "Not a FlagSparse repository: $repo"
[[ -f "$image" ]] || die "Image not found: $image"
[[ -f "$venv/bin/activate" ]] || die "Venv not found: $venv"
# Resolve paths before the container changes its working directory.
image=$(cd "$(dirname "$image")" && printf '%s/%s' "$PWD" "$(basename "$image")")
venv=$(cd "$venv" && pwd -P)
[[ "$entry" != /* && "$entry" != *..* && "$entry" == *.py ]] || die "--entry must be a repo-relative Python file"
[[ -f "$repo/$entry" ]] || die "Entry not found: $repo/$entry"

type module >/dev/null 2>&1 || die "module is unavailable; initialize the cluster module environment before sbatch"
module load apptainer
if command -v apptainer >/dev/null 2>&1; then
    container=apptainer
elif command -v singularity >/dev/null 2>&1; then
    container=singularity
else
    die "module load apptainer did not provide apptainer or singularity"
fi

job_tag=${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)-$$}
job_dir="$repo/slurm_results/$job_tag"
mkdir -p "$job_dir"
args=("$@")
if [[ "$entry" == run_flagsparse_pytest.py ]]; then
    has_results=0
    for arg in "${args[@]}"; do
        case "$arg" in --results-dir|--results-dir=*) has_results=1 ;; esac
    done
    if ((has_results == 0)); then
        args+=(--results-dir "$job_dir/results")
    fi
fi

# --cleanenv must not discard Slurm's GPU selection. Do not hard-code GPU 0.
gpu_env=()
if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
    gpu_env+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi
printf 'Job=%s Host=%s Repo=%s\nImage=%s Venv=%s Entry=%s\n' \
    "$job_tag" "$(hostname)" "$repo" "$image" "$venv" "$entry"

# Noninteractive exec replaces the interactive `singularity shell` workflow.
# Bind repo/venv explicitly as well as /data so paths outside /data also work.
exec "$container" exec --nv --cleanenv \
    --bind /data:/data --bind "$repo:$repo" --bind "$venv:$venv" \
    "${gpu_env[@]}" "$image" /bin/bash -s -- \
    "$repo" "$venv" "$libcuda" "$entry" "$job_dir" "${args[@]}" <<'CONTAINER'
set -eo pipefail
repo=$1 venv=$2 libcuda=$3 entry=$4 job_dir=$5
shift 5
source "$venv/bin/activate"
set -u
cd "$repo"
[[ -d "$libcuda" ]] || { echo "ERROR: libcuda directory missing: $libcuda" >&2; exit 2; }
[[ -e "$libcuda/libcuda.so" || -e "$libcuda/libcuda.so.1" ]] || {
    echo "ERROR: libcuda.so/libcuda.so.1 not found in $libcuda" >&2; exit 2;
}
unset TRITON_LIBCUDA_PATH
export TRITON_LIBCUDA_PATH="$libcuda"
export PYTHONPATH="$repo/src:$repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# Scope runner GPU ids to the Slurm allocation, including UUID-based masks.
if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
    export FLAGSPARSE_SLURM_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi
# Writable and isolated across jobs; compilation/warmup remain the runner's job.
export TRITON_CACHE_DIR="$job_dir/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"
python_command=$(command -v python)
# activate can retain the original /data/home/... spelling while the host
# resolved --venv through a symlink to /data/run01/.... Compare directories,
# not path strings or the Python executable (venvs can share its symlink target).
if [[ -z "${VIRTUAL_ENV:-}" || ! "$VIRTUAL_ENV" -ef "$venv" ||
      ! "$(dirname "$python_command")" -ef "$venv/bin" ]]; then
    printf 'ERROR: venv mismatch: requested=%s activated=%s Python=%s\n' \
        "$venv" "${VIRTUAL_ENV-<unset>}" "$python_command" >&2
    exit 2
fi
{
    printf 'Python=%s\nTRITON_LIBCUDA_PATH=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
        "$(command -v python)" "$TRITON_LIBCUDA_PATH" "${CUDA_VISIBLE_DEVICES-<unset>}"
    python --version
    printf 'Command:'; printf ' %q' python "$entry" "$@"; printf '\n'
} | tee "$job_dir/launch.txt"
# pipefail preserves the Python exit status while keeping a per-job log.
# Accuracy statuses must still be read from the runner's JSON/CSV reports.
python "$entry" "$@" 2>&1 | tee "$job_dir/test.log"
CONTAINER
