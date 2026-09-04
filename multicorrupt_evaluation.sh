#!/bin/bash
set -euo pipefail

# ==============================================================================
# MultiCorrupt Evaluation Script
# Strategy: build an assembly directory and symlink the corrupted data
#           subdirectories and the original .pkl files into it.
# Metrics:  RA, mRA, RRA and mRRA (over NDS and mAP) are computed automatically.
# RRA baseline: original MEFormer R50 (canonical local robustness baseline)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:${PYTHONPATH}}"

# --- Configuration -----------------------------------------------------------
corruptions=("beamsreducing" "brightness" "dark" "fog" "missingcamera" "motionblur" "pointsreducing" "snow" "spatialmisalignment" "temporalmisalignment")
severity_levels=("1" "2" "3")

multicorrupt_root="${MULTICORRUPT_ROOT:-}"
nuscenes_data_dir="${NUSCENES_DATA_DIR:-${ROOT_DIR}/data/nuscenes}"
nuscenes_backup_dir="${NUSCENES_BACKUP_DIR:-${ROOT_DIR}/data/nuscenes.multicorrupt-backup}"

config="${CONFIG:-}"
checkpoint="${CKPT:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

config_name="$(basename "$config" .py)"
model_label="${MODEL_LABEL:-$config_name}"
output_dir="${OUT_DIR:-${ROOT_DIR}/MultiCorrupt_results/${model_label}}"
logfile="${output_dir}/evaluation_log.txt"
metrics_file="${output_dir}/metrics_summary.txt"
edl_log_root="${output_dir}/edl_logs"
edl_manifest="${edl_log_root}/log_manifest.tsv"
viz_output_dir="${output_dir}/visualizations"
TEST_WORKERS_PER_GPU="${TEST_WORKERS_PER_GPU:-4}"
EXTRA_CFG_OPTIONS="${EXTRA_CFG_OPTIONS:-}"
EV_LOG_MODE="${EV_LOG_MODE:-full_stream}"
EV_LOG_COMPRESSION="${EV_LOG_COMPRESSION:-gzip}"
EV_LOG_FLUSH_INTERVAL="${EV_LOG_FLUSH_INTERVAL:-100}"
EV_LOG_PREDICTION_FULL="${EV_LOG_PREDICTION_FULL:-0}"
EV_LOG_SCENE_ONLY="${EV_LOG_SCENE_ONLY:-0}"
EDL_VIZ_JOBS="${EDL_VIZ_JOBS:-8}"
EDL_VIZ_HEARTBEAT_SAMPLES="${EDL_VIZ_HEARTBEAT_SAMPLES:-250}"
EDL_VIZ_STRICT_V6="${EDL_VIZ_STRICT_V6:-auto}"
RUN_VIZ="${RUN_VIZ:-1}"

die() {
    echo "[ERROR] $*" >&2
    exit 2
}

canonical_path() {
    realpath -m -- "$1"
}

# The configs use data_root='data/nuscenes'. Keep the temporary assembly at the
# repository path so evaluation cannot silently read a different dataset.
expected_nuscenes_dir="$(canonical_path "${ROOT_DIR}/data/nuscenes")"
nuscenes_data_dir="$(canonical_path "$nuscenes_data_dir")"
nuscenes_backup_dir="$(canonical_path "$nuscenes_backup_dir")"
multicorrupt_root="$(canonical_path "${multicorrupt_root:-.}")"

[[ -n "${MULTICORRUPT_ROOT:-}" ]] || die "set MULTICORRUPT_ROOT to the extracted MultiCorrupt root"
[[ -n "$config" ]] || die "set CONFIG to the exact evaluation config"
[[ -n "$checkpoint" ]] || die "set CKPT to the exact evaluation checkpoint"
[[ "$nuscenes_data_dir" == "$expected_nuscenes_dir" ]] || \
    die "NUSCENES_DATA_DIR must resolve to ${expected_nuscenes_dir}"
[[ "$nuscenes_backup_dir" != "$nuscenes_data_dir" ]] || \
    die "NUSCENES_BACKUP_DIR must differ from NUSCENES_DATA_DIR"
[[ "$(dirname "$nuscenes_backup_dir")" == "$(dirname "$nuscenes_data_dir")" ]] || \
    die "keep NUSCENES_BACKUP_DIR beside data/nuscenes for atomic restoration"
[[ -d "$multicorrupt_root" ]] || die "MultiCorrupt root not found: $multicorrupt_root"
[[ -d "$nuscenes_data_dir" || -L "$nuscenes_data_dir" ]] || \
    die "clean nuScenes directory not found: $nuscenes_data_dir"
[[ -f "$config" ]] || die "config not found: $config"
[[ -f "$checkpoint" ]] || die "checkpoint not found: $checkpoint"
"$PYTHON_BIN" -c 'import mmcv, torch, mmdet3d' >/dev/null 2>&1 || \
    die "$PYTHON_BIN cannot import mmcv, torch, and mmdet3d"

for corruption in "${corruptions[@]}"; do
    for severity in "${severity_levels[@]}"; do
        [[ -d "${multicorrupt_root}/${corruption}/${severity}" ]] || \
            die "incomplete MultiCorrupt tree: ${multicorrupt_root}/${corruption}/${severity}"
    done
done
[[ -f "${nuscenes_data_dir}/nuscenes_infos_val.pkl" ]] || \
    die "missing nuScenes validation infos; run bash tools/create_data.sh first"

ev_log_enabled() {
    case "${EV_LOG_MODE,,}" in
        off|0|false|none|no)
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}

# --- Clean baseline (M_clean) ------------------------------------------------
# Set to "auto" to have the script run a clean-data evaluation first, or fill in
# a known value (e.g. 0.739) to skip that clean run.
CLEAN_NDS="auto"
CLEAN_MAP="auto"

# Subdirectories to symlink from corruption datasets
DATA_SUBDIRS=("samples" "sweeps" "v1.0-trainval" "maps" "lidarseg")

# --- Safety: restore original dir on exit/error ------------------------------
restored=false
cleanup() {
    if [ "$restored" = false ] && [ -d "$nuscenes_backup_dir" ]; then
        echo "[CLEANUP] Restoring original data/nuscenes from backup..."
        rm -rf "$nuscenes_data_dir"
        mv "$nuscenes_backup_dir" "$nuscenes_data_dir"
        echo "[CLEANUP] Restored."
    fi
}
trap cleanup EXIT INT TERM

# --- Create output directory --------------------------------------------------
mkdir -p "$output_dir"
mkdir -p "$edl_log_root"
mkdir -p "$viz_output_dir"

find_first_log_match() {
    local pattern path
    for pattern in "$@"; do
        while IFS= read -r path; do
            if [ -f "$path" ]; then
                printf "%s" "$path"
                return 0
            fi
        done < <(compgen -G "$pattern" | sort)
    done
    return 1
}

write_manifest_row() {
    local split="$1"
    local corruption="$2"
    local severity="$3"
    local log_dir="$4"
    local eval_output="$5"
    write_manifest_rows "$split" "$corruption" "$severity" "$log_dir" "$eval_output"
}

write_manifest_rows() {
    local split="$1"
    local corruption="$2"
    local severity="$3"
    local log_dir="$4"
    local eval_output="$5"
    local scene_summary scene_base scene_ext scene_stem suffix
    local prediction_light query_semantic sidecar_meta log_json wrote=0

    while IFS= read -r scene_summary; do
        [ -f "$scene_summary" ] || continue
        scene_base="$(basename "$scene_summary")"
        if [[ "$scene_base" == *.jsonl.gz ]]; then
            scene_ext=".jsonl.gz"
            scene_stem="${scene_base%.jsonl.gz}"
        elif [[ "$scene_base" == *.jsonl.zst ]]; then
            scene_ext=".jsonl.zst"
            scene_stem="${scene_base%.jsonl.zst}"
        else
            scene_ext=".jsonl"
            scene_stem="${scene_base%.jsonl}"
        fi
        suffix="${scene_stem#ev_scene_summary}"
        prediction_light="$log_dir/ev_prediction_light${suffix}${scene_ext}"
        query_semantic="$log_dir/ev_query_semantic${suffix}${scene_ext}"
        sidecar_meta="$log_dir/ev_meta${suffix}.json"
        log_json="$log_dir/ev_sample_full${suffix}${scene_ext}"
        [ -f "$log_json" ] || log_json="$scene_summary"
        if [ -f "$sidecar_meta" ] && \
           { [ -f "$prediction_light" ] || [ "$EV_LOG_SCENE_ONLY" = "1" ] || \
             [ "$EV_LOG_SCENE_ONLY" = "true" ]; }; then
            [ -f "$prediction_light" ] || prediction_light=""
            [ -f "$query_semantic" ] || query_semantic=""
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$split" "$corruption" "$severity" "$log_dir" "$log_json" \
                "$scene_summary" "$prediction_light" "$query_semantic" \
                "$sidecar_meta" "$eval_output"
            wrote=1
        fi
    done < <(compgen -G "$log_dir/ev_scene_summary*.jsonl*" | sort)

    if [ "$wrote" -eq 0 ] && [ -f "$log_dir/edl_scene_summary.jsonl" ] && \
       [ -f "$log_dir/edl_prediction_light.jsonl" ] && \
       [ -f "$log_dir/edl_sidecar_meta.json" ]; then
        log_json="$log_dir/edl_uncertainty_log.json"
        [ -f "$log_json" ] || log_json="$log_dir/edl_scene_summary.jsonl"
        query_semantic="$log_dir/edl_query_semantic.jsonl"
        [ -f "$query_semantic" ] || query_semantic=""
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$split" "$corruption" "$severity" "$log_dir" "$log_json" \
            "$log_dir/edl_scene_summary.jsonl" \
            "$log_dir/edl_prediction_light.jsonl" \
            "$query_semantic" \
            "$log_dir/edl_sidecar_meta.json" \
            "$eval_output"
        wrote=1
    fi

    [ "$wrote" -eq 1 ]
}

has_manifest_logs() {
    if ! ev_log_enabled; then
        return 0
    fi
    write_manifest_rows "$@" >/dev/null
}

write_edl_manifest() {
    mkdir -p "$edl_log_root"
    local tmp_dir
    tmp_dir=$(mktemp -d)
    trap 'rm -rf "$tmp_dir"' RETURN

    local idx=0
    for corruption in "${corruptions[@]}"; do
        for severity in "${severity_levels[@]}"; do
            idx=$((idx + 1))
            {
                eval_output="${output_dir}/${corruption}_${severity}_output.txt"
                log_dir="${edl_log_root}/${corruption}/severity_${severity}"
                row=$(write_manifest_rows "corrupt" "$corruption" "$severity" "$log_dir" "$eval_output" || true)
                if [ -n "$row" ]; then
                    while IFS= read -r row_line; do
                        [ -n "$row_line" ] || continue
                        printf "%06d\t%s\n" "$idx" "$row_line"
                    done <<< "$row" > "${tmp_dir}/${idx}.tsv"
                fi
            } &
        done
    done
    wait

    {
        printf "split\tcorruption\tseverity\tedl_log_dir\tedl_log_json\tedl_scene_summary_jsonl\tedl_prediction_light_jsonl\tedl_query_semantic_jsonl\tedl_sidecar_meta_json\teval_output\n"

        write_manifest_rows "clean" "clean" "0" "$clean_edl_log_dir" "$clean_output_file" || true

        find "$tmp_dir" -maxdepth 1 -name '*.tsv' -print0 \
            | sort -z \
            | xargs -0 -r cat \
            | cut -f2-
    } > "$edl_manifest"
    echo "[EDL Viz] Manifest written to: $edl_manifest"
}

run_eval() {
    local edl_dir="$1"
    local output_file="$2"
    local split="$3"
    local corruption="$4"
    local severity="$5"
    local -a cmd=(
        "$PYTHON_BIN" tools/test.py "$config" "$checkpoint" --eval=bbox
        --cfg-options
        data.workers_per_gpu="$TEST_WORKERS_PER_GPU"
        data.test_dataloader.workers_per_gpu="$TEST_WORKERS_PER_GPU"
    )
    if ev_log_enabled; then
        cmd+=(
            --save-ev-log --ev-log-dir "$edl_dir"
            --ev-log-mode "$EV_LOG_MODE"
            --ev-log-compression "$EV_LOG_COMPRESSION"
            --ev-log-flush-interval "$EV_LOG_FLUSH_INTERVAL"
        )
    fi
    if [ -n "$EXTRA_CFG_OPTIONS" ]; then
        read -r -a extra_cfg_options_array <<< "$EXTRA_CFG_OPTIONS"
        cmd+=("${extra_cfg_options_array[@]}")
    fi
    if ev_log_enabled && { [ "$EV_LOG_PREDICTION_FULL" = "1" ] || [ "$EV_LOG_PREDICTION_FULL" = "true" ]; }; then
        cmd+=(--ev-log-prediction-full)
    fi
    if ev_log_enabled && { [ "$EV_LOG_SCENE_ONLY" = "1" ] || [ "$EV_LOG_SCENE_ONLY" = "true" ]; }; then
        cmd+=(--ev-log-scene-only)
    fi
    EV_SPLIT="$split" \
    EV_CORRUPTION="$corruption" \
    EV_SEVERITY="$severity" \
    EV_MODEL_LABEL="$model_label" \
        "${cmd[@]}" 2>&1 | tee "$output_file"
}

# --- Helper: assemble nuscenes dir --------------------------------------------
assemble_nuscenes() {
    local corrupt_dir="$1"

    mkdir -p "$nuscenes_data_dir"

    for subdir in "${DATA_SUBDIRS[@]}"; do
        if [ -d "${corrupt_dir}/${subdir}" ] || [ -L "${corrupt_dir}/${subdir}" ]; then
            ln -s "${corrupt_dir}/${subdir}" "${nuscenes_data_dir}/${subdir}"
        elif [ -d "${nuscenes_backup_dir}/${subdir}" ]; then
            ln -s "${nuscenes_backup_dir}/${subdir}" "${nuscenes_data_dir}/${subdir}"
        fi
    done

    for pkl in "${nuscenes_backup_dir}"/nuscenes_infos_*.pkl; do
        [ -f "$pkl" ] && ln -s "$pkl" "${nuscenes_data_dir}/$(basename "$pkl")"
    done
    if [ -f "${nuscenes_backup_dir}/nuscenes_dbinfos_train.pkl" ]; then
        ln -s "${nuscenes_backup_dir}/nuscenes_dbinfos_train.pkl" "${nuscenes_data_dir}/nuscenes_dbinfos_train.pkl"
    fi
}

# --- Step 0: Backup original data/nuscenes ------------------------------------
if [ -d "$nuscenes_backup_dir" ]; then
    echo "[ERROR] Backup directory $nuscenes_backup_dir already exists!"
    echo "        Previous run may have been interrupted."
    echo "        Please check and manually run: mv $nuscenes_backup_dir $nuscenes_data_dir"
    exit 1
fi

echo "======================================================================"
echo " Backing up original data/nuscenes -> data/nuscenes.bak"
echo "======================================================================"
mv "$nuscenes_data_dir" "$nuscenes_backup_dir"

# --- Step 1: Get clean baseline (M_clean) if needed ---------------------------
clean_output_file="${output_dir}/clean_output.txt"
clean_edl_log_dir="${edl_log_root}/clean"

if [ "$CLEAN_NDS" = "auto" ] || [ "$CLEAN_MAP" = "auto" ]; then
    echo ""
    echo "======================================================================"
    echo " [0/N] Evaluating CLEAN nuScenes baseline (M_clean)"
    echo "======================================================================"

    if [ -f "$clean_output_file" ] && grep -q "NDS" "$clean_output_file" 2>/dev/null && \
       has_manifest_logs "clean" "clean" "0" "$clean_edl_log_dir" "$clean_output_file"; then
        echo "[SKIP] Clean output already exists: $clean_output_file"
    else
        assemble_nuscenes "$nuscenes_backup_dir"
        mkdir -p "$clean_edl_log_dir"
        run_eval "$clean_edl_log_dir" "$clean_output_file" "clean" "clean" "0"
        rm -rf "$nuscenes_data_dir"
    fi

    CLEAN_NDS=$(grep -oP 'NDS:\s*\K[0-9.]+' "$clean_output_file" || echo "")
    CLEAN_MAP=$(grep -oP 'mAP:\s*\K[0-9.]+' "$clean_output_file" || echo "")

    if [ -z "$CLEAN_NDS" ] || [ -z "$CLEAN_MAP" ]; then
        echo "[ERROR] Failed to extract clean NDS/mAP from $clean_output_file"
        rm -rf "$nuscenes_data_dir" 2>/dev/null || true
        mv "$nuscenes_backup_dir" "$nuscenes_data_dir"
        restored=true
        exit 1
    fi
fi

echo ""
echo "  Clean baseline: NDS=$CLEAN_NDS, mAP=$CLEAN_MAP"
echo ""

# --- Step 2: Loop over corruptions and severity levels ------------------------
total=$((${#corruptions[@]} * ${#severity_levels[@]}))
count=0

for corruption in "${corruptions[@]}"; do
    for severity in "${severity_levels[@]}"; do
        count=$((count + 1))
        corrupt_dir="${multicorrupt_root}/${corruption}/${severity}"
        output_file="${output_dir}/${corruption}_${severity}_output.txt"
        edl_log_dir="${edl_log_root}/${corruption}/severity_${severity}"

        echo ""
        echo "======================================================================"
        echo " [$count/$total] Corruption=$corruption, Severity=$severity"
        echo "======================================================================"

        # Skip if already evaluated
        if [ -f "$output_file" ] && grep -q "NDS" "$output_file" 2>/dev/null && \
           has_manifest_logs "corrupt" "$corruption" "$severity" "$edl_log_dir" "$output_file"; then
            echo "[SKIP] Output already exists: $output_file"
            if ! grep -q "Corruption: $corruption, Severity: $severity" "$logfile" 2>/dev/null; then
                nds=$(grep -oP 'NDS:\s*\K[0-9.]+' "$output_file" || echo "N/A")
                map_score=$(grep -oP 'mAP:\s*\K[0-9.]+' "$output_file" || echo "N/A")
                echo "Corruption: $corruption, Severity: $severity, NDS: $nds, mAP: $map_score" >> "$logfile"
            fi
            continue
        fi

        if [ ! -d "$corrupt_dir" ]; then
            echo "[ERROR] Corruption directory not found: $corrupt_dir"
            echo "Corruption: $corruption, Severity: $severity, NDS: ERROR, mAP: ERROR" >> "$logfile"
            continue
        fi

        assemble_nuscenes "$corrupt_dir"
        echo "  Assembly complete. Running evaluation..."
        mkdir -p "$edl_log_dir"

        run_eval "$edl_log_dir" "$output_file" "corrupt" "$corruption" "$severity"

        nds=$(grep -oP 'NDS:\s*\K[0-9.]+' "$output_file" || echo "N/A")
        map_score=$(grep -oP 'mAP:\s*\K[0-9.]+' "$output_file" || echo "N/A")
        echo "Corruption: $corruption, Severity: $severity, NDS: $nds, mAP: $map_score" >> "$logfile"
        echo "  => NDS: $nds, mAP: $map_score"
        echo "  => EDL log dir: $edl_log_dir"

        rm -rf "$nuscenes_data_dir"
    done
done

# --- Step 3: Restore original data/nuscenes -----------------------------------
echo ""
echo "======================================================================"
echo " Restoring original data/nuscenes"
echo "======================================================================"
rm -rf "$nuscenes_data_dir"
mv "$nuscenes_backup_dir" "$nuscenes_data_dir"
restored=true

# --- Step 4: Compute RA, mRA, RRA, mRRA --------------------------------------
echo ""
echo "======================================================================"
echo " Computing RA / mRA / RRA / mRRA metrics"
echo "======================================================================"

"$PYTHON_BIN" - "$logfile" "$CLEAN_NDS" "$CLEAN_MAP" "$metrics_file" << 'PYTHON_SCRIPT'
import sys
import re

logfile = sys.argv[1]
clean_nds = float(sys.argv[2])
clean_map = float(sys.argv[3])
metrics_file = sys.argv[4] if len(sys.argv) > 4 else ""

# ─── Original MEFormer R50 baseline (canonical local robustness baseline) ───
# Source: ai_memory canonical baseline, from formal MultiCorrupt result
# Clean: mAP = 0.6768, NDS = 0.7096
BASELINE_NAME = "MEFormer R50"
BASELINE_CLEAN_NDS = 0.7096
BASELINE_CLEAN_MAP = 0.6768
BASELINE_RA_NDS = {
    "beamsreducing":        0.7162,
    "brightness":           0.9517,
    "dark":                 0.9596,
    "fog":                  0.7764,
    "missingcamera":        0.9552,
    "motionblur":           0.8470,
    "pointsreducing":       0.9014,
    "snow":                 0.7843,
    "spatialmisalignment":  0.7469,
    "temporalmisalignment": 0.7836,
}
# ─────────────────────────────────────────────────────────────────

# Parse evaluation log
results = {}
pattern = re.compile(
    r"Corruption:\s*(\w+),\s*Severity:\s*(\d+),\s*NDS:\s*([\d.]+|N/A|ERROR),\s*mAP:\s*([\d.]+|N/A|ERROR)"
)

with open(logfile, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            try:
                results[(m.group(1), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
            except ValueError:
                pass

all_corruptions = sorted(set(c for c, s in results.keys()))

output_lines = []
def out(s=""):
    print(s)
    output_lines.append(s)

# ─── RA Table ───
out("=" * 90)
out("  Resistance Ability (RA)  —  RA_{c,s} = M_{c,s} / M_clean")
out("=" * 90)
out(f"  Clean baseline:  NDS = {clean_nds:.4f},  mAP = {clean_map:.4f}")
out("")
out(f"{'Corruption':<25} {'Sev':>3} {'NDS':>8} {'mAP':>8} {'RA_NDS':>8} {'RA_mAP':>8}")
out("-" * 90)

ra_nds_per_c = {}
ra_map_per_c = {}

for corr in all_corruptions:
    ra_nds_list = []
    ra_map_list = []
    for sev in [1, 2, 3]:
        if (corr, sev) in results:
            nds, mAP = results[(corr, sev)]
            ra_n = nds / clean_nds if clean_nds > 0 else 0
            ra_m = mAP / clean_map if clean_map > 0 else 0
            ra_nds_list.append(ra_n)
            ra_map_list.append(ra_m)
            out(f"{corr:<25} {sev:>3} {nds:>8.4f} {mAP:>8.4f} {ra_n:>8.4f} {ra_m:>8.4f}")
        else:
            out(f"{corr:<25} {sev:>3} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")

    if ra_nds_list:
        ra_c_n = sum(ra_nds_list) / len(ra_nds_list)
        ra_c_m = sum(ra_map_list) / len(ra_map_list)
        ra_nds_per_c[corr] = ra_c_n
        ra_map_per_c[corr] = ra_c_m
        out(f"  {'>> RA_' + corr:<23} {'':>3} {'':>8} {'':>8} {ra_c_n:>8.4f} {ra_c_m:>8.4f}")
    out("")

if ra_nds_per_c:
    mra_n = sum(ra_nds_per_c.values()) / len(ra_nds_per_c)
    mra_m = sum(ra_map_per_c.values()) / len(ra_map_per_c)
    out("-" * 90)
    out(f"  {'mRA (mean RA)':<23} {'':>3} {'':>8} {'':>8} {mra_n:>8.4f} {mra_m:>8.4f}")
    out("")

# ─── RRA Table (vs original MEFormer R50) ───
# RRA_c = (RA_c * M_clean) / (RA_c_baseline * M_clean_baseline) - 1
# This is derived from:
#   RRA_c = Σ M_{c,s} / Σ M_{baseline,c,s} - 1
#         = (3 * RA_c * M_clean) / (3 * RA_c_base * M_clean_base) - 1
out("")
out("=" * 90)
out(f"  Relative Resistance Ability (RRA) vs {BASELINE_NAME} (NDS)")
out("  RRA_c = (RA_c × M_clean) / (RA_c_baseline × M_clean_baseline) - 1")
out("=" * 90)
out(f"  Your model:   Clean NDS = {clean_nds:.4f}")
out(f"  {BASELINE_NAME}:    Clean NDS = {BASELINE_CLEAN_NDS:.4f}, Clean mAP = {BASELINE_CLEAN_MAP:.4f}")
out("")
out(f"{'Corruption':<25} {'RA (yours)':>10} {'RA (base)':>10} {'RRA (%)':>10}")
out("-" * 60)

rra_nds_list = []

for corr in all_corruptions:
    if corr in ra_nds_per_c and corr in BASELINE_RA_NDS:
        ra_ours = ra_nds_per_c[corr]
        ra_base = BASELINE_RA_NDS[corr]
        rra = (ra_ours * clean_nds) / (ra_base * BASELINE_CLEAN_NDS) - 1
        rra_pct = rra * 100
        rra_nds_list.append(rra_pct)
        out(f"{corr:<25} {ra_ours:>10.4f} {ra_base:>10.4f} {rra_pct:>+10.3f}")
    else:
        out(f"{corr:<25} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

if rra_nds_list:
    mrra = sum(rra_nds_list) / len(rra_nds_list)
    out("-" * 60)
    out(f"{'mRRA':<25} {'':>10} {'':>10} {mrra:>+10.3f}")
out("")

# ─── Leaderboard comparison ───
out("=" * 90)
out(f"  Reference: Original {BASELINE_NAME} MultiCorrupt baseline")
out(f"  Clean mAP = {BASELINE_CLEAN_MAP:.4f}, Clean NDS = {BASELINE_CLEAN_NDS:.4f}")
out("  mRA_NDS = 0.8422, mRA_mAP = 0.7470, mRRA = +1.251%")
out("  Secondary reference: official MEFormer VoV leaderboard mRA = 0.856, mRRA = +7.391%")
out("=" * 90)

# Save to file
if metrics_file:
    with open(metrics_file, "w") as f:
        f.write("\n".join(output_lines) + "\n")
    print(f"\nMetrics saved to: {metrics_file}")
PYTHON_SCRIPT

# --- Step 5: Build EDL log manifest and generate visualizations ----------------
echo ""
echo "======================================================================"
echo " Building EDL log manifest"
echo "======================================================================"

if ev_log_enabled; then
    write_edl_manifest
elif [ "$RUN_VIZ" = "1" ]; then
    echo "[WARN] EV_LOG_MODE=off; skipping visualization because no EV/EDL logs were saved."
fi
if ev_log_enabled && [ "$RUN_VIZ" = "1" ]; then
    echo ""
    echo "======================================================================"
    echo " Building EDL visualizations"
    echo "======================================================================"
    declare -a VIZ_ARGS=()
    if [ "$EDL_VIZ_STRICT_V6" = "1" ] || [ "$EDL_VIZ_STRICT_V6" = "true" ]; then
        VIZ_ARGS+=(--strict-v6)
    elif [ "$EDL_VIZ_STRICT_V6" = "auto" ]; then
        if grep -q "gated_primary_router" "$config" 2>/dev/null || [[ "$config_name" == *gated_primary_router* ]] || [[ "$config_name" == *v6* ]]; then
            VIZ_ARGS+=(--strict-v6)
        fi
    fi
    "$PYTHON_BIN" -u tools/visualize_edl_logs.py \
        --manifest "$edl_manifest" \
        --output-dir "$viz_output_dir" \
        --eval-log "$logfile" \
        --clean-nds "$CLEAN_NDS" \
        --clean-map "$CLEAN_MAP" \
        --jobs "$EDL_VIZ_JOBS" \
        --heartbeat-samples "$EDL_VIZ_HEARTBEAT_SAMPLES" \
        "${VIZ_ARGS[@]}"
fi

echo ""
echo "======================================================================"
echo " All evaluations complete! Results saved to:"
echo "   Log:     $logfile"
echo "   Metrics: $metrics_file"
echo "   Outputs: $output_dir/"
echo "======================================================================"
