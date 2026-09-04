#!/usr/bin/env bash
set -euo pipefail

# Generic targeted MultiCorrupt probe.
# Runs clean first, then a small corruption subset sequentially, using the same
# data/nuscenes backup + symlink assembly strategy as multicorrupt_evaluation.sh.

CONFIG=${CONFIG:-projects/configs/edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs_v10_router_selector_frozen.py}
MODEL_LABEL=${MODEL_LABEL:-$(basename "$CONFIG" .py)}
CKPT=${CKPT:-"$PWD/ckpts/${MODEL_LABEL}.pth"}

CORRUPTIONS=${CORRUPTIONS:-"beamsreducing fog pointsreducing spatialmisalignment temporalmisalignment motionblur"}
SEVERITY_LEVELS=${SEVERITY_LEVELS:-"1 2 3"}
read -r -a CORRUPTION_LIST <<< "$CORRUPTIONS"
read -r -a SEVERITY_LIST <<< "$SEVERITY_LEVELS"
DATA_SUBDIRS=(samples sweeps v1.0-trainval maps lidarseg)

MULTICORRUPT_ROOT=${MULTICORRUPT_ROOT:-/media/ting/Dataset/MultiCorrupt}
NUSCENES_DATA_DIR=${NUSCENES_DATA_DIR:-"$PWD/data/nuscenes"}
NUSCENES_BACKUP_DIR=${NUSCENES_BACKUP_DIR:-"${NUSCENES_DATA_DIR}.bak"}

OUT_DIR=${OUT_DIR:-"$PWD/MultiCorrupt_results/${MODEL_LABEL}_targeted_probe"}
LOGFILE="$OUT_DIR/evaluation_log.txt"
METRICS_FILE="$OUT_DIR/metrics_summary.txt"
EDL_LOG_ROOT="$OUT_DIR/edl_logs"
EDL_MANIFEST="$EDL_LOG_ROOT/log_manifest.tsv"
VIZ_OUTPUT_DIR="$OUT_DIR/visualizations"
CLEAN_OUT=${CLEAN_OUT:-"$OUT_DIR/clean_output.txt"}
CLEAN_LOG_DIR=${CLEAN_LOG_DIR:-"$EDL_LOG_ROOT/clean"}
CLEAN_NDS=${CLEAN_NDS:-auto}
CLEAN_MAP=${CLEAN_MAP:-auto}

TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-4}
RERUN_CLEAN=${RERUN_CLEAN:-0}
RERUN_EXISTING=${RERUN_EXISTING:-1}
RUN_VIZ=${RUN_VIZ:-1}
REPAIR_JSONL=${REPAIR_JSONL:-1}
EDL_VIZ_JOBS=${EDL_VIZ_JOBS:-8}
EDL_VIZ_HEARTBEAT_SAMPLES=${EDL_VIZ_HEARTBEAT_SAMPLES:-250}
EXTRA_CFG_OPTIONS=${EXTRA_CFG_OPTIONS:-}
EDL_VIZ_EXTRA_ARGS=${EDL_VIZ_EXTRA_ARGS:-}
EV_LOG_MODE=${EV_LOG_MODE:-full_stream}
EV_LOG_COMPRESSION=${EV_LOG_COMPRESSION:-gzip}
EV_LOG_FLUSH_INTERVAL=${EV_LOG_FLUSH_INTERVAL:-100}
EV_LOG_PREDICTION_FULL=${EV_LOG_PREDICTION_FULL:-0}
EV_LOG_SCENE_ONLY=${EV_LOG_SCENE_ONLY:-0}

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

if [ ! -f "$CONFIG" ]; then
    echo "[ERROR] Missing config: $CONFIG"
    exit 1
fi
if [ ! -f "$CKPT" ]; then
    echo "[ERROR] Missing checkpoint: $CKPT"
    exit 1
fi
if [ ! -d "$NUSCENES_DATA_DIR" ]; then
    echo "[ERROR] Missing nuScenes data dir: $NUSCENES_DATA_DIR"
    exit 1
fi
if [ -e "$NUSCENES_BACKUP_DIR" ]; then
    echo "[ERROR] Backup path already exists: $NUSCENES_BACKUP_DIR"
    echo "        Restore or remove it manually before running this probe."
    exit 1
fi

mkdir -p "$OUT_DIR" "$EDL_LOG_ROOT" "$VIZ_OUTPUT_DIR"
: > "$LOGFILE"

restored=false
cleanup() {
    if [ "$restored" = false ] && [ -d "$NUSCENES_BACKUP_DIR" ]; then
        echo "[CLEANUP] Restoring original data/nuscenes from backup..."
        rm -rf "$NUSCENES_DATA_DIR"
        mv "$NUSCENES_BACKUP_DIR" "$NUSCENES_DATA_DIR"
        restored=true
        echo "[CLEANUP] Restored."
    fi
}
trap cleanup EXIT INT TERM

echo "[INFO] Config:       $CONFIG"
echo "[INFO] Checkpoint:   $CKPT"
echo "[INFO] Model label:  $MODEL_LABEL"
echo "[INFO] Corruptions:  ${CORRUPTION_LIST[*]}"
echo "[INFO] Severities:   ${SEVERITY_LIST[*]}"
echo "[INFO] Output dir:   $OUT_DIR"
echo "[INFO] Data dir:     $NUSCENES_DATA_DIR"
echo "[INFO] Backup dir:   $NUSCENES_BACKUP_DIR"
echo "[INFO] EV log mode:  $EV_LOG_MODE ($EV_LOG_COMPRESSION)"
echo "[INFO] EV pred full: $EV_LOG_PREDICTION_FULL"
echo "[INFO] EV scene only: $EV_LOG_SCENE_ONLY"
echo "[INFO] Extra cfg:    ${EXTRA_CFG_OPTIONS:-<none>}"

assemble_nuscenes() {
    local corrupt_dir="$1"

    rm -rf "$NUSCENES_DATA_DIR"
    mkdir -p "$NUSCENES_DATA_DIR"

    for subdir in "${DATA_SUBDIRS[@]}"; do
        if [ -d "$corrupt_dir/$subdir" ] || [ -L "$corrupt_dir/$subdir" ]; then
            ln -s "$corrupt_dir/$subdir" "$NUSCENES_DATA_DIR/$subdir"
        elif [ -d "$NUSCENES_BACKUP_DIR/$subdir" ]; then
            ln -s "$NUSCENES_BACKUP_DIR/$subdir" "$NUSCENES_DATA_DIR/$subdir"
        fi
    done

    for pkl in "$NUSCENES_BACKUP_DIR"/nuscenes_infos_*.pkl; do
        [ -f "$pkl" ] && ln -s "$pkl" "$NUSCENES_DATA_DIR/$(basename "$pkl")"
    done
    if [ -f "$NUSCENES_BACKUP_DIR/nuscenes_dbinfos_train.pkl" ]; then
        ln -s "$NUSCENES_BACKUP_DIR/nuscenes_dbinfos_train.pkl" \
            "$NUSCENES_DATA_DIR/nuscenes_dbinfos_train.pkl"
    fi
}

run_eval() {
    local edl_dir="$1"
    local output_file="$2"
    local split="$3"
    local corruption="$4"
    local severity="$5"
    local -a cmd=(
        python tools/test.py "$CONFIG" "$CKPT" --eval=bbox
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
    EV_MODEL_LABEL="$MODEL_LABEL" \
        "${cmd[@]}" 2>&1 | tee "$output_file"
}

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
    local prediction_light sidecar_meta log_json wrote=0

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
    mkdir -p "$EDL_LOG_ROOT"
    {
        printf "split\tcorruption\tseverity\tedl_log_dir\tedl_log_json\tedl_scene_summary_jsonl\tedl_prediction_light_jsonl\tedl_query_semantic_jsonl\tedl_sidecar_meta_json\teval_output\n"

        write_manifest_rows "clean" "clean" "0" "$CLEAN_LOG_DIR" "$CLEAN_OUT" || true

        for corruption in "${CORRUPTION_LIST[@]}"; do
            for severity in "${SEVERITY_LIST[@]}"; do
                local log_dir="$EDL_LOG_ROOT/$corruption/severity_$severity"
                local output_file="$OUT_DIR/${corruption}_${severity}_output.txt"
                write_manifest_rows "corrupt" "$corruption" "$severity" "$log_dir" "$output_file" || true
            done
        done
    } > "$EDL_MANIFEST"
    echo "[EDL Viz] Manifest written to: $EDL_MANIFEST"
}

compute_probe_metrics() {
    python3 - "$LOGFILE" "$CLEAN_NDS" "$CLEAN_MAP" "$METRICS_FILE" "$MODEL_LABEL" <<'PYTHON_SCRIPT'
import re
import sys

logfile = sys.argv[1]
clean_nds = float(sys.argv[2])
clean_map = float(sys.argv[3])
metrics_file = sys.argv[4]
model_label = sys.argv[5]

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

pattern = re.compile(
    r"Corruption:\s*(\w+),\s*Severity:\s*(\d+),\s*NDS:\s*([\d.]+|N/A|ERROR),\s*mAP:\s*([\d.]+|N/A|ERROR)"
)
results = {}
with open(logfile, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if not match:
            continue
        try:
            results[(match.group(1), int(match.group(2)))] = (
                float(match.group(3)),
                float(match.group(4)),
            )
        except ValueError:
            continue

lines = []

def out(text=""):
    print(text)
    lines.append(text)

out("=" * 90)
out("  Targeted Resistance Ability (RA)")
out("=" * 90)
out(f"  Model: {model_label}")
out(f"  Clean baseline: NDS = {clean_nds:.4f}, mAP = {clean_map:.4f}")
out("")
out(f"{'Corruption':<25} {'Sev':>3} {'NDS':>8} {'mAP':>8} {'RA_NDS':>8} {'RA_mAP':>8}")
out("-" * 90)

ra_nds_by_corr = {}
ra_map_by_corr = {}
for corr in sorted(set(c for c, _ in results)):
    ra_nds_values = []
    ra_map_values = []
    for sev in (1, 2, 3):
        if (corr, sev) not in results:
            out(f"{corr:<25} {sev:>3} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
            continue
        nds, map_score = results[(corr, sev)]
        ra_nds = nds / clean_nds if clean_nds > 0 else 0.0
        ra_map = map_score / clean_map if clean_map > 0 else 0.0
        ra_nds_values.append(ra_nds)
        ra_map_values.append(ra_map)
        out(f"{corr:<25} {sev:>3} {nds:>8.4f} {map_score:>8.4f} {ra_nds:>8.4f} {ra_map:>8.4f}")
    if ra_nds_values:
        ra_nds_by_corr[corr] = sum(ra_nds_values) / len(ra_nds_values)
        ra_map_by_corr[corr] = sum(ra_map_values) / len(ra_map_values)
        out(f"  {'>> RA_' + corr:<23} {'':>3} {'':>8} {'':>8} {ra_nds_by_corr[corr]:>8.4f} {ra_map_by_corr[corr]:>8.4f}")
    out("")

if ra_nds_by_corr:
    mra_nds = sum(ra_nds_by_corr.values()) / len(ra_nds_by_corr)
    mra_map = sum(ra_map_by_corr.values()) / len(ra_map_by_corr)
    out("-" * 90)
    out(f"  {'mRA_targeted':<23} {'':>3} {'':>8} {'':>8} {mra_nds:>8.4f} {mra_map:>8.4f}")
    out("")

out("")
out("=" * 90)
out(f"  Relative Resistance Ability (RRA) vs {BASELINE_NAME} (NDS)")
out("  RRA_c = (RA_c * M_clean) / (RA_c_baseline * M_clean_baseline) - 1")
out("=" * 90)
out(f"  Your model:   Clean NDS = {clean_nds:.4f}, Clean mAP = {clean_map:.4f}")
out(f"  {BASELINE_NAME}: Clean NDS = {BASELINE_CLEAN_NDS:.4f}, Clean mAP = {BASELINE_CLEAN_MAP:.4f}")
out("")
out(f"{'Corruption':<25} {'RA (yours)':>13} {'RA (base)':>10} {'RRA (%)':>10}")
out("-" * 64)

rra_nds_values = []
for corr in sorted(set(c for c, _ in results)):
    if corr not in ra_nds_by_corr or corr not in BASELINE_RA_NDS:
        out(f"{corr:<25} {'N/A':>13} {'N/A':>10} {'N/A':>10}")
        continue
    ra_ours = ra_nds_by_corr[corr]
    ra_base = BASELINE_RA_NDS[corr]
    rra_pct = ((ra_ours * clean_nds) / (ra_base * BASELINE_CLEAN_NDS) - 1.0) * 100.0
    rra_nds_values.append(rra_pct)
    out(f"{corr:<25} {ra_ours:>13.4f} {ra_base:>10.4f} {rra_pct:>+10.3f}")

if rra_nds_values:
    mrra = sum(rra_nds_values) / len(rra_nds_values)
    out("-" * 64)
    out(f"{'mRRA_targeted':<25} {'':>13} {'':>10} {mrra:>+10.3f}")
out("")

out("=" * 90)
out(f"  Reference: Original {BASELINE_NAME} MultiCorrupt baseline")
out(f"  Clean mAP = {BASELINE_CLEAN_MAP:.4f}, Clean NDS = {BASELINE_CLEAN_NDS:.4f}")
out("  mRA_NDS = 0.8422, mRA_mAP = 0.7470, mRRA = +1.251%")
out("  Secondary reference: official MEFormer VoV leaderboard mRA = 0.856, mRRA = +7.391%")
out("=" * 90)

with open(metrics_file, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nMetrics saved to: {metrics_file}")
PYTHON_SCRIPT
}

echo ""
echo "======================================================================"
echo " Backing up original data/nuscenes"
echo "======================================================================"
mv "$NUSCENES_DATA_DIR" "$NUSCENES_BACKUP_DIR"

if [ "$CLEAN_NDS" = "auto" ] || [ "$CLEAN_MAP" = "auto" ]; then
    echo ""
    echo "======================================================================"
    echo " [0/N] Evaluating CLEAN nuScenes baseline"
    echo "======================================================================"

    if [ "$RERUN_CLEAN" = "0" ] && [ -f "$CLEAN_OUT" ] && \
       grep -q "NDS" "$CLEAN_OUT" 2>/dev/null && \
       has_manifest_logs "clean" "clean" "0" "$CLEAN_LOG_DIR" "$CLEAN_OUT"; then
        echo "[SKIP] Clean output already exists: $CLEAN_OUT"
    else
        assemble_nuscenes "$NUSCENES_BACKUP_DIR"
        rm -rf "$CLEAN_LOG_DIR"
        mkdir -p "$CLEAN_LOG_DIR" "$(dirname "$CLEAN_OUT")"
        run_eval "$CLEAN_LOG_DIR" "$CLEAN_OUT" "clean" "clean" "0"
        rm -rf "$NUSCENES_DATA_DIR"
    fi

    CLEAN_NDS=$(grep -oP 'NDS:\s*\K[0-9.]+' "$CLEAN_OUT" 2>/dev/null | tail -1 || true)
    CLEAN_MAP=$(grep -oP 'mAP:\s*\K[0-9.]+' "$CLEAN_OUT" 2>/dev/null | tail -1 || true)
    if [ -z "$CLEAN_NDS" ] || [ -z "$CLEAN_MAP" ]; then
        echo "[ERROR] Failed to extract clean NDS/mAP from: $CLEAN_OUT"
        exit 1
    fi
fi

echo ""
echo "  Clean baseline: NDS=$CLEAN_NDS, mAP=$CLEAN_MAP"
echo ""

total=$((${#CORRUPTION_LIST[@]} * ${#SEVERITY_LIST[@]}))
count=0
for corruption in "${CORRUPTION_LIST[@]}"; do
    for severity in "${SEVERITY_LIST[@]}"; do
        count=$((count + 1))
        corrupt_dir="$MULTICORRUPT_ROOT/$corruption/$severity"
        output_file="$OUT_DIR/${corruption}_${severity}_output.txt"
        edl_log_dir="$EDL_LOG_ROOT/$corruption/severity_$severity"

        echo ""
        echo "======================================================================"
        echo " [$count/$total] Corruption=$corruption, Severity=$severity"
        echo "======================================================================"

        if [ "$RERUN_EXISTING" = "0" ] && [ -f "$output_file" ] && \
           grep -q "NDS" "$output_file" 2>/dev/null && \
           has_manifest_logs "corrupt" "$corruption" "$severity" "$edl_log_dir" "$output_file"; then
            echo "[SKIP] Output already exists: $output_file"
            nds=$(grep -oP 'NDS:\s*\K[0-9.]+' "$output_file" | tail -1 || echo "N/A")
            map_score=$(grep -oP 'mAP:\s*\K[0-9.]+' "$output_file" | tail -1 || echo "N/A")
            echo "Corruption: $corruption, Severity: $severity, NDS: $nds, mAP: $map_score" >> "$LOGFILE"
            continue
        fi

        if [ ! -d "$corrupt_dir" ]; then
            echo "[ERROR] Corruption directory not found: $corrupt_dir"
            echo "Corruption: $corruption, Severity: $severity, NDS: ERROR, mAP: ERROR" >> "$LOGFILE"
            continue
        fi

        assemble_nuscenes "$corrupt_dir"
        echo "  Assembly complete. Running evaluation..."
        rm -rf "$edl_log_dir"
        mkdir -p "$edl_log_dir"
        run_eval "$edl_log_dir" "$output_file" "corrupt" "$corruption" "$severity"

        nds=$(grep -oP 'NDS:\s*\K[0-9.]+' "$output_file" | tail -1 || echo "N/A")
        map_score=$(grep -oP 'mAP:\s*\K[0-9.]+' "$output_file" | tail -1 || echo "N/A")
        echo "Corruption: $corruption, Severity: $severity, NDS: $nds, mAP: $map_score" >> "$LOGFILE"
        echo "  => NDS: $nds, mAP: $map_score"
        echo "  => EDL log dir: $edl_log_dir"
        rm -rf "$NUSCENES_DATA_DIR"
    done
done

echo ""
echo "======================================================================"
echo " Restoring original data/nuscenes"
echo "======================================================================"
rm -rf "$NUSCENES_DATA_DIR"
mv "$NUSCENES_BACKUP_DIR" "$NUSCENES_DATA_DIR"
restored=true

echo ""
echo "======================================================================"
echo " Computing targeted RA / RRA metrics"
echo "======================================================================"
compute_probe_metrics

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
    if [ "$REPAIR_JSONL" = "1" ] && [ "$EV_LOG_COMPRESSION" = "none" ]; then
        python3 tools/repair_edl_jsonl.py --manifest "$EDL_MANIFEST" --repair-tail
    elif [ "$REPAIR_JSONL" = "1" ]; then
        echo "[EDL Viz] Skipping JSONL repair for compressed EV logs."
    fi
    read -r -a edl_viz_extra_args_array <<< "$EDL_VIZ_EXTRA_ARGS"
    python3 -u tools/visualize_edl_logs.py \
        --manifest "$EDL_MANIFEST" \
        --output-dir "$VIZ_OUTPUT_DIR" \
        --eval-log "$LOGFILE" \
        --clean-nds "$CLEAN_NDS" \
        --clean-map "$CLEAN_MAP" \
        --jobs "$EDL_VIZ_JOBS" \
        --heartbeat-samples "$EDL_VIZ_HEARTBEAT_SAMPLES" \
        "${edl_viz_extra_args_array[@]}"
fi

echo ""
echo "======================================================================"
echo " Targeted probe complete"
echo "   Log:     $LOGFILE"
echo "   Metrics: $METRICS_FILE"
echo "   Outputs: $OUT_DIR/"
echo "======================================================================"
