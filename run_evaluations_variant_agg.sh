#!/usr/bin/env bash
# Usage: bash run_batch_fullcmds.bash PORT_NUMBER
# Example: bash run_batch_fullcmds.bash 5556

set -u
mkdir -p logs

PORT="${1:-5556}"
LOG_FILE="logs/evaluation_results-${PORT}.log"
: > "$LOG_FILE"

# TODO: Add more!!!!!!!!!!!!!
read -r -d '' COMMANDS <<EOF
python eval_simpler.py --env_name OpenDrawerCustomInScene-v0 --scene_name frl_apartment_stage_simple --env_reset_options $(cat env_reset_options1.json) --additional_env_build_kwargs station_name=mk_station_recolor light_mode=simple disable_bad_material=True urdf_version=recolor_cabinet_visual_matching_1 shader_dir=rt --episode_length 300

EOF
# ----------------------------------

echo "Starting evaluations on port ${PORT}..." | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"

for CMD in "${COMMANDS[@]}"; do
  [[ -z "$CMD" || "$CMD" =~ ^# ]] && continue
  CMD="$CMD --robot_type google --groot_port ${PORT}"

  echo "Running: $CMD" | tee -a "$LOG_FILE"
  echo "--------------------------------------" | tee -a "$LOG_FILE"

  output=$($CMD 2>&1)
  exit_code=$?

  last_lines=$(echo "$output" | tail -n 2)

  {
    echo "Command: $CMD"
    echo "Last 2 lines:"
    echo "$last_lines"
    echo "Exit code: $exit_code"
    echo "--------------------------------------"
    echo
  } >> "$LOG_FILE"

  echo "Last 2 lines:"
  echo "$last_lines"
  echo "Exit code: $exit_code"
  echo
done <<< "$COMMANDS"

echo "All evaluations completed. Results saved to $LOG_FILE"