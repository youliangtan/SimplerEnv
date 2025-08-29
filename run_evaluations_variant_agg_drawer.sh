#!/usr/bin/env bash
# Usage: bash run_batch_fullcmds.bash PORT_NUMBER
# Example: bash run_batch_fullcmds.bash 5556

set -u
mkdir -p logs

PORT="${1:-5556}"
LOG_FILE="logs/evaluation_results-${PORT}.log"
: > "$LOG_FILE"

# Collect multi-line commands into an array via a helper.
declare -a COMMANDS=()

add_cmd() {
  # Reads a heredoc from stdin and appends it as one array element.
  local _cmd
  _cmd="$(cat)"
  # Ignore empty blocks
  [[ -n "${_cmd//[[:space:]]/}" ]] && COMMANDS+=("$_cmd")
}

add_cmd <<'CMD'
python eval_simpler.py
  --env_name OpenTopDrawerCustomInScene-v0
  --scene_name frl_apartment_stage_simple
  --robot_init_x_range 0.65 0.85 3
  --robot_init_y_range -0.2 0.2 3
  --robot_init_rot_quat_center 0 0 0 1
  --robot_init_rot_rpy_range 0 0 1 0 0 1 0.0 0.0 1
  --obj_init_x_range 0 0 1
  --obj_init_y_range 0 0 1
  --additional_env_build_kwargs light_mode=simple shader_dir=rt
  --episode_length 300
CMD

add_cmd <<'CMD'
python eval_simpler.py
  --env_name CloseBottomDrawerCustomInScene-v0
  --scene_name frl_apartment_stage_simple
  --robot_init_x_range 0.65 0.85 3
  --robot_init_y_range -0.2 0.2 3
  --robot_init_rot_quat_center 0 0 0 1
  --robot_init_rot_rpy_range 0 0 1 0 0 1 0.0 0.0 1
  --obj_init_x_range 0 0 1
  --obj_init_y_range 0 0 1
  --additional_env_build_kwargs light_mode=simple shader_dir=rt
  --episode_length 300
CMD

# Add more blocks as needed:
# add_cmd <<'CMD'
# python eval_simpler.py \
#   ...your other config...
# CMD

# ----------------------------------

echo "Starting evaluations on port ${PORT}..." | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"

for idx in "${!COMMANDS[@]}"; do
  cmd="${COMMANDS[$idx]}"
  full_cmd="$cmd --robot_type google --groot_port ${PORT}"

  echo "Running: $full_cmd" | tee -a "$LOG_FILE"
  echo "--------------------------------------" | tee -a "$LOG_FILE"

  output=$($full_cmd 2>&1)
  exit_code=$?

  last_lines=$(echo "$output" | tail -n 2)

  {
    echo "Command: $full_cmd"
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