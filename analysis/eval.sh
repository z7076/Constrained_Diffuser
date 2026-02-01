#!/bin/bash

# --- Configuration ---
# Define the sweep files to run, separated by spaces
SWEEP_FILES=(
  "default_inv_1.jsonl"
  "default_inv_2.jsonl"
  "default_inv_3.jsonl"
  "default_inv_4.jsonl"
)
# Set the name for the output log file
LOG_FILE="eval_inv_cbf_angular_s+_num.log"
# Set the path to your Python script (assuming it accepts --sweep_file argument)
PYTHON_SCRIPT="eval.py"
# Set the python executable if needed (e.g., python3 or /path/to/your/venv/bin/python)
PYTHON_CMD="python"
# --- End Configuration ---

# Optional: Clear the log file before starting, or comment out to append to existing file
# > "$LOG_FILE"

# Get the total number of sweep files
NUM_SWEEPS=${#SWEEP_FILES[@]}

echo "Starting $NUM_SWEEPS evaluation runs using specific sweep files. Logging to $LOG_FILE" | tee -a "$LOG_FILE"
echo "======================================================" | tee -a "$LOG_FILE"

# Initialize a counter for the run number
run_count=0

# Loop through each sweep file defined in the SWEEP_FILES array
for sweep_file in "${SWEEP_FILES[@]}"
do
  # Increment the run counter
  ((run_count++))

  # Get the current timestamp
  TIMESTAMP_START=$(date +"%Y-%m-%d %H:%M:%S")
  echo "--------------------------------------" | tee -a "$LOG_FILE"
  echo "Starting Run $run_count/$NUM_SWEEPS at $TIMESTAMP_START" | tee -a "$LOG_FILE"
  echo "Using sweep file: $sweep_file" | tee -a "$LOG_FILE"
  echo "--------------------------------------" | tee -a "$LOG_FILE"

  # Construct the command with the --sweep_file argument
  COMMAND="$PYTHON_CMD $PYTHON_SCRIPT --sweep_file $sweep_file"
  echo "Executing: $COMMAND" | tee -a "$LOG_FILE"

  # Execute the Python script with the specific sweep file argument
  # '>> "$LOG_FILE"' appends standard output (stdout) to the log file
  # '2>&1' redirects standard error (stderr) to the same place as stdout (the log file)
  stdbuf -oL -eL $COMMAND >> "$LOG_FILE" 2>&1

  # Capture the exit code of the last command (the python script)
  EXIT_CODE=$?

  TIMESTAMP_END=$(date +"%Y-%m-%d %H:%M:%S")
  # Check if the script executed successfully (exit code 0)
  if [ $EXIT_CODE -eq 0 ]; then
    echo "Run $run_count ($sweep_file) finished successfully at $TIMESTAMP_END." | tee -a "$LOG_FILE"
  else
    # If the script failed, log the exit code
    echo "Run $run_count ($sweep_file) failed with exit code $EXIT_CODE at $TIMESTAMP_END. Check $LOG_FILE for details." | tee -a "$LOG_FILE"
  fi

  # Optional: Add a small delay (in seconds) between runs if needed
  # echo "Waiting 5 seconds before next run..." | tee -a "$LOG_FILE"
  # sleep 5

done

echo "======================================================" | tee -a "$LOG_FILE"
echo "All $NUM_SWEEPS runs completed. Full output logged to $LOG_FILE" | tee -a "$LOG_FILE"

exit 0
