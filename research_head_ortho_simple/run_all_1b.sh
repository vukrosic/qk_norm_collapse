#!/bin/bash

# Run Baseline
echo "Starting Baseline..."
python research_head_ortho_simple/run_full_1b.py --mode baseline

# Run Mode 1
echo "Starting Mode 1..."
python research_head_ortho_simple/run_full_1b.py --mode mode1

# Run Mode 3
echo "Starting Mode 3..."
python research_head_ortho_simple/run_full_1b.py --mode mode3

echo "All 1B experiments complete!"
