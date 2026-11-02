#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "🎓 Student Dropout Risk Prediction Pipeline & Dashboard"
echo "=========================================================="

cd "$(dirname "$0")"

# 1. Run the PySpark Pipeline
echo -e "\n▶ Step 1: Running PySpark Training & Scoring Pipeline..."
python3 -m src.pipeline_runner

# 2. Run automated tests
echo -e "\n▶ Step 2: Running PyTest Suite..."
pytest -v

# 3. Launch Streamlit Dashboard
echo -e "\n▶ Step 3: Launching Streamlit Early Warning Dashboard on port 8501..."
streamlit run dashboard/app.py --server.port 8501
