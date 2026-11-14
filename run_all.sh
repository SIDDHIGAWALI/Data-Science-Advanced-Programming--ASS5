#!/usr/bin/env bash
# One-click setup + pipeline on macOS/Linux. Then run:  streamlit run src/dashboard.py
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/generate_data.py
python src/train.py
python -m pytest -q tests
python src/build_deck.py
echo; echo "DONE. Start the API: python src/api.py | Dashboard: streamlit run src/dashboard.py"
