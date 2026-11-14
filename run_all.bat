@echo off
REM One-click setup + pipeline on Windows. Then run:  streamlit run src\dashboard.py
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
python src\generate_data.py
python src\train.py
python -m pytest -q tests
python src\build_deck.py
echo.
echo DONE. Start the API:        python src\api.py
echo       Start the dashboard: streamlit run src\dashboard.py
