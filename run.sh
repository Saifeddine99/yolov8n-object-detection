#!/bin/bash

# Activate virtual environment if existing:
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py