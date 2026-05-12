#!/bin/bash
echo "============================================"
echo " NeuroScan Pro - Startup Script"
echo "============================================"

echo ""
echo "[1/4] Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "[2/4] Training ML model..."
python3 ml_model/train_model.py

echo ""
echo "[3/4] Setting up database..."
python3 manage.py migrate

echo ""
echo "[4/4] Starting server at http://127.0.0.1:8000"
python3 manage.py runserver
