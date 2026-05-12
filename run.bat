@echo off
echo ============================================
echo   NeuroScan Pro - Startup Script
echo ============================================
echo.
 
echo [1/4] Installing dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies.
    echo Make sure Python is installed and added to PATH.
    pause
    exit /b 1
)
 
echo.
echo [2/4] Setting up database...
python manage.py migrate
if %errorlevel% neq 0 (
    echo ERROR: Database migration failed.
    pause
    exit /b 1
)
 
echo.
echo [3/4] Training ML model...
IF EXIST "ml_model\tumor_model.pkl" (
    echo  ML model already exists - skipping training.
) ELSE (
    IF EXIST "kaggle_dataset\kaggle_3m" (
        echo  Found Kaggle dataset - training with real MRI data...
        python ml_model/train_model.py --dataset "kaggle_dataset\kaggle_3m"
    ) ELSE IF EXIST "kaggle_dataset" (
        echo  Found kaggle_dataset folder - training with it...
        python ml_model/train_model.py --dataset "kaggle_dataset"
    ) ELSE (
        echo  No Kaggle dataset found - training with synthetic data...
        python ml_model/train_model.py
    )
    if %errorlevel% neq 0 (
        echo ERROR: ML model training failed.
        pause
        exit /b 1
    )
)
 
echo.
echo [4/4] Starting server...
echo ============================================
echo   Open your browser at: http://127.0.0.1:8000
echo ============================================
echo.
python manage.py runserver
 
pause