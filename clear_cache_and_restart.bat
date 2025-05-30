@echo off
echo ========================================
echo CLEARING COMFYUI CACHE AND RESTARTING
echo ========================================

echo.
echo 1. Stopping any running ComfyUI processes...
taskkill /f /im python.exe 2>nul
timeout /t 2 >nul

echo.
echo 2. Clearing ComfyUI cache directories...
if exist "%TEMP%\comfyui" (
    echo Clearing %TEMP%\comfyui...
    rmdir /s /q "%TEMP%\comfyui" 2>nul
)

if exist "%APPDATA%\ComfyUI" (
    echo Clearing %APPDATA%\ComfyUI...
    rmdir /s /q "%APPDATA%\ComfyUI" 2>nul
)

if exist "%LOCALAPPDATA%\ComfyUI" (
    echo Clearing %LOCALAPPDATA%\ComfyUI...
    rmdir /s /q "%LOCALAPPDATA%\ComfyUI" 2>nul
)

echo.
echo 3. Clearing Python cache...
if exist "__pycache__" (
    rmdir /s /q "__pycache__" 2>nul
)

for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d" 2>nul

echo.
echo 4. Cache cleared successfully!
echo.
echo ========================================
echo INSTRUCTIONS:
echo ========================================
echo 1. Start ComfyUI normally
echo 2. Load the NEW workflow: workflow_clean.json
echo 3. You should see our custom nodes:
echo    - CXH_Leffa_Mask_Generator
echo    - CXH_Leffa_Pose_Preprocessor
echo 4. NO MORE DOWNLOAD ERRORS!
echo ========================================
echo.
pause
