@echo off
echo ========================================
echo DISABLING PROBLEMATIC NODES TEMPORARILY
echo ========================================

set COMFYUI_DIR=C:\Users\chira\Downloads\ninja downlaod\ComfyUI_windows_portable_nvidia_1\ComfyUI_windows_portable\ComfyUI

echo.
echo 1. Renaming problematic extensions to disable them...

if exist "%COMFYUI_DIR%\custom_nodes\comfyui_controlnet_aux" (
    echo Disabling comfyui_controlnet_aux...
    ren "%COMFYUI_DIR%\custom_nodes\comfyui_controlnet_aux" "comfyui_controlnet_aux.disabled"
)

if exist "%COMFYUI_DIR%\custom_nodes\ComfyUI-tbox" (
    echo Disabling ComfyUI-tbox...
    ren "%COMFYUI_DIR%\custom_nodes\ComfyUI-tbox" "ComfyUI-tbox.disabled"
)

if exist "%COMFYUI_DIR%\custom_nodes\comfyui_layerstyle" (
    echo Disabling comfyui_layerstyle...
    ren "%COMFYUI_DIR%\custom_nodes\comfyui_layerstyle" "comfyui_layerstyle.disabled"
)

echo.
echo 2. Clearing cache...
if exist "%TEMP%\comfyui" (
    rmdir /s /q "%TEMP%\comfyui" 2>nul
)

echo.
echo ========================================
echo PROBLEMATIC NODES DISABLED!
echo ========================================
echo.
echo Now:
echo 1. Start ComfyUI
echo 2. Load workflow_minimal.json
echo 3. Only our custom nodes will be available
echo 4. No more download errors!
echo.
echo To re-enable later, run: enable_nodes.bat
echo ========================================
echo.
pause
