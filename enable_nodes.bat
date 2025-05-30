@echo off
echo ========================================
echo RE-ENABLING DISABLED NODES
echo ========================================

set COMFYUI_DIR=C:\Users\chira\Downloads\ninja downlaod\ComfyUI_windows_portable_nvidia_1\ComfyUI_windows_portable\ComfyUI

echo.
echo Re-enabling previously disabled extensions...

if exist "%COMFYUI_DIR%\custom_nodes\comfyui_controlnet_aux.disabled" (
    echo Re-enabling comfyui_controlnet_aux...
    ren "%COMFYUI_DIR%\custom_nodes\comfyui_controlnet_aux.disabled" "comfyui_controlnet_aux"
)

if exist "%COMFYUI_DIR%\custom_nodes\ComfyUI-tbox.disabled" (
    echo Re-enabling ComfyUI-tbox...
    ren "%COMFYUI_DIR%\custom_nodes\ComfyUI-tbox.disabled" "ComfyUI-tbox"
)

if exist "%COMFYUI_DIR%\custom_nodes\comfyui_layerstyle.disabled" (
    echo Re-enabling comfyui_layerstyle...
    ren "%COMFYUI_DIR%\custom_nodes\comfyui_layerstyle.disabled" "comfyui_layerstyle"
)

echo.
echo ========================================
echo NODES RE-ENABLED!
echo ========================================
echo.
pause
