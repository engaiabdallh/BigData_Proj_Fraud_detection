@echo off
echo ============================================
echo   Cleaning Up Project - Removing Junk Files
echo ============================================

echo Removing CRC files...
del /s /q *.crc 2>nul
echo [OK] CRC files removed

echo Removing _SUCCESS files...
for /r %%i in (_SUCCESS) do del /q "%%i" 2>nul
echo [OK] _SUCCESS files removed

echo Removing Jupyter cache...
if exist .virtual_documents rmdir /s /q .virtual_documents
echo [OK] Jupyter cache removed

echo Removing Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
echo [OK] Python cache removed

echo.
echo ============================================
echo   Cleanup Complete!
echo ============================================
echo.
echo Ready for distribution!
pause