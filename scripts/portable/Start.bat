@echo off
rem Gesture Camera Effects in a window. Works from any folder or drive letter.
rem Extra flags are passed on, for example:  Start.bat --camera 1
start "" "%~dp0GestureFX.exe" %*
