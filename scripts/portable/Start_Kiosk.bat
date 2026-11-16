@echo off
rem Event mode: fullscreen, hidden cursor, larger UI. q or Esc quits.
rem
rem To show an event name in the top-left corner, type it between = and the closing
rem quote on the next line, for example:  set "BRAND=Spring Fair 2027"
rem Left empty, no name is shown.
set "BRAND="

if not defined BRAND goto nobrand
start "" "%~dp0GestureFX.exe" --kiosk --brand "%BRAND%" %*
goto :eof

:nobrand
start "" "%~dp0GestureFX.exe" --kiosk %*
