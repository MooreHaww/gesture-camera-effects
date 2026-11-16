GESTURE CAMERA EFFECTS - PORTABLE WINDOWS VERSION
=================================================

Runs on Windows 10 or 11 (64-bit). No Python and no install needed. The folder can
live anywhere: a USB stick, the desktop, any drive letter.

RUN IT
  Start.bat         opens the app in a window
  Start_Kiosk.bat   event mode: fullscreen, hidden mouse cursor, larger UI
                    (to show an event name, open it in Notepad and fill in BRAND)
  q or Esc quits.

  Plug in the webcam BEFORE starting. Close Teams, Zoom, Skype, the Camera app and
  browser video calls first: Windows lets only one app use a camera at a time.

PICKING A CAMERA
  With more than one camera (say the laptop's own plus the USB webcam) the first start
  shows a picker with a live picture of each. Click one, or press its number, or use
  the arrow keys and Enter.
  The choice is saved in settings.json in this folder and used next time. If that
  camera is not plugged in, the picker comes back.
  While running, press  c  to switch to the next camera (also saved).
  To force a camera:  Start.bat --camera 1   (0 is the first camera)
  To pick again from scratch, delete settings.json.

FIRST START: "WINDOWS PROTECTED YOUR PC"
  The app is not signed, so SmartScreen may warn. Click "More info", then
  "Run anyway". It asks only once per computer.

CAMERA BLOCKED OR "BUSY"
  If the app says the camera is busy or there is no camera:
  1. Close other apps that use the camera.
  2. Windows Settings > Privacy & security > Camera:
       "Camera access" On, and "Let desktop apps access your camera" On.
  3. Unplug the webcam, plug it into another USB port, wait a few seconds.
  The app keeps retrying; there is no need to restart it.

KEYS
  1-9, 0  choose a mode        p  photo           c  next camera
  x       clear neon drawing   i  info card       f  fullscreen on/off
  v       FPS                  l  hand skeleton   s  screenshot (to debug\)
  q, Esc  quit

FILES THE APP WRITES (in this folder)
  photos\        photos taken with two peace signs or p
  debug\         screenshots (s) and landmark checks (d)
  settings.json  the chosen camera
  crash.log      what happened, if something goes wrong; send this with a bug report
  If this folder is read-only (a locked USB stick), the app says so and saves to
  %TEMP%\GestureFX instead.

CHECK THE INSTALL WITHOUT A CAMERA
  GestureFX.exe --self-test   then open crash.log: the last line should start with
  "Self-test OK".
