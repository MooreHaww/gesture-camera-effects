"""Which cameras are plugged in, and which one to use.

On Windows the list comes from DirectShow, the same list and order OpenCV's DSHOW
backend opens by index, so the names shown in the picker match what opens. It is read
with plain ctypes COM calls (no extra package), which is quick and does not open any
camera. If that fails the first few indices are probed instead.

On Linux (the Raspberry Pi) the USB webcams are read from /sys/class/video4linux; this
is only used by the c key, startup still opens --camera or index 0 as before.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

PROBE_MAX = 4  # indices tried when the device list cannot be read


@dataclass(frozen=True)
class CameraInfo:
    index: int
    name: str

    def as_setting(self) -> dict:
        return {"index": self.index, "name": self.name}


def list_cameras() -> list[CameraInfo]:
    if sys.platform == "win32":
        names = dshow_names()
        if names is not None:
            return [CameraInfo(i, n) for i, n in enumerate(names)]
    elif sys.platform.startswith("linux"):
        cams = v4l2_usb_cameras()
        if cams:
            return cams
    return probe_cameras()


def choose_saved(cams: list[CameraInfo], saved) -> CameraInfo | None:
    """The saved camera if it is plugged in, else None.

    Matched by name first, since the index of the same webcam changes between laptops
    and USB ports; among cameras with the same name the saved index wins. A setting
    without a name (or a camera list without names) falls back to the index.
    """
    if not isinstance(saved, dict) or not cams:
        return None
    name, index = saved.get("name"), saved.get("index")
    if isinstance(name, str) and name:
        same = [c for c in cams if c.name == name]
        if same:
            return next((c for c in same if c.index == index), same[0])
        return None
    return next((c for c in cams if c.index == index), None)


def next_camera(cams: list[CameraInfo], current: int) -> CameraInfo | None:
    """The camera after the one with index current, wrapping around."""
    if not cams:
        return None
    for pos, c in enumerate(cams):
        if c.index == current:
            return cams[(pos + 1) % len(cams)]
    return cams[0]


def probe_cameras(limit: int = PROBE_MAX) -> list[CameraInfo]:
    """Open indices 0..limit-1 and keep the ones that open (slow: about 0.5 s each)."""
    import cv2
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    found = []
    for i in range(limit):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            found.append(CameraInfo(i, f"Camera {i + 1}"))
        cap.release()
    return found


def v4l2_usb_cameras() -> list[CameraInfo]:
    """USB webcams on Linux: the first video node of each (the others are metadata)."""
    cams = []
    for node in sorted(Path("/sys/class/video4linux").glob("video*"),
                       key=lambda p: int(p.name[5:]) if p.name[5:].isdigit() else 999):
        try:
            if (node / "index").read_text().strip() != "0":
                continue
            if "usb" not in str((node / "device").resolve()):
                continue
            name = (node / "name").read_text().strip()
            cams.append(CameraInfo(int(node.name[5:]), name or node.name))
        except (OSError, ValueError):
            continue
    return cams


# ---------------------------------------------------------------------------
# DirectShow device names (Windows)
# ---------------------------------------------------------------------------
_CLSID_SYSTEM_DEVICE_ENUM = "{62BE5D10-60EB-11D0-BD3B-00A0C911CE86}"
_IID_ICREATE_DEV_ENUM = "{29840822-5B84-11D0-BD3B-00A0C911CE86}"
_CLSID_VIDEO_INPUT_CATEGORY = "{860BB310-5D01-11D0-BD3B-00A0C911CE86}"
_IID_IPROPERTY_BAG = "{55272A00-42CB-11CE-8135-00AA004BB851}"
_VT_BSTR = 8


def dshow_names() -> list[str] | None:
    """Friendly names of the DirectShow video inputs in OpenCV's index order, or None
    when they cannot be read (not Windows, COM error)."""
    if sys.platform != "win32":
        return None
    try:
        return _dshow_names()
    except Exception:
        return None


def _dshow_names() -> list[str]:
    import ctypes
    from ctypes import POINTER, byref, c_long, c_ulong, c_void_p

    class GUID(ctypes.Structure):
        _fields_ = [("d1", ctypes.c_uint32), ("d2", ctypes.c_uint16),
                    ("d3", ctypes.c_uint16), ("d4", ctypes.c_ubyte * 8)]

    class VARIANT(ctypes.Structure):
        _fields_ = [("vt", ctypes.c_ushort), ("r1", ctypes.c_ushort),
                    ("r2", ctypes.c_ushort), ("r3", ctypes.c_ushort),
                    ("val", c_void_p), ("pad", c_void_p)]

    ole32, oleaut32 = ctypes.oledll.ole32, ctypes.windll.oleaut32

    def guid(text):
        g = GUID()
        ole32.CLSIDFromString(ctypes.c_wchar_p(text), byref(g))
        return g

    def method(obj, slot, *argtypes):
        """COM method number slot of obj (an interface pointer), returning an HRESULT."""
        vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
        return ctypes.WINFUNCTYPE(c_long, c_void_p, *argtypes)(vtbl[slot])

    def release(obj):
        if obj:
            method(obj, 2)(obj)

    try:
        ole32.CoInitializeEx(None, 0x2)   # apartment threaded, like OpenCV's DSHOW
        need_uninit = True
    except OSError:                       # already initialised in another mode: fine
        need_uninit = False

    names = []
    dev_enum, enum, = c_void_p(), c_void_p()
    try:
        ole32.CoCreateInstance(byref(guid(_CLSID_SYSTEM_DEVICE_ENUM)), None, 1,  # INPROC
                               byref(guid(_IID_ICREATE_DEV_ENUM)), byref(dev_enum))
        # ICreateDevEnum::CreateClassEnumerator; S_FALSE and a null enum = no devices
        method(dev_enum, 3, c_void_p, c_void_p, c_ulong)(
            dev_enum, byref(guid(_CLSID_VIDEO_INPUT_CATEGORY)), byref(enum), 0)
        if not enum:
            return []
        iid_bag = guid(_IID_IPROPERTY_BAG)
        while True:
            moniker, fetched = c_void_p(), c_ulong()
            if method(enum, 3, c_ulong, c_void_p, c_void_p)(enum, 1, byref(moniker),
                                                              byref(fetched)) != 0:
                break                     # IEnumMoniker::Next: S_FALSE when done
            bag, name = c_void_p(), None
            try:
                # IMoniker::BindToStorage (slot 9 after IUnknown, IPersist, IPersistStream)
                if method(moniker, 9, c_void_p, c_void_p, c_void_p, c_void_p)(
                        moniker, None, None, byref(iid_bag), byref(bag)) == 0:
                    var = VARIANT()
                    # IPropertyBag::Read
                    if method(bag, 3, ctypes.c_wchar_p, c_void_p, c_void_p)(
                            bag, "FriendlyName", byref(var), None) == 0:
                        if var.vt == _VT_BSTR and var.val:
                            name = ctypes.wstring_at(var.val)
                        oleaut32.VariantClear(byref(var))
            finally:
                release(bag)
                release(moniker)
            names.append(name or f"Camera {len(names) + 1}")
    finally:
        release(enum)
        release(dev_enum)
        if need_uninit:
            ole32.CoUninitialize()
    return names
