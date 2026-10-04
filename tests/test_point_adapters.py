"""Exercise point adapters with every native API replaced, never on the host."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from glide.computer import macos, windows
from glide.computer.point_types import PointTarget, point_box

# Originals at collection; each test explicitly restores only its tested method
# after replacing all of its native dependencies. conftest guards all other calls.
MAC_TARGET, MAC_REGION = macos.point_target, macos.point_region
WIN_TARGET, WIN_REGION = windows.point_target, windows.point_region


def test_windows_password_is_checked_before_reading_any_content(monkeypatch):
    class Secret:
        ControlTypeName = "EditControl"
        IsPassword = True

        @property
        def Name(self):
            pytest.fail("protected name must not be read")

    seen = []
    monkeypatch.setattr(windows, "auto", SimpleNamespace(ControlFromPoint=lambda *p: (seen.append(p), Secret())[1]))
    monkeypatch.setattr(windows, "_ui_value", lambda element: pytest.fail("protected value must not be read"))
    monkeypatch.setattr(windows, "point_target", WIN_TARGET)
    assert windows.point_target((204.2, 101.1)).protected
    assert seen == [(204, 101)]


def test_windows_reads_the_exact_control_and_preserves_its_text(monkeypatch):
    item = SimpleNamespace(ControlTypeName="ButtonControl", IsPassword=False, Name="保存 0003", HelpText="Save this document")
    monkeypatch.setattr(windows, "auto", SimpleNamespace(ControlFromPoint=lambda *p: item))
    monkeypatch.setattr(windows, "_ui_value", lambda element: "003")
    monkeypatch.setattr(windows, "point_target", WIN_TARGET)
    target = windows.point_target((10, 20))
    assert target.packet() == {"role": "AXButton", "label": "保存 0003", "value": "003", "help": "Save this document"}


@pytest.mark.parametrize("role,subrole", [("AXSecureTextField", ""), ("AXTextField", "AXSecureTextField")])
def test_macos_secure_roles_are_checked_before_content(monkeypatch, role, subrole):
    calls = []
    system, element = object(), object()
    monkeypatch.setattr(
        macos,
        "AS",
        SimpleNamespace(
            AXUIElementCreateSystemWide=lambda: system,
            AXUIElementSetMessagingTimeout=lambda *a: None,
            AXUIElementCopyElementAtPosition=lambda *a: (0, element),
            kAXRoleAttribute="AXRole",
            kAXValueAttribute="AXValue",
        ),
    )

    def attr(el, name):
        calls.append(name)
        if name not in {"AXRole", "AXSubrole"}:
            pytest.fail("protected contents must not be read")
        return role if name == "AXRole" else subrole

    monkeypatch.setattr(macos, "_ax_attr", attr)
    monkeypatch.setattr(macos, "point_target", MAC_TARGET)
    assert macos.point_target((20, 30)).protected
    assert set(calls) <= {"AXRole", "AXSubrole"}


def test_macos_hit_test_errors_do_not_fall_back_to_pixels(monkeypatch):
    monkeypatch.setattr(
        macos,
        "AS",
        SimpleNamespace(
            AXUIElementCreateSystemWide=lambda: object(),
            AXUIElementSetMessagingTimeout=lambda *a: None,
            AXUIElementCopyElementAtPosition=lambda *a: (1, None),
        ),
    )
    monkeypatch.setattr(macos, "point_target", MAC_TARGET)
    with pytest.raises(RuntimeError, match="hit test failed"):
        macos.point_target((20, 30))


@pytest.mark.parametrize("failed", [False, True])
def test_macos_captures_only_the_rectangle_and_removes_its_temp_file(monkeypatch, failed):
    calls = []
    monkeypatch.setattr(
        macos,
        "Quartz",
        SimpleNamespace(
            CGMainDisplayID=lambda: 1,
            CGDisplayBounds=lambda display: SimpleNamespace(size=SimpleNamespace(width=1200, height=800)),
        ),
    )

    def capture(args, **kwargs):
        calls.append(args)
        Image.new("RGB", (400, 400)).save(args[-1])  # synthetic Retina crop only
        if failed:
            raise RuntimeError("synthetic failure")

    monkeypatch.setattr(macos.subprocess, "run", capture)
    monkeypatch.setattr(macos, "point_region", MAC_REGION)
    if failed:
        with pytest.raises(RuntimeError, match="synthetic"):
            macos.point_region((200, 200), 100)
    else:
        image, box = macos.point_region((200, 200), 100)
        assert box == (100, 100, 300, 300) and image.size == (400, 400)
        image.close()
    assert calls[0][:3] == ["screencapture", "-x", "-R100,100,200,200"]
    assert not Path(calls[0][-1]).parent.exists()
    assert "glide-point-" in calls[0][-1]


@pytest.mark.parametrize("failed", [False, True])
def test_windows_copies_only_a_bounded_rectangle_and_cleans_up_native_handles(monkeypatch, failed):
    calls = []
    source = SimpleNamespace()
    bitmap = SimpleNamespace(
        CreateCompatibleBitmap=lambda dc, w, h: calls.append(("size", w, h)),
        GetHandle=lambda: "bitmap",
        GetInfo=lambda: {"bmBitsPixel": 32, "bmWidthBytes": 800},
        GetBitmapBits=lambda as_string: bytes([10, 20, 30, 255]) * 200 * 200,
    )

    def blit(dest, size, dc, origin, mode):
        calls.append(("blit", dest, size, origin))
        if failed:
            raise RuntimeError("synthetic failure")

    memory = SimpleNamespace(
        DeleteDC=lambda: calls.append(("delete-memory",)),
        SelectObject=lambda obj: (calls.append(("select", obj)), "previous")[1],
        BitBlt=blit,
    )
    source.CreateCompatibleDC = lambda: memory
    monkeypatch.setattr(windows, "win32api", SimpleNamespace(GetSystemMetrics=lambda n: (1200, 800)[n]))
    monkeypatch.setattr(windows, "win32con", SimpleNamespace(SRCCOPY=0xCC0020))
    monkeypatch.setattr(
        windows,
        "win32gui",
        SimpleNamespace(
            CreateDC=lambda *a: "display",
            DeleteDC=lambda dc: calls.append(("delete-display", dc)),
            DeleteObject=lambda obj: calls.append(("delete-bitmap", obj)),
        ),
    )
    monkeypatch.setattr(windows, "win32ui", SimpleNamespace(CreateDCFromHandle=lambda dc: source, CreateBitmap=lambda: bitmap))
    monkeypatch.setattr(windows, "point_region", WIN_REGION)
    if failed:
        with pytest.raises(RuntimeError, match="synthetic"):
            windows.point_region((200, 200), 100)
    else:
        image, box = windows.point_region((200, 200), 100)
        assert box == (100, 100, 300, 300) and image.size == (200, 200)
        assert image.getpixel((0, 0)) == (30, 20, 10)
        image.close()
    assert ("size", 200, 200) in calls
    assert ("blit", (0, 0), (200, 200), (100, 100)) in calls
    assert calls[-4:] == [("select", "previous"), ("delete-bitmap", "bitmap"), ("delete-memory",), ("delete-display", "display")]


def test_a_protected_target_cannot_be_turned_into_a_packet_and_a_packet_is_bounded():
    with pytest.raises(ValueError, match="Protected"):
        PointTarget("AXTextField", protected=True).packet()
    packet = PointTarget("r" * 500, "l" * 5000, "v" * 5000, "h" * 5000).packet()
    assert [len(packet[key]) for key in ("role", "label", "value", "help")] == [80, 1024, 2048, 1024]


@pytest.mark.parametrize("point, expected", [((0, 0), (0, 0, 160, 160)), ((1199, 799), (1039, 639, 1200, 800))])
def test_edge_crops_are_clamped_without_moving_the_point(point, expected):
    assert point_box(point, 160, (0, 0, 1200, 800)) == expected


@pytest.mark.parametrize(
    "point,radius", [((-1, 10), 160), ((1200, 5), 160), ((2, 3), 241), ((2, 3), 23), ((float("nan"), 3), 160)]
)
def test_secondary_display_and_invalid_crop_coordinates_are_refused(point, radius):
    with pytest.raises(ValueError):
        point_box(point, radius, (0, 0, 1200, 800))
