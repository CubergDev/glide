"""Raccoon pixel art, drawn in code. Each mood is a short loop of frames."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw

GRID_W, GRID_H = 43, 43  # logical pixels per frame

INK = (10, 18, 28, 255)
PAL = {
    "fur": "#9aa9b9",
    "fur2": "#6f8196",
    "pale": "#eef3f8",
    "pale2": "#c4cfdb",
    "mask": "#1b2230",
    "dark": "#2b3442",
    "ear": "#e59aa5",
    "mint": "#67e8ba",
    "mint2": "#2fae86",
    "slate": "#3a5672",
    "slate2": "#2a4058",
    "amber": "#ffd27a",
    "pink": "#ffadb5",
    "white": "#ffffff",
    "black": "#0a121c",
}


@dataclass(frozen=True)
class Pose:
    dx: int = 0
    dy: int = 0
    eyes: str = "open"  # open, closed, happy, wide, sad
    look: int = 0  # -1 left, 0 ahead, 1 right
    mouth: str = "none"  # none, smile, open, frown
    tail: int = 0  # -1, 0, 1 sway
    stride: int = 0  # -1, 0, 1 walking legs
    arms: str = "down"  # down, up, type_l, type_r
    perk: bool = False
    light: bool = True
    prop: str = ""  # "", question, sweat, sweat2, spark, sound, sound2, z
    laptop: bool = False


class Canvas:
    def __init__(self, pose: Pose):
        self.im = Image.new("RGBA", (GRID_W, GRID_H), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)
        self.ox, self.oy = 5 + pose.dx, 7 + pose.dy

    def _p(self, pts):
        return [(x + self.ox, y + self.oy) for x, y in pts]

    def rect(self, x0, y0, x1, y1, col):
        self.d.rectangle(self._p([(x0, y0), (x1, y1)]), fill=PAL[col])

    def ell(self, x0, y0, x1, y1, col, shade=None):
        """Flat ellipse; with `shade`, a lower-right cel-shadow crescent."""
        box = self._p([(x0, y0), (x1, y1)])
        if not shade:
            self.d.ellipse(box, fill=PAL[col])
            return
        self.d.ellipse(box, fill=PAL[shade])
        lit = Image.new("L", self.im.size, 0)
        ImageDraw.Draw(lit).ellipse(self._p([(x0, y0), (x1 - 2, y1 - 2)]), fill=255)
        clip = Image.new("L", self.im.size, 0)
        ImageDraw.Draw(clip).ellipse(box, fill=255)
        lit.paste(0, mask=Image.eval(clip, lambda v: 255 - v))
        self.im.paste(PAL[col], mask=lit)

    def poly(self, pts, col):
        self.d.polygon(self._p(pts), fill=PAL[col])

    def px(self, pts, col):
        for p in self._p(pts):
            self.d.point(p, fill=PAL[col])

    def line(self, pts, col):
        self.d.line(self._p(pts), fill=PAL[col])

    def outlined(self):
        """Thick-look 1px ink outline around the silhouette, the Codex pet style."""
        w, h = self.im.size
        src = self.im.load()
        out = self.im.copy()
        dst = out.load()
        for y in range(h):
            for x in range(w):
                if src[x, y][3]:
                    continue
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < w and 0 <= ny < h and src[nx, ny][3]:
                        dst[x, y] = INK
                        break
        return out


def draw(pose: Pose) -> Image.Image:
    c = Canvas(pose)
    t = pose.tail
    # Ringed tail behind the body.
    for i, (x, y) in enumerate([(23, 28), (27, 26), (30 + t, 22), (31 + t, 17)]):
        c.ell(x - 3, y - 3, x + 3, y + 3, "dark" if i % 2 else "fur", None if i % 2 else "fur2")
    c.ell(28 + t, 9, 33 + t, 14, "dark")
    # Ears.
    e = 1 if pose.perk else 0
    c.poly([(3, 12), (2, 0 - e), (12, 5)], "fur2")
    c.poly([(28, 12), (29, 0 - e), (19, 5)], "fur2")
    c.poly([(4, 8), (4, 3 - e), (8, 6)], "ear")
    c.poly([(27, 8), (27, 3 - e), (23, 6)], "ear")
    # Body and legs.
    lift_l, lift_r = (1, 0) if pose.stride < 0 else (0, 1) if pose.stride > 0 else (0, 0)
    c.rect(11, 30 - lift_l, 13, 31 - lift_l, "dark")
    c.rect(18, 30 - lift_r, 20, 31 - lift_r, "dark")
    c.rect(11, 23, 20, 30, "fur2")
    c.rect(11, 23, 19, 29, "fur")
    c.rect(13, 24, 18, 29, "pale2")
    c.rect(13, 24, 17, 28, "pale")
    # Head.
    c.ell(3, 4, 28, 25, "fur", "fur2")
    c.rect(15, 5, 16, 9, "fur2")
    c.ell(6, 14, 25, 25, "pale", "pale2")  # cheeks
    c.ell(5, 11, 15, 18, "mask")
    c.ell(16, 11, 26, 18, "mask")
    c.rect(7, 10, 13, 10, "pale")  # brows
    c.rect(18, 10, 24, 10, "pale")
    c.ell(11, 16, 20, 24, "pale")  # muzzle
    c.rect(14, 18, 17, 19, "black")  # nose
    c.px([(15, 18)], "slate")
    _eyes(c, pose)
    _mouth(c, pose)
    # Headset: band, cups with status light.
    c.line([(4, 11), (6, 6), (10, 3), (16, 2), (21, 3), (25, 6), (27, 11)], "black")
    c.rect(1, 13, 3, 19, "black")
    c.rect(28, 13, 30, 19, "black")
    c.rect(2, 15, 2, 16, "mint" if pose.light else "slate")
    c.rect(29, 15, 29, 16, "mint" if pose.light else "slate")
    _arms(c, pose)
    if pose.laptop:
        _laptop(c, pose)
    _prop(c, pose)
    return c.outlined()


def _eyes(c, pose):
    look = pose.look
    for x in (9, 20):
        x += look
        if pose.eyes in ("open", "wide"):
            top = 12 if pose.eyes == "wide" else 13
            c.rect(x, top, x + 2, 15, "mint")
            c.px([(x, top)], "white")
        elif pose.eyes == "closed":
            c.rect(x, 15, x + 2, 15, "mint2")
        elif pose.eyes == "happy":
            c.px([(x, 15), (x + 1, 14), (x + 2, 15)], "mint")
        elif pose.eyes == "sad":
            c.rect(x, 15, x + 2, 15, "mint2")
            c.px([(x if x < 16 else x + 2, 14)], "mint2")


def _mouth(c, pose):
    if pose.mouth == "smile":
        c.px([(13, 21), (14, 22), (15, 22), (16, 22), (17, 22), (18, 21)], "black")
    elif pose.mouth == "open":
        c.rect(14, 21, 17, 22, "black")
        c.rect(15, 22, 16, 22, "pink")
    elif pose.mouth == "frown":
        c.px([(13, 22), (14, 21), (15, 21), (16, 21), (17, 21), (18, 22)], "black")
    else:
        c.px([(15, 21), (16, 21)], "black")


def _arms(c, pose):
    a = pose.arms
    left = {"down": (8, 24, 10, 27), "up": (7, 17, 9, 21), "type_l": (9, 26, 11, 28), "type_r": (9, 25, 11, 26)}
    right = {
        "down": (21, 24, 23, 27),
        "up": (22, 17, 24, 21),
        "type_l": (20, 25, 22, 26),
        "type_r": (20, 26, 22, 28),
    }
    c.rect(*left.get(a, left["down"]), "fur2")
    c.rect(*right.get(a, right["down"]), "fur2")


def _laptop(c, pose):
    c.rect(7, 29, 24, 31, "slate2")
    c.rect(9, 23, 22, 29, "slate")
    c.rect(10, 24, 21, 28, "slate2")
    c.px([(15, 26), (16, 26)], "mint" if pose.light else "mint2")
    # paws resting on the keyboard edge
    c.rect(9 if pose.arms == "type_l" else 10, 28, 11, 29, "fur2")
    c.rect(20, 28, 22 if pose.arms == "type_r" else 21, 29, "fur2")


def _prop(c, pose):
    p = pose.prop
    if p == "question":
        c.rect(29, -4, 32, -4, "white")
        c.rect(32, -3, 32, -2, "white")
        c.rect(30, -1, 31, -1, "white")
        c.rect(30, 0, 30, 0, "white")
        c.rect(30, 2, 30, 2, "white")
    elif p in ("sweat", "sweat2"):
        y = 6 if p == "sweat" else 8
        c.rect(27, y, 27, y, "mint")
        c.rect(26, y + 1, 28, y + 2, "mint")
    elif p == "spark":
        for x, y in ((0, 0), (30, 1)):
            c.px([(x, y - 1), (x - 1, y), (x, y), (x + 1, y), (x, y + 1)], "amber")
    elif p in ("sound", "sound2"):
        c.px([(-1, 14), (-1, 15), (-1, 16), (-1, 17)], "mint")
        if p == "sound2":
            c.px([(-3, 12), (-3, 13), (-3, 18), (-3, 19), (-3, 14), (-3, 17)], "mint")
    elif p == "z":
        c.rect(28, -3, 31, -3, "white")
        c.px([(30, -2), (29, -1)], "white")
        c.rect(28, 0, 31, 0, "white")


def moods() -> dict[str, list[Pose]]:
    P = Pose
    return {
        "idle": [P(), P(), P(dy=1, tail=1), P(dy=1, tail=1, eyes="closed"), P(), P(tail=-1)],
        "listening": [
            P(perk=True, look=lk, light=on, eyes="wide" if lk == 0 else "open")
            for lk, on in ((0, True), (0, False), (-1, True), (-1, False), (1, True), (1, False))
        ],
        "typing": [P(perk=True, look=lk, eyes="open", tail=t) for lk, t in ((0, 0), (0, 1), (1, 1), (1, 0), (0, -1), (0, 0))],
        "thinking": [
            P(laptop=True, arms=a, look=lk, light=i % 2 == 0, tail=(i % 3) - 1)
            for i, (a, lk) in enumerate(
                [("type_l", -1), ("type_r", -1), ("type_l", 0), ("type_r", 1), ("type_l", 1), ("type_r", 0)]
            )
        ],
        "talking": [
            P(mouth="open", prop="sound"),
            P(mouth="smile", eyes="happy", prop="sound2"),
            P(mouth="open", prop="sound", tail=1),
            P(mouth="none", prop="sound2", tail=1),
        ],
        "question": [
            P(prop="question", look=lk, eyes=e, tail=t)
            for lk, e, t in ((1, "open", 0), (1, "open", 1), (1, "wide", 1), (0, "open", 0), (-1, "open", -1), (0, "closed", 0))
        ],
        "happy": [
            P(dy=1, eyes="happy"),
            P(dy=-2, arms="up", eyes="happy", mouth="open"),
            P(dy=-4, arms="up", eyes="happy", mouth="open", prop="spark", tail=1),
            P(dy=-2, arms="up", eyes="happy", mouth="smile", tail=-1),
            P(dy=0, eyes="happy", mouth="smile"),
        ],
        "sad": [
            P(dy=1, eyes="sad", mouth="frown", light=False, prop=s)
            for s in ("", "sweat", "sweat", "sweat2", "sweat2", "", "", "")
        ],
        "sleeping": [P(dy=1, eyes="closed", light=False, prop=z) for z in ("", "", "z", "z", "z", "")],
    }


FRAME_MS = {
    "idle": 260,
    "listening": 200,
    "typing": 240,
    "thinking": 150,
    "talking": 150,
    "question": 220,
    "happy": 130,
    "sad": 200,
    "sleeping": 420,
}


def render(pose: Pose, scale: int) -> Image.Image:
    return draw(pose).resize((GRID_W * scale, GRID_H * scale), Image.Resampling.NEAREST)
