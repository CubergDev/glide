# Recording the demo on macOS

How to record the screen and your voice for [DEMO.md](DEMO.md), with tools this Mac already has, how to trim and export, where the
file goes, and what to check before anyone else sees it. Nothing here was run by whoever wrote it: the tool inventory below came from
`which` and `ls` only, the flags from `man screencapture` and `ffmpeg -hide_banner -devices`.

Recording the screen and the microphone is itself something that takes over this machine's screen and audio: it needs your yes
(AGENTS.md), and macOS needs Screen Recording (and Microphone, with audio) permission for the app that records.

## What is installed here (checked with `which` and `ls`)

| Tool | Found |
|---|---|
| `screencapture` (built in) | `/usr/sbin/screencapture` |
| QuickTime Player (built in) | `/System/Applications/QuickTime Player.app` |
| Screenshot toolbar (built in) | Command-Shift-5 (no path to check) |
| `ffmpeg`, `ffprobe` | `/opt/homebrew/bin/ffmpeg`, `/opt/homebrew/bin/ffprobe` (ffmpeg 9.0.2, with `libx264`, `h264_videotoolbox`, `aac`, and the `avfoundation` input device) |
| `sox` | not found |
| A loopback audio driver (BlackHole, Loopback, Soundflower) | none found. `/Library/Audio/Plug-Ins/HAL` holds only `ParrotAudioPlugin.driver`, which this document does not identify; do not rely on it |
| `swift` | `/usr/bin/swift` (for scene f's app build) |

Run the same check on the machine you record on: `which screencapture ffmpeg ffprobe sox; ls -d "/System/Applications/QuickTime Player.app"; ls /Library/Audio/Plug-Ins/HAL`.

## Audio: what can and cannot be recorded

- The built-in recorders capture your **microphone** (the default input), not the Mac's own sound. With a loopback driver you could
  capture Glide's voice too; none is installed here, and installing one is out of scope.
- So: use a **headset** for scene b (it keeps the echo out of the microphone), and **show Glide's side as text on the screen**: say aloud
  what it printed, or use the large terminal. A recording made with speakers picks up Glide's voice as an echo through the microphone,
  which is also what confuses the demo itself.
- Record a ten-second test and listen to it before the take.

## Settings

- **Resolution.** Record one display at a size that stays readable when shrunk: set the display scaling to 1920 x 1080 (or the nearest
  "looks like" size) in System Settings > Displays, or record a region. A Retina capture is 2x that and big; scale it down when
  exporting (below). This machine reports a 2560 x 1664 Retina built-in and a 1920 x 1200 external display; choose one and record that
  one (`-D 1` is the main display, `-D 2` the second).
- **Frame rate.** `screencapture` has no frame-rate option in its manual: you get what macOS gives (often 60 on a Retina display).
  Export at 30 frames per second (below), which is plenty for a terminal and a browser. If you need a fixed rate while recording,
  `ffmpeg` can set `-framerate 30` on the `avfoundation` input (see the second method).
- **Terminal font** at 18 to 20 points, light or dark but high contrast; browser zoom 125 percent.
- **Length.** 5:45 planned; allow 8 minutes of raw footage and cut.

## Method 1: `screencapture` (built in, one command)

Flags, from `man screencapture`: `-v` records video of the screen, `-V <seconds>` records for that many seconds, `-g` adds audio from
the default input, `-G <id>` a specific audio source, `-k` shows clicks, `-x` plays no sound, `-D <display>` picks the display,
`-R x,y,w,h` records a rectangle. The manual does not say how a bare `-v` is stopped, so use `-V` and rehearse.

```sh
mkdir -p ~/Movies/glide-demo      # OUTSIDE the repository
# TAKES OVER THE MACHINE: records the screen and the default microphone for 420 seconds
screencapture -x -v -g -k -D 1 -V 420 ~/Movies/glide-demo/raw-take1.mov
```

Start it from a terminal window that is not in the picture, or a second display, and start the demo after it begins. The terminal
(or QuickTime) that runs it needs Screen Recording and Microphone permission. A countdown and the menu bar recording indicator are
shown by macOS and may be in the picture.

## Method 2: QuickTime Player (built in, with a window)

File > New Screen Recording. In the options menu choose the **Microphone** (your headset), "Show Mouse Clicks in Recording", and the
destination. Click to record the whole screen or drag a region. Stop from the menu bar. Save to `~/Movies/glide-demo/`.

## Method 3: `ffmpeg` (installed; use it on the saved file)

Use `ffmpeg` for trimming and exporting (below). It can also record through `avfoundation`, but listing the devices opens a macOS
permission prompt, so that part is yours to run, with your yes:

```sh
# TAKES OVER THE MACHINE: lists screen and microphone devices (may trigger permission prompts)
ffmpeg -hide_banner -f avfoundation -list_devices true -i ""
```

Then, using the index it prints for the screen and the microphone (replace `1` and `0`):
`ffmpeg -f avfoundation -framerate 30 -capture_cursor 1 -i "1:0" -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -c:a aac ~/Movies/glide-demo/raw-take1.mp4`
(Stop with `q`.) Prefer Method 1 or 2 unless you need a fixed frame rate.

## Trim and export

Read the raw file first (no approval needed, it reads a file): `ffprobe -hide_banner ~/Movies/glide-demo/raw-take1.mov`.

Trim and export to a 1080p, 30 fps H.264 file with AAC audio, dropping all metadata (a recording's metadata can carry the machine
name and time):

```sh
ffmpeg -hide_banner -ss 00:00:06 -to 00:06:10 -i ~/Movies/glide-demo/raw-take1.mov \
  -vf "scale=1920:-2,fps=30" -c:v libx264 -crf 20 -preset medium -pix_fmt yuv420p \
  -c:a aac -b:a 160k -map_metadata -1 -movflags +faststart ~/Movies/glide-demo/glide-demo-v1.mp4
```

`-ss` and `-to` are the cut (put them before `-i` for speed; the cut is accurate because the file is re-encoded). Change the two
times to the start of scene a and the end of scene g. To cut a bad section out of the middle, export two ranges and join them with
the concat demuxer, or use QuickTime Player's Edit > Trim for a single in-and-out. To blank a leaked item, re-record; do not rely on a blur.

## Where the file goes

`~/Movies/glide-demo/` (or any folder **outside** the repository and outside any synced or shared folder until you have checked it).
The repository's `.gitignore` ignores `runs/` and `.env` but **not** `.mov` or `.mp4`: never `git add` a recording, and do not put
one under `/Users/cuberg/glide`. Keep the raw takes private; delete them once the export is checked.

## Privacy check before sharing (do all of it, on the exported file)

1. **Watch the whole export, once, at normal speed, with sound.** Not a skim.
2. **Contact sheet** for a second look, one frame every 5 seconds:
   `ffmpeg -hide_banner -i ~/Movies/glide-demo/glide-demo-v1.mp4 -vf "fps=1/5,scale=480:-1,tile=4x6" -frames:v 1 ~/Movies/glide-demo/sheet.png`,
   then open `sheet.png` and look at every tile.
3. **Keys and secrets.** No environment variable value, no `.env` content, no `export ...=` line, no `printenv`, no token in a URL,
   no `Authorization` header. `$G doctor` shows variable names only; make sure nothing else did. The scene-a wrong key is
   `invalid-demo-key`, which is fine; a real key anywhere is not. If a real key appeared: rotate it first, then discard the file.
4. **Emails and names.** Your email, your full name, your home folder name in a path, hostnames, the Git author, a signed-in account
   in a menu bar or browser, a Wi-Fi network name.
5. **Tabs and windows.** Browser tabs other than the demo's, bookmarks, history suggestions in the address bar, other apps behind
   the terminal, the Dock, the desktop, Finder sidebars.
6. **Notifications.** Any banner, badge count or Focus status that slipped through; the menu bar clock and icons.
7. **Audio.** Nothing said that you do not want public, no other voices, no sound from another app or a call.
8. **Run folders on screen.** `run.json` shown in scene c and d must be the content-free one (no goal, no page text). If a folder
   from a `--record-content` run is on screen, discard the take.
9. **Metadata.** `ffprobe -hide_banner ~/Movies/glide-demo/glide-demo-v1.mp4` shows no machine name, location or software tags you do
   not want (the export above removed them).
10. **Say the limits are in it.** The video must contain the "never run live" statements of DEMO.md; a cut that removes them is a
    different, misleading video.

Only then share it. Afterwards: delete the raw takes, `rm -rf ~/glide-demo-profile ~/glide-demo/runs`, stop the web server and
the demo browser, remove `/tmp/glide-demo.sock`, and note the result of the real run next to each scene in
[LIVE_CHECKS.md](LIVE_CHECKS.md) terms (a ticked box, or the step that failed and the commit from `git rev-parse --short HEAD`).
