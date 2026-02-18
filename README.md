# vc-client

This repo contains the **desktop client** for `vc-client`.

- UI: PySide6 (Qt)
- Signaling: WebSocket JSON protocol (to the server)
- Media: WebRTC audio (peer-to-peer mesh) via `aiortc`

The client connects to the signaling server, joins a room, and then establishes a direct WebRTC connection to each peer in that room.

## What it does

- Connects to a signaling server WebSocket (default: `ws://127.0.0.1:8765/ws`)
- Joins a room (`default` by default)
- Negotiates WebRTC with other peers via signaling messages (`offer`/`answer`/`ice`)
- Captures microphone audio and sends it to peers
- Plays remote peers’ audio (best-effort)

Implementation highlights:
- GUI controller + asyncio thread bridge: [ui/app.py](ui/app.py)
- Signaling client + protocol: [net/signaling_client.py](net/signaling_client.py), [net/protocol.py](net/protocol.py)
- WebRTC mesh logic: [rtc/peer_manager.py](rtc/peer_manager.py)
- Per-peer RTCPeerConnection: [rtc/webrtc_peer.py](rtc/webrtc_peer.py)
- Audio capture / playback helpers: [rtc/audio.py](rtc/audio.py)

## Requirements

- Python (the GitHub Actions build uses Python 3.13)
- A working audio stack
  - Linux: PulseAudio/PipeWire (Pulse compat) or ALSA (best-effort)
  - Windows: uses `sounddevice` (PortAudio) when available

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Command-line flags are defined in [main.py](main.py).

Environment variables (used as defaults):
- `VC_SERVER_URL` (default `ws://127.0.0.1:8765/ws`)
- `VC_ROOM` (default `default`)
- `VC_NAME` (default `$USER`)
- Logging:
  - `VC_CLIENT_LOG_LEVEL` or `VC_LOG_LEVEL`

## Notes / limitations

- **Mesh topology:** each client creates one RTCPeerConnection per remote peer. Great for small rooms; not for large groups.
- **NAT traversal:** the current UI path doesn’t configure STUN/TURN servers. It will work best on LAN / easy NAT.
- **Audio devices:** changing device selection does not renegotiate existing peer connections; it affects new peers only.

## Downloading a Windows .exe (GitHub Releases)

If you don’t want to run from source, you can download a prebuilt Windows executable from the project’s GitHub Releases.

1. Open the repository on GitHub and go to **Releases**.
2. Open the latest release.
3. Under **Assets**, download `vc-client-windows.zip`.
4. Extract the zip and run `vc-client.exe`.

Notes:
- Windows may show SmartScreen for unsigned executables.
- The client needs a reachable signaling server URL (the `vc-server` WebSocket endpoint).

## Building the Windows .exe yourself

If you want to produce the executable locally (Windows), you can build it with PyInstaller.

Because the code uses package-relative imports (e.g. `from .ui.app import ...`), run PyInstaller from a directory where the `client` package is importable.

One simple approach on Windows is to open PowerShell in the **parent directory** that contains the repo folder (checked out as `client`).

```powershell
python -m pip install --upgrade pip
python -m pip install -r client\requirements.txt

@'
from client.main import main
raise SystemExit(main())
'@ | Set-Content -Encoding UTF8 build_entry.py

python -m PyInstaller --noconfirm --clean --onefile --windowed --name vc-client build_entry.py
```

Output:
- The executable will be at `dist\vc-client.exe`.
