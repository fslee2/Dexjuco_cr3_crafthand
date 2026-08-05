# Quest 3 WebXR Teleop MVP

**Date**: 2026-07-09
**Status**: First-pass MVP - input bridge working, no visual feedback yet.

---

## 1. Why Quest 3 Replaces Webcam/HaMeR/MediaPipe

| Old input source | Limitation |
|---|---|
| Single webcam + HaMeR/MediaPipe | Requires camera calibration, visual inference, PC GPU, line-of-sight |
| Vive tracker + UDP | Requires SteamVR Base Stations, Vive tracker hardware |

The Quest 3 provides:
- 6-DoF controller/hand pose at sub-mm accuracy via inside-out tracking
- Trigger, grip, and (optional) pinch values via gamepad/hand-tracking APIs
- A browser-based WebXR interface - no native app/build/APK needed for MVP

This opens teleop to any WiFi-connected Quest 3 without GPU inference,
camera setup, or external tracking infrastructure.

---

## 2. MVP Architecture

```text
Quest Browser WebXR page
  -> WebSocket JSON: ws://host:8765 or wss://host:8765
  -> Python QuestWebSocketReceiver thread
  -> QuestTeleopState
  -> QuestActionMapper.map(state, ee_matrix)
  -> 22D action [xyz + quat + craft15]
  -> Cr3CraftClickMouseShellEnv
```

---

## 3. Files

| Path | Purpose |
|---|---|
| `dexjoco/dexjoco/tasks/quest_teleop.py` | Core module: state, config, mapper, receiver, synthetic source |
| `teleop/webxr_quest/index.html` | Quest Browser WebXR client (static HTML/JS) |
| `teleop/webxr_quest/README.md` | Client usage notes |
| `scripts/quest_cr3_craft_click_mouse_shell.py` | Entrypoint script |
| `scripts/serve_quest_webxr.py` | HTTPS static page server for Quest Browser |
| `tests/test_quest_teleop.py` | Smoke tests (no Quest required) |
| `docs/quest3_teleop_mvp.md` | This document |

---

## 4. What Works Now

- QuestTeleopState dataclass with xyzw/wxyz quaternion conventions clearly labeled
- Thread-safe QuestStateStore for sharing state between network thread and env loop
- QuestWebSocketReceiver: background asyncio WebSocket server (requires `websockets` lib)
- QuestActionMapper: Quest pose delta to 22D CR3+CRAFT action
  - Position delta with configurable scale, smoothing, and per-step clamping
  - Rotation delta via scipy.spatial.transform.Rotation
  - craft15 interpolation from open to close based on max(trigger, grip, pinch)
- SyntheticQuestSource: circle-motion test data with oscillating hand close
- SingleArmQuestTeleopWrapper: Gym-style wrapper that can be used as drop-in
- Entrypoint script with --synthetic, --viewer, --host, --port flags
- Smoke tests that verify action shapes, calibration, movement, craft response

---

## 5. What Does NOT Work Yet

| Gap | Notes |
|---|---|
| Stereo visual feedback | No OpenTeleVision-style stereo rendering or WebRTC streaming |
| Quest raw passthrough camera | Not accessible from browser WebXR |
| Polished hand retargeting | Basic open/close interpolation only; no per-finger mapping from hand joints |
| Automatic HTTPS for Quest Browser | Requires manual cert setup or reverse proxy |
| Bimanual (dual-controller) teleop | MVP is right-hand single-arm only |
| Full OpenTeleVision clone | No ngrok automation, no WebRTC, no Unity APK |

These are explicitly out of scope for this first pass.

---

## 6. How to Run Synthetic Smoke Test

```bash
cd <DEXJOCO_ROOT>
export PYTHONPATH=<DEXJOCO_ROOT>/dexjoco
export MUJOCO_GL=egl

# Compile-check all new Python files
python -m py_compile dexjoco/dexjoco/tasks/quest_teleop.py
python -m py_compile scripts/quest_cr3_craft_click_mouse_shell.py

# Run synthetic smoke test (no Quest, no network)
python scripts/quest_cr3_craft_click_mouse_shell.py --synthetic --steps 20

# Run pytest suite
python -m pytest tests/test_quest_teleop.py -v
```

---

## 7. How to Try With Real Quest Browser

### Prerequisites
- Quest 3 on same WiFi as workstation
- Python environment with `websockets` installed: `pip install websockets`
- TLS certificate for HTTPS (Quest Browser requires HTTPS for immersive WebXR)

Generate a certificate once (use mkcert or openssl):
```bash
# Option A: mkcert (recommended, installs local CA on Quest via adb)
mkcert -install
mkcert -cert-file cert.pem -key-file key.pem localhost 192.168.1.x

# Option B: self-signed (Quest will show a warning to accept)
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem \
    -days 365 -nodes -subj "/CN=localhost"
```

### Two-Terminal Flow

**Terminal A** -- start the WSS pose receiver with viewer:
```bash
python scripts/quest_cr3_craft_click_mouse_shell.py --viewer \
    --certfile cert.pem --keyfile key.pem
```

**Terminal B** -- serve the WebXR page over HTTPS:
```bash
python scripts/serve_quest_webxr.py \
    --certfile cert.pem --keyfile key.pem \
    --ws-url wss://<workstation-ip>:8765
```

The script prints the exact Quest Browser URL:
```
https://192.168.1.10:8443/?ws=wss://192.168.1.10:8765
```

### On Quest Browser
1. Navigate to the URL printed by `serve_quest_webxr.py`.
2. Accept the certificate warning if using a self-signed cert.
3. Click "Connect WS", then "Start XR".
4. Grant immersive-vr permission when prompted.

### Fallback: Synthetic (First Validation)
If you don't have TLS set up yet, verify correctness first with:
```bash
python scripts/quest_cr3_craft_click_mouse_shell.py --synthetic --steps 20
python -m pytest tests/test_quest_teleop.py -v
```

---

## 8. Recommended Next Steps

1. **Real Quest end-to-end test** - Verify WebSocket latency, pose tracking quality, and hand close response with actual Quest hardware.
2. **Stereo visual feedback** - Implement two-camera rendering and WebRTC streaming (OpenTeleVision-style) so the operator can see the robot workspace.
3. **Per-finger hand mapping** - When Quest hand tracking is enabled, map individual finger joints through the WebXR hand API into per-joint craft15 commands.
4. **Bimanual support** - Extend to dual-controller for two-arm tasks.
5. **Certificate setup helper** - Optionally add a helper around mkcert/openssl. The current server only consumes user-provided cert/key files.
6. **Calibration workflow** - Add a UI/CLI calibration step to align Quest reference frame with robot workspace.
