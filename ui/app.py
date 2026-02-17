from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

from concurrent.futures import Future

from PySide6 import QtCore, QtWidgets

from ..net.signaling_client import SignalingCallbacks, SignalingClient
from ..rtc.audio import AudioDevice, list_audio_inputs, list_audio_outputs
from ..rtc.peer_manager import ManagerCallbacks, PeerManager
from .windows import MainWindow


logger = logging.getLogger(__name__)


class AsyncioThread:
    """Runs an asyncio loop in a background thread and schedules coroutines."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if not self._loop:
            raise RuntimeError("AsyncioThread not started")
        return self._loop

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=_run, name="asyncio-thread", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    def stop(self) -> None:
        if not self._loop:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)

    def submit(self, coro) -> Future:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)


class UiBridge(QtCore.QObject):
    log = QtCore.Signal(str)
    status = QtCore.Signal(str)
    peers_changed = QtCore.Signal(list)  # list[str]


@dataclass
class AppConfig:
    server_url: str
    room: str
    name: str


class VCClientApp(QtCore.QObject):
    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg

        self.window = MainWindow()
        self.bridge = UiBridge()
        self.asyncio_thread = AsyncioThread()

        self._roster: Dict[str, str] = {}  # peer_id -> display string

        self._selected_input: Optional[AudioDevice] = None
        self._selected_output: Optional[AudioDevice] = None

        self.peer_manager = PeerManager(
            callbacks=ManagerCallbacks(
                on_log=self._on_async_log,
                on_peer_state=self._on_peer_state,
            )
        )

        self.signaling = SignalingClient(
            url=self.cfg.server_url,
            callbacks=SignalingCallbacks(
                on_log=self._on_async_log,
                on_welcome=self._on_welcome,
                on_joined=self._on_joined,
                on_left=self._on_left,
                on_peer_joined=self._on_peer_joined,
                on_peer_left=self._on_peer_left,
                on_offer=self._on_offer,
                on_answer=self._on_answer,
                on_ice=self._on_ice,
                on_error=self._on_error,
            ),
        )
        self.peer_manager.set_signaling(self.signaling)

        self._wire_ui()
        self._wire_bridge()
        self._init_audio_device_selectors()

        # Defaults
        self.window.server_url_edit.setText(cfg.server_url)
        self.window.room_edit.setText(cfg.room)
        self.window.name_edit.setText(cfg.name)

    def start(self) -> None:
        self.asyncio_thread.start()
        self._apply_audio_device_selection()
        self.window.show()
        self.bridge.status.emit("Ready")
        logger.info("ui started")

    def shutdown(self) -> None:
        # Best-effort cleanup
        logger.info("ui shutdown")
        try:
            self.asyncio_thread.submit(self.signaling.disconnect())
        except Exception:
            pass
        self.asyncio_thread.stop()

    def _wire_ui(self) -> None:
        self.window.connect_clicked.connect(self._on_connect_clicked)
        self.window.disconnect_clicked.connect(self._on_disconnect_clicked)
        self.window.join_clicked.connect(self._on_join_clicked)
        self.window.leave_clicked.connect(self._on_leave_clicked)

        self.window.mic_combo.currentIndexChanged.connect(self._on_audio_device_changed)
        self.window.speaker_combo.currentIndexChanged.connect(self._on_audio_device_changed)

    def _init_audio_device_selectors(self) -> None:
        """Populate device dropdowns (best-effort)."""

        inputs = list_audio_inputs()
        outputs = list_audio_outputs()

        self.window.mic_combo.blockSignals(True)
        self.window.speaker_combo.blockSignals(True)
        try:
            self.window.mic_combo.clear()
            for d in inputs:
                self.window.mic_combo.addItem(d.label, userData=d)

            self.window.speaker_combo.clear()
            for d in outputs:
                self.window.speaker_combo.addItem(d.label, userData=d)

            # Default to first entry (usually "System default").
            if self.window.mic_combo.count() > 0:
                self.window.mic_combo.setCurrentIndex(0)
            if self.window.speaker_combo.count() > 0:
                self.window.speaker_combo.setCurrentIndex(0)
        finally:
            self.window.mic_combo.blockSignals(False)
            self.window.speaker_combo.blockSignals(False)

        # Cache current values for applying once asyncio thread starts.
        self._selected_input, self._selected_output = self._get_selected_devices()

    def _get_selected_devices(self) -> tuple[Optional[AudioDevice], Optional[AudioDevice]]:
        inp = self.window.mic_combo.currentData()
        out = self.window.speaker_combo.currentData()
        return (
            inp if isinstance(inp, AudioDevice) else None,
            out if isinstance(out, AudioDevice) else None,
        )

    def _apply_audio_device_selection(self) -> None:
        if not getattr(self.asyncio_thread, "_loop", None):
            return
        self.asyncio_thread.submit(
            self.peer_manager.set_audio_devices(input_device=self._selected_input, output_device=self._selected_output)
        )

    @QtCore.Slot(int)
    def _on_audio_device_changed(self, _index: int) -> None:
        self._selected_input, self._selected_output = self._get_selected_devices()
        self._apply_audio_device_selection()

    def _wire_bridge(self) -> None:
        self.bridge.log.connect(self.window.log_panel.append_log)
        self.bridge.status.connect(self.window.set_status)
        self.bridge.peers_changed.connect(self._update_peer_list)

    @QtCore.Slot()
    def _on_connect_clicked(self) -> None:
        self.signaling.url = self.window.server_url_edit.text().strip()
        self.bridge.status.emit("Connecting...")
        logger.info("ui connect clicked url=%s", self.signaling.url)
        self.asyncio_thread.submit(self.signaling.connect())

    @QtCore.Slot()
    def _on_disconnect_clicked(self) -> None:
        self.bridge.status.emit("Disconnecting...")
        logger.info("ui disconnect clicked")
        self.asyncio_thread.submit(self.peer_manager.leave_room())
        self.asyncio_thread.submit(self.signaling.disconnect())

    @QtCore.Slot()
    def _on_join_clicked(self) -> None:
        room = self.window.room_edit.text().strip()
        name = self.window.name_edit.text().strip()
        if not room:
            self.bridge.log.emit("Room is required")
            return
        self.bridge.status.emit(f"Joining {room}...")
        logger.info("ui join clicked room=%s name_set=%s", room, bool(name.strip()))
        self.asyncio_thread.submit(self.signaling.join(room, name))

    @QtCore.Slot()
    def _on_leave_clicked(self) -> None:
        self.bridge.status.emit("Leaving...")
        logger.info("ui leave clicked")
        self.asyncio_thread.submit(self.peer_manager.leave_room())
        self.asyncio_thread.submit(self.signaling.leave())
        self._roster.clear()
        self.bridge.peers_changed.emit([])

    @QtCore.Slot(list)
    def _update_peer_list(self, peers: list) -> None:
        self.window.peer_list.set_peers([str(p) for p in peers])

    # ----------------------
    # Async callbacks (run in asyncio thread)
    # ----------------------
    async def _on_async_log(self, message: str) -> None:
        self.bridge.log.emit(message)

    async def _on_peer_state(self, peer_id: str, state: str) -> None:
        self.bridge.log.emit(f"Peer {peer_id} state: {state}")

    async def _on_welcome(self, peer_id: str) -> None:
        self.peer_manager.set_self_id(peer_id)
        self.bridge.status.emit(f"Connected as {peer_id}")

    async def _on_joined(self, room: str, peers: list[dict]) -> None:
        self.bridge.log.emit(f"Joined room {room}")
        self._roster = {str(p.get('peer_id')): self._fmt_peer(p) for p in peers if p.get('peer_id')}
        await self.peer_manager.join_room(room, peers)
        self.bridge.peers_changed.emit(list(self._roster.values()))

    async def _on_left(self, room: str) -> None:
        self.bridge.log.emit(f"Left room {room}")
        self._roster.clear()
        await self.peer_manager.leave_room()
        self.bridge.peers_changed.emit([])

    async def _on_peer_joined(self, peer: dict) -> None:
        pid = str(peer.get("peer_id", ""))
        if pid:
            self._roster[pid] = self._fmt_peer(peer)
            self.bridge.peers_changed.emit(list(self._roster.values()))
        await self.peer_manager.handle_peer_joined(peer)

    async def _on_peer_left(self, peer_id: str, reason: str) -> None:
        if peer_id in self._roster:
            self._roster.pop(peer_id, None)
            self.bridge.peers_changed.emit(list(self._roster.values()))
        await self.peer_manager.handle_peer_left(peer_id)
        self.bridge.log.emit(f"Peer left: {peer_id} ({reason})")

    async def _on_offer(self, from_peer: str, sdp: str) -> None:
        await self.peer_manager.handle_offer(from_peer, sdp)

    async def _on_answer(self, from_peer: str, sdp: str) -> None:
        await self.peer_manager.handle_answer(from_peer, sdp)

    async def _on_ice(self, from_peer: str, candidate: Any) -> None:
        await self.peer_manager.handle_ice(from_peer, candidate)

    async def _on_error(self, error: str, payload: dict) -> None:
        self.bridge.log.emit(f"Error: {error} {payload}")
        self.bridge.status.emit(f"Error: {error}")

    def _fmt_peer(self, peer: dict) -> str:
        pid = str(peer.get("peer_id", ""))
        name = str(peer.get("name", "")).strip()
        return f"{name} ({pid})" if name else pid


def create_qt_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app  # type: ignore
