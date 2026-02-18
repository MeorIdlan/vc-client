"""Peer manager (audio-only mesh)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, cast

from aiortc.rtcconfiguration import RTCConfiguration

from ..net.signaling_client import SignalingClient
from ..net.protocol import IceCandidateDict
from .audio import AudioActivityConfig, AudioDevice, LocalAudio
from .webrtc_peer import PeerCallbacks, WebRTCPeer


logger = logging.getLogger(__name__)


AsyncCallback = Callable[..., Awaitable[None]]


@dataclass
class ManagerCallbacks:
    on_log: Optional[AsyncCallback] = None
    on_peer_state: Optional[AsyncCallback] = None  # (peer_id: str, state: str)


class PeerManager:
    def __init__(self, callbacks: Optional[ManagerCallbacks] = None):
        self._callbacks = callbacks or ManagerCallbacks()
        self._signaling: Optional[SignalingClient] = None
        self._self_id: Optional[str] = None
        self._room: Optional[str] = None

        self._peers: Dict[str, WebRTCPeer] = {}
        self._preferred_input: Optional[AudioDevice] = None
        self._preferred_output: Optional[AudioDevice] = None
        self._audio_activity_config = AudioActivityConfig.from_env()
        self._local_audio = LocalAudio.create(self._preferred_input, activity_config=self._audio_activity_config)
        self._rtc_config: Optional[RTCConfiguration] = None

        self._lock = asyncio.Lock()

    @property
    def audio_activity_config(self) -> AudioActivityConfig:
        return self._audio_activity_config

    async def set_audio_activity_threshold(self, rms_start: int) -> None:
        """Update the audio activity detection threshold.

        Uses hysteresis: stop is kept below start to avoid flicker.
        """

        start = max(0, int(rms_start))
        stop = int(start * 0.66)
        if stop >= start:
            stop = max(0, start - 1)

        async with self._lock:
            self._audio_activity_config.rms_start = start
            self._audio_activity_config.rms_stop = stop
        logger.info("rtc audio activity threshold updated start=%s stop=%s", start, stop)

    async def set_audio_devices(self, *, input_device: Optional[AudioDevice], output_device: Optional[AudioDevice]) -> None:
        """Set preferred audio devices for future tracks.

        Note: changing input device does not renegotiate existing peer
        connections; it will only apply to peers created after the change.
        """

        async with self._lock:
            self._preferred_input = input_device
            self._preferred_output = output_device

            old = self._local_audio
            self._local_audio = LocalAudio.create(self._preferred_input, activity_config=self._audio_activity_config)

        try:
            old.close()
        except Exception:
            pass

    def set_signaling(self, signaling: SignalingClient) -> None:
        self._signaling = signaling

    def set_self_id(self, peer_id: str) -> None:
        self._self_id = peer_id

    def set_rtc_configuration(self, rtc_config: RTCConfiguration) -> None:
        self._rtc_config = rtc_config

    async def join_room(self, room: str, peers_list: list[dict]) -> None:
        self._room = room
        logger.info("rtc join room=%s peers=%s", room, len(peers_list))
        for peer in peers_list:
            other_id = str(peer.get("peer_id", ""))
            if other_id:
                await self._ensure_peer(other_id)

        # Start offers where we are the offerer.
        for other_id in list(self._peers.keys()):
            await self._maybe_make_offer(other_id)

    async def handle_peer_joined(self, peer: dict) -> None:
        other_id = str(peer.get("peer_id", ""))
        if not other_id:
            return
        logger.info("rtc peer joined peer_id=%s", other_id)
        await self._ensure_peer(other_id)
        await self._maybe_make_offer(other_id)

    async def handle_peer_left(self, peer_id: str) -> None:
        logger.info("rtc peer left peer_id=%s", peer_id)
        await self._remove_peer(peer_id)

    async def handle_offer(self, from_peer: str, sdp: str) -> None:
        logger.info("rtc offer received from=%s sdp_len=%s", from_peer, len(sdp))
        p = await self._ensure_peer(from_peer)
        answer_sdp = await p.apply_offer_and_create_answer(sdp)
        if self._signaling:
            await self._signaling.send_answer(from_peer, answer_sdp)

    async def handle_answer(self, from_peer: str, sdp: str) -> None:
        p = self._peers.get(from_peer)
        if not p:
            return
        logger.info("rtc answer received from=%s sdp_len=%s", from_peer, len(sdp))
        await p.apply_answer(sdp)

    async def handle_ice(self, from_peer: str, candidate: Any) -> None:
        p = self._peers.get(from_peer)
        if not p:
            # If ICE arrives before we created the peer, create it.
            p = await self._ensure_peer(from_peer)
        logger.debug("rtc ice received from=%s has_candidate=%s", from_peer, bool(candidate))
        await p.add_ice_candidate(candidate)

    async def leave_room(self) -> None:
        self._room = None
        logger.info("rtc leave room")
        # Close all peer connections
        for pid in list(self._peers.keys()):
            await self._remove_peer(pid)

    async def _ensure_peer(self, peer_id: str) -> WebRTCPeer:
        async with self._lock:
            existing = self._peers.get(peer_id)
            if existing:
                return existing

            cb = PeerCallbacks(
                on_log=self._log,
                on_connection_state=self._on_peer_state,
                on_local_ice=self._on_local_ice,
            )
            peer = WebRTCPeer(
                peer_id=peer_id,
                local_audio_track=self._local_audio.track,
                preferred_output=self._preferred_output,
                audio_activity_config=self._audio_activity_config,
                callbacks=cb,
                rtc_config=self._rtc_config,
            )
            self._peers[peer_id] = peer
            await self._log(f"Created peer pc for {peer_id}")
            logger.debug("rtc created peer pc peer_id=%s", peer_id)
            return peer

    async def _remove_peer(self, peer_id: str) -> None:
        async with self._lock:
            peer = self._peers.pop(peer_id, None)
        if peer:
            await self._log(f"Closing peer pc for {peer_id}")
            logger.debug("rtc closing peer pc peer_id=%s", peer_id)
            await peer.close()

    async def _maybe_make_offer(self, other_id: str) -> None:
        if not self._signaling or not self._self_id:
            return

        # Offerer rule: lexicographically smaller peer_id offers.
        if self._self_id >= other_id:
            logger.debug("rtc offer skipped (not offerer) self=%s other=%s", self._self_id, other_id)
            return

        peer = self._peers.get(other_id)
        if not peer:
            return

        await self._log(f"Creating offer to {other_id}")
        logger.info("rtc creating offer to=%s", other_id)
        sdp = await peer.create_offer()
        await self._signaling.send_offer(other_id, sdp)

    async def _on_local_ice(self, peer_id: str, candidate: dict) -> None:
        if self._signaling:
            logger.debug("rtc local ice peer_id=%s", peer_id)
            await self._signaling.send_ice(peer_id, cast(IceCandidateDict, candidate))

    async def _on_peer_state(self, peer_id: str, state: str) -> None:
        if self._callbacks.on_peer_state:
            await self._callbacks.on_peer_state(peer_id, state)

    async def _log(self, message: str) -> None:
        if self._callbacks.on_log:
            await self._callbacks.on_log(message)
