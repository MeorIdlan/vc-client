"""One WebRTC connection to one peer (audio-only mesh)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from aiortc import (
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.rtcconfiguration import RTCConfiguration
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp

from .audio import AudioActivityConfig, AudioDevice, RemoteAudioSink, wrap_with_audio_activity_log


AsyncPeerCallback = Callable[..., Awaitable[None]]


def _candidate_to_json(candidate: RTCIceCandidate) -> Dict[str, Any]:
    return {
        "candidate": candidate_to_sdp(candidate),
        "sdpMid": getattr(candidate, "sdpMid", None),
        "sdpMLineIndex": getattr(candidate, "sdpMLineIndex", None),
    }


def _candidate_from_json(obj: Dict[str, Any]):
    cand_sdp = obj.get("candidate")
    if not isinstance(cand_sdp, str) or not cand_sdp:
        raise ValueError("missing candidate")
    cand = candidate_from_sdp(cand_sdp)
    cand.sdpMid = obj.get("sdpMid")
    cand.sdpMLineIndex = obj.get("sdpMLineIndex")
    return cand


@dataclass
class PeerCallbacks:
    on_log: Optional[AsyncPeerCallback] = None  # (msg: str)
    on_connection_state: Optional[AsyncPeerCallback] = None  # (peer_id: str, state: str)
    on_local_ice: Optional[AsyncPeerCallback] = None  # (peer_id: str, candidate: dict)


class WebRTCPeer:
    def __init__(
        self,
        peer_id: str,
        local_audio_track,
        preferred_output: Optional[AudioDevice] = None,
        audio_activity_config: Optional[AudioActivityConfig] = None,
        callbacks: Optional[PeerCallbacks] = None,
        rtc_config: Optional[RTCConfiguration] = None,
    ):
        self.peer_id = peer_id
        self._callbacks = callbacks or PeerCallbacks()
        self._pc = RTCPeerConnection(configuration=rtc_config)

        self._remote_sink: Optional[RemoteAudioSink] = None
        self._preferred_output = preferred_output
        self._audio_activity_config = audio_activity_config
        self._closed = False

        if local_audio_track is not None:
            # audio-only: send microphone to peer
            self._pc.addTrack(local_audio_track)

        @self._pc.on("icecandidate")
        async def on_icecandidate(event) -> None:
            if event is None or event.candidate is None:
                return
            if self._callbacks.on_local_ice:
                await self._callbacks.on_local_ice(self.peer_id, _candidate_to_json(event.candidate))

        @self._pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            state = self._pc.connectionState
            await self._log(f"pc[{self.peer_id}] connectionState={state}")
            if self._callbacks.on_connection_state:
                await self._callbacks.on_connection_state(self.peer_id, state)

        @self._pc.on("track")
        async def on_track(track) -> None:
            await self._log(f"pc[{self.peer_id}] remote track kind={track.kind}")
            if track.kind == "audio":
                self._remote_sink = RemoteAudioSink(output=self._preferred_output)

                wrapped = wrap_with_audio_activity_log(
                    track,
                    label=f"RX peer={self.peer_id}",
                    config=self._audio_activity_config,
                )
                await self._remote_sink.start(wrapped or track)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._remote_sink:
                await self._remote_sink.stop()
        finally:
            await self._pc.close()

    async def create_offer(self) -> str:
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        assert self._pc.localDescription is not None
        return self._pc.localDescription.sdp

    async def apply_answer(self, sdp: str) -> None:
        await self._pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="answer"))

    async def apply_offer_and_create_answer(self, sdp: str) -> str:
        await self._pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)
        assert self._pc.localDescription is not None
        return self._pc.localDescription.sdp

    async def add_ice_candidate(self, candidate_obj: Any) -> None:
        if not candidate_obj:
            return
        if not isinstance(candidate_obj, dict):
            return
        try:
            cand = _candidate_from_json(candidate_obj)
        except Exception:
            return
        # aiortc expects addIceCandidate to accept None (end-of-candidates) too
        await self._pc.addIceCandidate(cand)

    async def _log(self, msg: str) -> None:
        if self._callbacks.on_log:
            await self._callbacks.on_log(msg)
