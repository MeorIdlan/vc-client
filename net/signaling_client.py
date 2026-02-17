"""WebSocket signaling client.

This is intentionally unaware of aiortc. It only speaks the JSON protocol
implemented by `server/app.py`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

import websockets

from . import protocol


logger = logging.getLogger(__name__)


AsyncCallback = Callable[..., Awaitable[None]]


@dataclass
class SignalingCallbacks:
	on_log: Optional[AsyncCallback] = None
	on_welcome: Optional[AsyncCallback] = None  # (peer_id: str)
	on_joined: Optional[AsyncCallback] = None  # (room: str, peers: list[dict])
	on_left: Optional[AsyncCallback] = None  # (room: str)
	on_peer_joined: Optional[AsyncCallback] = None  # (peer: dict)
	on_peer_left: Optional[AsyncCallback] = None  # (peer_id: str, reason: str)
	on_offer: Optional[AsyncCallback] = None  # (from_peer: str, sdp: str)
	on_answer: Optional[AsyncCallback] = None  # (from_peer: str, sdp: str)
	on_ice: Optional[AsyncCallback] = None  # (from_peer: str, candidate: dict)
	on_error: Optional[AsyncCallback] = None  # (error: str, payload: dict)


class SignalingClient:
	def __init__(self, url: str, callbacks: Optional[SignalingCallbacks] = None):
		self.url = url
		self.callbacks = callbacks or SignalingCallbacks()

		self.peer_id: Optional[str] = None
		# websockets' protocol types moved between versions; keep runtime-safe.
		self._ws: Optional[Any] = None
		self._recv_task: Optional[asyncio.Task[None]] = None
		self._send_lock = asyncio.Lock()
		self._connected_evt = asyncio.Event()

	@property
	def is_connected(self) -> bool:
		return self._ws is not None and not self._ws.closed

	async def connect(self) -> None:
		if self._recv_task and not self._recv_task.done():
			return

		await self._log(f"Connecting to {self.url}")
		logger.info("signaling connect url=%s", self.url)
		try:
			self._ws = await websockets.connect(self.url)
		except Exception:
			logger.exception("signaling connect failed url=%s", self.url)
			await self._emit_error("connect-failed", {"url": self.url})
			return
		self._connected_evt.set()
		self._recv_task = asyncio.create_task(self._recv_loop(), name="signaling-recv")

	async def disconnect(self) -> None:
		await self._log("Disconnecting")
		logger.info("signaling disconnect")
		self._connected_evt.clear()
		if self._recv_task:
			self._recv_task.cancel()
			try:
				await self._recv_task
			except Exception:
				pass
			self._recv_task = None

		if self._ws:
			try:
				await self._ws.close()
			except Exception:
				pass
		self._ws = None
		self.peer_id = None

	async def join(self, room: str, name: str) -> None:
		await self._send(protocol.make_join(room, name))

	async def leave(self) -> None:
		await self._send(protocol.make_leave())

	async def send_offer(self, to_peer: str, sdp: str) -> None:
		await self._send(protocol.make_offer(to_peer, sdp))

	async def send_answer(self, to_peer: str, sdp: str) -> None:
		await self._send(protocol.make_answer(to_peer, sdp))

	async def send_ice(self, to_peer: str, candidate: protocol.IceCandidateDict) -> None:
		await self._send(protocol.make_ice(to_peer, candidate))


	async def _send(self, payload: Dict[str, Any]) -> None:
		await self._connected_evt.wait()
		if not self._ws:
			raise RuntimeError("Signaling not connected")
		mtype = payload.get("type")
		to_peer = payload.get("to")
		if mtype in (protocol.OFFER, protocol.ANSWER):
			logger.info("signaling send type=%s to=%s sdp_len=%s", mtype, to_peer, len(str(payload.get("sdp", ""))))
		elif mtype == protocol.ICE:
			logger.debug("signaling send type=ice to=%s", to_peer)
		else:
			logger.debug("signaling send type=%s", mtype)
		raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
		async with self._send_lock:
			await self._ws.send(raw)

	async def _recv_loop(self) -> None:
		assert self._ws is not None
		ws = self._ws
		logger.debug("signaling recv loop started")

		try:
			async for raw in ws:
				try:
					msg = json.loads(raw)
				except json.JSONDecodeError:
					await self._emit_error("invalid-json", {"raw": raw})
					continue

				if not isinstance(msg, dict):
					await self._emit_error("invalid-message", {"msg": msg})
					continue

				mtype = msg.get("type")
				if not isinstance(mtype, str):
					await self._emit_error("missing-type", msg)
					continue

				if mtype == protocol.PING:
					await self._send(protocol.make_pong(msg.get("ts")))
					continue

				if mtype == protocol.WELCOME:
					self.peer_id = str(msg.get("peer_id", "")) or None
					logger.info("signaling welcome peer_id=%s", self.peer_id)
					if self.peer_id and self.callbacks.on_welcome:
						await self.callbacks.on_welcome(self.peer_id)
					continue

				if mtype == protocol.JOINED:
					room = str(msg.get("room", ""))
					peers = msg.get("peers", [])
					logger.info("signaling joined room=%s peers=%s", room, (len(peers) if isinstance(peers, list) else "?"))
					if self.callbacks.on_joined:
						await self.callbacks.on_joined(room, peers)
					continue

				if mtype == protocol.LEFT:
					room = str(msg.get("room", ""))
					logger.info("signaling left room=%s", room)
					if self.callbacks.on_left:
						await self.callbacks.on_left(room)
					continue

				if mtype == protocol.PEER_JOINED:
					peer = msg.get("peer", {})
					logger.info("signaling peer-joined peer_id=%s", (peer.get("peer_id") if isinstance(peer, dict) else None))
					if self.callbacks.on_peer_joined:
						await self.callbacks.on_peer_joined(peer)
					continue

				if mtype == protocol.PEER_LEFT:
					peer_id = str(msg.get("peer_id", ""))
					reason = str(msg.get("reason", ""))
					logger.info("signaling peer-left peer_id=%s reason=%s", peer_id, reason)
					if self.callbacks.on_peer_left:
						await self.callbacks.on_peer_left(peer_id, reason)
					continue

				if mtype == protocol.OFFER:
					from_peer = str(msg.get("from", ""))
					sdp = str(msg.get("sdp", ""))
					logger.info("signaling offer from=%s sdp_len=%s", from_peer, len(sdp))
					if self.callbacks.on_offer:
						await self.callbacks.on_offer(from_peer, sdp)
					continue

				if mtype == protocol.ANSWER:
					from_peer = str(msg.get("from", ""))
					sdp = str(msg.get("sdp", ""))
					logger.info("signaling answer from=%s sdp_len=%s", from_peer, len(sdp))
					if self.callbacks.on_answer:
						await self.callbacks.on_answer(from_peer, sdp)
					continue

				if mtype == protocol.ICE:
					from_peer = str(msg.get("from", ""))
					candidate = msg.get("candidate")
					logger.debug("signaling ice from=%s has_candidate=%s", from_peer, bool(candidate))
					if self.callbacks.on_ice:
						await self.callbacks.on_ice(from_peer, candidate)
					continue

				if mtype == protocol.ERROR:
					await self._emit_error(str(msg.get("error", "error")), msg)
					continue

				await self._emit_error("unknown-type", msg)

		except asyncio.CancelledError:
			pass
		except Exception as e:
			logger.exception("signaling recv loop crashed")
			await self._emit_error(f"recv-loop-exception: {e}", {})
		finally:
			self._connected_evt.clear()
			logger.debug("signaling recv loop stopped")
			try:
				await ws.close()
			except Exception:
				pass
			if self._ws is ws:
				self._ws = None

	async def _emit_error(self, error: str, payload: Dict[str, Any]) -> None:
		await self._log(f"Signaling error: {error}")
		if self.callbacks.on_error:
			await self.callbacks.on_error(error, payload)

	async def _log(self, message: str) -> None:
		if self.callbacks.on_log:
			await self.callbacks.on_log(message)
