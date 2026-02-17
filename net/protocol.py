"""Signaling protocol helpers.

The signaling server expects JSON objects on a single WebSocket.
See `server/app.py` for authoritative behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, TypedDict


# Message type constants
WELCOME = "welcome"
JOIN = "join"
JOINED = "joined"
LEAVE = "leave"
LEFT = "left"
PEER_JOINED = "peer-joined"
PEER_LEFT = "peer-left"

OFFER = "offer"
ANSWER = "answer"
ICE = "ice"

PING = "ping"
PONG = "pong"
ERROR = "error"


class PeerInfo(TypedDict, total=False):
	peer_id: str
	name: str


class IceCandidateDict(TypedDict, total=False):
	candidate: str
	sdpMid: Optional[str]
	sdpMLineIndex: Optional[int]


def make_join(room: str, name: str) -> Dict[str, Any]:
	return {"type": JOIN, "room": room, "name": name}


def make_leave() -> Dict[str, Any]:
	return {"type": LEAVE}


def make_pong(ts: Optional[int] = None) -> Dict[str, Any]:
	msg: Dict[str, Any] = {"type": PONG}
	if ts is not None:
		msg["ts"] = ts
	return msg


def make_offer(to_peer: str, sdp: str) -> Dict[str, Any]:
	return {"type": OFFER, "to": to_peer, "sdp": sdp}


def make_answer(to_peer: str, sdp: str) -> Dict[str, Any]:
	return {"type": ANSWER, "to": to_peer, "sdp": sdp}


def make_ice(to_peer: str, candidate: IceCandidateDict) -> Dict[str, Any]:
	return {"type": ICE, "to": to_peer, "candidate": candidate}


@dataclass(frozen=True)
class ProtocolError(Exception):
	message: str
