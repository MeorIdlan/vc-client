"""Audio helpers for aiortc.

MVP scope:
- Create a local microphone audio track (best-effort on Windows).
- Provide a best-effort sink for remote audio (playback if possible, else discard).
"""

from __future__ import annotations

import asyncio
import array
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from queue import Queue
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, cast

import av
from aiortc import MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

try:
	import numpy as np  # type: ignore
except Exception:  # pragma: no cover
	np = None  # type: ignore

try:
	import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover
	sd = None  # type: ignore


logger = logging.getLogger(__name__)


def _env_truthy(name: str) -> bool:
	v = os.environ.get(name, "").strip().casefold()
	return v in {"1", "true", "yes", "on"}


def _audio_enum_log_enabled() -> bool:
	# Enable automatically when debug logging is on, or force via env var.
	return logger.isEnabledFor(logging.DEBUG) or _env_truthy("VC_AUDIO_DEVICE_ENUM_LOG")


def _audio_enum_log(msg: str, *args: object) -> None:
	if not _audio_enum_log_enabled():
		return
	# If explicitly enabled via env var, emit at INFO so it shows even when the
	# app isn't running with debug logs.
	if _env_truthy("VC_AUDIO_DEVICE_ENUM_LOG") and not logger.isEnabledFor(logging.DEBUG):
		logger.info(msg, *args)
	else:
		logger.debug(msg, *args)


def _audio_enum_dump_enabled() -> bool:
	# Separate switch because device dumps can be quite noisy.
	return _env_truthy("VC_AUDIO_DEVICE_ENUM_DUMP")


def _env_int(name: str, default: int) -> int:
	v = os.environ.get(name)
	if v is None:
		return default
	try:
		return int(v)
	except Exception:
		return default


def _env_float(name: str, default: float) -> float:
	v = os.environ.get(name)
	if v is None:
		return default
	try:
		return float(v)
	except Exception:
		return default


@dataclass
class AudioActivityConfig:
	"""Configuration for AudioActivityLogTrack.

	Values are on an int16-ish RMS scale (roughly 0..32768).
	"""

	rms_start: int = 900
	rms_stop: int = 600
	log_interval_sec: float = 1.0

	@classmethod
	def from_env(cls) -> "AudioActivityConfig":
		return cls(
			rms_start=int(_env_int("VC_AUDIO_RMS_START", cls.rms_start)),
			rms_stop=int(_env_int("VC_AUDIO_RMS_STOP", cls.rms_stop)),
			log_interval_sec=float(_env_float("VC_AUDIO_LOG_INTERVAL_SEC", cls.log_interval_sec)),
		)


class AudioActivityLogTrack(MediaStreamTrack):
	"""Pass-through audio track that emits debug logs when voice is detected.

	This is meant purely for local testing / debugging.

	Tuning (optional env vars):
	- VC_AUDIO_RMS_START: int16 RMS threshold to enter "talking" state.
	- VC_AUDIO_RMS_STOP: int16 RMS threshold to exit "talking" state.
	- VC_AUDIO_LOG_INTERVAL_SEC: throttle interval for repeated "talking" logs.
	"""

	kind = "audio"

	def __init__(
		self,
		source: MediaStreamTrack,
		*,
		label: str,
		config: Optional[AudioActivityConfig] = None,
		on_activity: Optional[Callable[[bool, int, str], None]] = None,
	):
		super().__init__()
		self._source = source
		self._label = label
		self._config = config or AudioActivityConfig.from_env()
		self._talking = False
		self._last_log_ts = 0.0
		self._on_activity = on_activity

	async def recv(self):  # type: ignore[override]
		frame = await self._source.recv()

		# Keep overhead near-zero unless we are logging or a caller wants activity events.
		if not logger.isEnabledFor(logging.DEBUG) and self._on_activity is None:
			return frame

		if not isinstance(frame, av.AudioFrame):
			return frame

		rms = self._compute_rms_int16(cast(av.AudioFrame, frame))
		if rms is None:
			return frame

		now = time.monotonic()
		rms_start = int(getattr(self._config, "rms_start", 900))
		rms_stop = int(getattr(self._config, "rms_stop", 600))
		log_interval_sec = float(getattr(self._config, "log_interval_sec", 1.0))
		prev_talking = self._talking
		if not self._talking:
			if rms >= rms_start:
				self._talking = True
				self._last_log_ts = now
				logger.debug("audio activity %s state=talking rms=%s", self._label, rms)
		else:
			if rms <= rms_stop:
				self._talking = False
				logger.debug("audio activity %s state=silent rms=%s", self._label, rms)
			elif (now - self._last_log_ts) >= log_interval_sec:
				self._last_log_ts = now
				logger.debug("audio activity %s state=talking rms=%s", self._label, rms)

		if self._on_activity is not None and self._talking != prev_talking:
			try:
				self._on_activity(self._talking, int(rms), self._label)
			except Exception:
				# Never break media pipeline due to UI/debug callbacks.
				pass

		return frame

	def stop(self) -> None:  # type: ignore[override]
		try:
			stop = getattr(self._source, "stop", None)
			if callable(stop):
				stop()
		except Exception:
			pass
		finally:
			super().stop()

	@staticmethod
	def _compute_rms_int16(frame: av.AudioFrame) -> Optional[int]:
		"""Return int16 RMS estimate, or None if unsupported."""
		try:
			fmt = str(getattr(frame, "format", "") or "")
			fmt_name = getattr(getattr(frame, "format", None), "name", None)
			fmt_s = str(fmt_name or fmt)
		except Exception:
			fmt_s = ""

		# Fast path for s16 / s16p using audioop when numpy isn't available.
		if "s16" in fmt_s and np is None:
			try:
				plane0 = frame.planes[0]
				pcm = bytes(plane0)
				samples = array.array("h")
				samples.frombytes(pcm)
				if not samples:
					return 0
				sum_sq = 0.0
				for s in samples:
					sum_sq += float(s) * float(s)
				rms = (sum_sq / float(len(samples))) ** 0.5
				return int(rms)
			except Exception:
				return None

		# Numpy-based path supports more formats, and is usually available when
		# using sounddevice on Windows.
		if np is None:
			return None
		assert np is not None

		try:
			arr = frame.to_ndarray()
			# Common shapes: (channels, samples) for planar, or (samples, channels)
			if arr.ndim == 2 and arr.shape[0] in (1, 2) and arr.shape[0] < arr.shape[1]:
				arr = arr.T
			# Mixdown to mono
			if arr.ndim == 2:
				arr = arr.mean(axis=1)
			arr = arr.astype(np.float32, copy=False)
			rms = float(np.sqrt(np.mean(arr * arr)))
			# If source is float, rms may be in [-1,1] scale.
			if rms <= 2.0:
				rms *= 32768.0
			return int(rms)
		except Exception:
			return None


def wrap_with_audio_activity_log(
	track: Optional[MediaStreamTrack],
	*,
	label: str,
	config: Optional[AudioActivityConfig] = None,
	on_activity: Optional[Callable[[bool, int, str], None]] = None,
) -> Optional[MediaStreamTrack]:
	if track is None:
		return None
	try:
		return AudioActivityLogTrack(track, label=label, config=config, on_activity=on_activity)
	except Exception:
		# Never break call setup due to debug-only logging.
		return track


@dataclass(frozen=True)
class AudioDevice:
	"""A selectable audio device.

	`backend` matches the ffmpeg/aiortc format string (e.g. "pulse", "alsa").
	`device` is the ffmpeg device name passed to MediaPlayer/MediaRecorder.
	"""

	backend: str
	device: Any
	label: str


def _is_windows() -> bool:
	# sys.platform is reliable in Python; platform.system() is for logs.
	return sys.platform.startswith("win")


_LOOPBACK_SUFFIX_RE = re.compile(r"\s*\(\s*loopback\s*\)\s*$", re.IGNORECASE)
_END_PAREN_RE = re.compile(r"\(([^()]*)\)\s*$")
_WINDOWS_ENDPOINT_PREFIX_RE = re.compile(
	r"^(?:speakers|headphones|microphone|headset microphone|line|digital output)\b\s*",
	re.IGNORECASE,
)


def _normalize_device_name_for_dedupe(name: str) -> str:
	"""Normalize endpoint names for dedupe and cross-library matching."""
	s = (name or "").strip()
	s = _LOOPBACK_SUFFIX_RE.sub("", s).strip()
	s = re.sub(r"\s+", " ", s)
	return s.casefold()


def _device_match_keys(name: str) -> set[str]:
	"""Return a set of normalized keys for cross-library matching.

	SoundCard and PortAudio sometimes label the same endpoint differently
	(e.g. "Speakers (...)" vs "Headphones (...)"), so we generate a few keys.
	"""
	norm = _normalize_device_name_for_dedupe(name)
	keys: set[str] = set()
	if norm:
		keys.add(norm)
		# Strip common Windows direction/prefix words.
		stripped = _WINDOWS_ENDPOINT_PREFIX_RE.sub("", norm).strip()
		if stripped and stripped != norm:
			keys.add(stripped)
		# Add the last parenthetical section as a key, which often contains the
		# stable product name.
		m = _END_PAREN_RE.search(norm)
		if m:
			paren = _normalize_device_name_for_dedupe(m.group(1))
			if paren:
				keys.add(paren)
	return keys


@dataclass(frozen=True)
class _EndpointCandidate:
	index: int
	name: str
	normalized: str
	channels: int
	default_samplerate: float


def _sounddevice_wasapi_hostapi_indices() -> set[int]:
	"""Return PortAudio host API indices for WASAPI."""
	if sd is None:
		return set()
	try:
		hostapis = sd.query_hostapis()  # type: ignore[attr-defined]
	except Exception:
		return set()
	if _audio_enum_dump_enabled():
		try:
			for i, h in enumerate(hostapis):
				_audio_enum_log("audio enum hostapi idx=%s name=%s", i, str(h.get("name", "") or ""))
		except Exception:
			pass
	indices: set[int] = set()
	for i, h in enumerate(hostapis):
		try:
			name = str(h.get("name", "") or "")
		except Exception:
			name = ""
		if "wasapi" in name.casefold():
			indices.add(int(i))
	return indices


def _collect_wasapi_candidates(*, direction: str) -> list[_EndpointCandidate]:
	"""(1)(2) Enumerate sounddevice devices, restricted to WASAPI and direction."""
	if sd is None:
		return []
	allowed_hostapis = _sounddevice_wasapi_hostapi_indices()
	_audio_enum_log(
		"audio enum start direction=%s allowed_wasapi_hostapis=%s",
		direction,
		sorted(allowed_hostapis),
	)
	if not allowed_hostapis:
		return []

	expected_key = "max_input_channels" if direction == "input" else "max_output_channels"
	try:
		devices = list(sd.query_devices())
	except Exception:
		return []

	if _audio_enum_dump_enabled():
		try:
			hostapis = sd.query_hostapis()  # type: ignore[attr-defined]
			hostapi_names = {i: str(h.get("name", "") or "") for i, h in enumerate(hostapis)}
		except Exception:
			hostapi_names = {}
		for idx, dev in enumerate(devices):
			try:
				name = str(dev.get("name", "") or "").strip()
				hostapi_idx = int(dev.get("hostapi", -1))
				in_ch = int(dev.get("max_input_channels", 0) or 0)
				out_ch = int(dev.get("max_output_channels", 0) or 0)
				sr = float(dev.get("default_samplerate", 0.0) or 0.0)
			except Exception:
				continue
			_audio_enum_log(
				"audio enum dev idx=%s hostapi=%s(%s) in=%s out=%s sr=%s name=%s",
				idx,
				hostapi_idx,
				hostapi_names.get(hostapi_idx, ""),
				in_ch,
				out_ch,
				sr,
				name,
			)

	candidates: list[_EndpointCandidate] = []
	for idx, dev in enumerate(devices):
		try:
			hostapi_idx = int(dev.get("hostapi", -1))
			channels = int(dev.get(expected_key, 0) or 0)
			name = str(dev.get("name", "") or "").strip()
			default_sr = float(dev.get("default_samplerate", 0.0) or 0.0)
		except Exception:
			continue

		if hostapi_idx not in allowed_hostapis:
			if _audio_enum_dump_enabled() and name:
				_audio_enum_log(
					"audio enum skip_non_wasapi direction=%s idx=%s hostapi=%s name=%s",
					direction,
					idx,
					hostapi_idx,
					name,
				)
			continue
		if channels <= 0:
			if _audio_enum_dump_enabled() and name:
				_audio_enum_log(
					"audio enum skip_no_channels direction=%s idx=%s name=%s",
					direction,
					idx,
					name,
				)
			continue
		if not name:
			continue
		norm = _normalize_device_name_for_dedupe(name)
		if not norm:
			continue
		candidates.append(
			_EndpointCandidate(
				index=int(idx),
				name=name,
				normalized=norm,
				channels=int(channels),
				default_samplerate=float(default_sr),
			)
		)

	_audio_enum_log("audio enum raw_candidates=%s", len(candidates))
	return candidates


def _dedupe_candidates(candidates: list[_EndpointCandidate]) -> list[_EndpointCandidate]:
	"""(3) Dedupe within WASAPI by normalized name, choosing the best candidate."""
	best_by_norm: dict[str, _EndpointCandidate] = {}
	for c in candidates:
		prev = best_by_norm.get(c.normalized)
		if prev is None or (c.channels, c.default_samplerate, -c.index) > (
			prev.channels,
			prev.default_samplerate,
			-prev.index,
		):
			best_by_norm[c.normalized] = c
	selected = list(best_by_norm.values())
	selected.sort(key=lambda x: x.name.casefold())
	_audio_enum_log("audio enum deduped=%s", len(selected))
	return selected


def _is_openable(*, candidate: _EndpointCandidate, direction: str) -> bool:
	"""(4) Drop devices PortAudio reports as unopenable."""
	if sd is None:
		return False
	sd_mod = cast(Any, sd)

	def _check(sr: float) -> bool:
		try:
			samplerate = float(sr)
			if samplerate <= 0:
				samplerate = None  # type: ignore[assignment]
			dtype = "int16"
			if direction == "input":
				test_channels = 1
				sd_mod.check_input_settings(
					device=candidate.index,
					channels=test_channels,
					samplerate=samplerate,
					dtype=dtype,
				)
			else:
				test_channels = 2 if candidate.channels >= 2 else 1
				sd_mod.check_output_settings(
					device=candidate.index,
					channels=test_channels,
					samplerate=samplerate,
					dtype=dtype,
				)
			return True
		except Exception:
			return False

	# Prefer 48kHz for the app; fall back to the device default.
	return _check(48000.0) or _check(candidate.default_samplerate)


def _filter_openable(candidates: list[_EndpointCandidate], *, direction: str) -> list[_EndpointCandidate]:
	openable: list[_EndpointCandidate] = []
	for c in candidates:
		if _is_openable(candidate=c, direction=direction):
			openable.append(c)
		else:
			_audio_enum_log("audio enum drop_unopenable idx=%s name=%s", c.index, c.name)
	_audio_enum_log("audio enum openable=%s", len(openable))
	return openable


def _soundcard_active_endpoint_keys(*, direction: str) -> set[str]:
	"""(5) Active Windows endpoints (plugged in / enabled) via SoundCard."""
	if not _is_windows():
		return set()
	try:
		import soundcard as sc  # type: ignore
	except Exception:
		if _audio_enum_dump_enabled():
			_audio_enum_log("audio enum soundcard import failed direction=%s", direction)
		return set()

	try:
		endpoints = sc.all_microphones() if direction == "input" else sc.all_speakers()
	except Exception:
		return set()

	keys: set[str] = set()
	for e in endpoints:
		try:
			name = str(getattr(e, "name", "") or "")
		except Exception:
			name = ""
		keys.update(_device_match_keys(name))
	_audio_enum_log("audio enum soundcard active_%s keys=%s", direction, len(keys))
	return keys



def _matches_active(candidate_name: str, active_keys: set[str]) -> bool:
	if not active_keys:
		return True
	ck = _device_match_keys(candidate_name)
	if ck & active_keys:
		return True
	# Fallback fuzzy match across keys.
	for c in ck:
		for a in active_keys:
			if c and a and (c in a or a in c):
				return True
	return False


def _filter_active_endpoints(candidates: list[_EndpointCandidate], *, direction: str) -> list[_EndpointCandidate]:
	active_keys = _soundcard_active_endpoint_keys(direction=direction)
	if not active_keys:
		return candidates
	filtered: list[_EndpointCandidate] = []
	dropped: list[_EndpointCandidate] = []
	for c in candidates:
		if _matches_active(c.name, active_keys):
			filtered.append(c)
		else:
			dropped.append(c)
	if not filtered and candidates:
		_audio_enum_log("audio enum active_filter empty; keeping unfiltered")
		return candidates
	if dropped:
		_audio_enum_log("audio enum active_filter dropped=%s", len(dropped))
		for d in dropped:
			_audio_enum_log("audio enum drop_inactive idx=%s name=%s", d.index, d.name)
	_audio_enum_log("audio enum active_filtered=%s", len(filtered))
	return filtered


def _pick_default_candidate(candidates: list[_EndpointCandidate], *, direction: str) -> Optional[_EndpointCandidate]:
	"""Pick a default candidate for the 'System default' entry."""
	if not candidates:
		return None

	# Prefer SoundCard default endpoint when available.
	default_norm: Optional[str] = None
	if _is_windows():
		try:
			import soundcard as sc  # type: ignore
			ep = sc.default_microphone() if direction == "input" else sc.default_speaker()
			default_norm = _normalize_device_name_for_dedupe(str(getattr(ep, "name", "") or ""))
		except Exception:
			default_norm = None
	if default_norm:
		for c in candidates:
			if _matches_active(c.name, _device_match_keys(default_norm)):
				return c

	# Else: best overall (more channels, higher samplerate).
	return max(candidates, key=lambda x: (int(x.channels), float(x.default_samplerate), -int(x.index)))


class SoundDeviceAudioTrack(MediaStreamTrack):
	kind = "audio"

	def __init__(
		self,
		*,
		device: Any = None,
		samplerate: int = 48000,
		channels: int = 1,
		blocksize: int = 960,
	):
		super().__init__()
		self._samplerate = int(samplerate)
		self._channels = int(channels)
		self._blocksize = int(blocksize)
		self._queue: Queue[bytes] = Queue(maxsize=50)
		self._timestamp = 0
		self._time_base = Fraction(1, self._samplerate)
		self._stream = None

		if sd is None or np is None:
			raise RuntimeError("sounddevice/numpy not available")

		def _callback(indata, frames, time, status) -> None:  # noqa: ANN001
			try:
				self._queue.put_nowait(bytes(indata))
			except Exception:
				# Drop if consumer is too slow.
				pass

		# Raw stream gives us bytes directly (int16 PCM).
		self._stream = sd.RawInputStream(
			samplerate=self._samplerate,
			channels=self._channels,
			dtype="int16",
			blocksize=self._blocksize,
			device=device,
			callback=_callback,
		)
		self._stream.start()
		logger.info(
			"local audio using sounddevice os=%s device=%s rate=%s ch=%s",
			platform.system(),
			device,
			self._samplerate,
			self._channels,
		)

	async def recv(self):  # type: ignore[override]
		if self.readyState != "live":
			raise asyncio.CancelledError
		if np is None:
			raise RuntimeError("numpy not available")
		assert np is not None

		loop = asyncio.get_running_loop()
		data = await loop.run_in_executor(None, self._queue.get)

		sample_width = 2
		samples = int(len(data) / (self._channels * sample_width))
		arr = np.frombuffer(data, dtype=np.int16)
		try:
			arr = arr.reshape((samples, self._channels))
		except Exception:
			# If shape is odd, drop.
			return await self.recv()

		layout = "mono" if self._channels == 1 else "stereo"
		frame = av.AudioFrame.from_ndarray(arr, format="s16", layout=layout)
		frame.sample_rate = self._samplerate
		frame.pts = self._timestamp
		frame.time_base = self._time_base
		self._timestamp += samples
		return frame

	def stop(self) -> None:  # type: ignore[override]
		try:
			if self._stream is not None:
				self._stream.stop()
				self._stream.close()
		except Exception:
			pass
		finally:
			self._stream = None
			super().stop()


def list_audio_inputs() -> list[AudioDevice]:
	"""List microphone capture devices.

	Windows-only discovery: WASAPI-only via sounddevice + soundcard active endpoints.
	"""
	_audio_enum_log("audio enum inputs start os=%s is_windows=%s", platform.system(), _is_windows())
	if not _is_windows() or sd is None:
		# Non-Windows is out of scope for this app.
		return [AudioDevice(backend="sounddevice", device="default", label="System default")]

	cands = _collect_wasapi_candidates(direction="input")
	cands = _dedupe_candidates(cands)
	cands = _filter_openable(cands, direction="input")
	cands = _filter_active_endpoints(cands, direction="input")

	default_cand = _pick_default_candidate(cands, direction="input")
	devices: list[AudioDevice] = []
	if default_cand is not None:
		# Keep the system default entry but also retain the named device so
		# explicit devices (e.g. Razer) aren't hidden behind the "System default" label.
		devices.append(AudioDevice(backend="sounddevice", device=default_cand.index, label="System default"))
	else:
		# If WASAPI isn't available for some reason, keep a fallback entry.
		devices.append(AudioDevice(backend="sounddevice", device="default", label="System default"))

	for c in cands:
		devices.append(AudioDevice(backend="sounddevice", device=c.index, label=c.name))

	_audio_enum_log("audio enum inputs final_count=%s", len(devices))
	for d in devices:
		_audio_enum_log("audio enum inputs final backend=%s device=%s label=%s", d.backend, d.device, d.label)
	return devices


def list_audio_outputs() -> list[AudioDevice]:
	"""List speaker/headphone output devices.

	Windows-only discovery: WASAPI-only via sounddevice + soundcard active endpoints.
	"""
	_audio_enum_log("audio enum outputs start os=%s is_windows=%s", platform.system(), _is_windows())
	if not _is_windows() or sd is None:
		return [AudioDevice(backend="sounddevice", device="default", label="System default")]

	cands = _collect_wasapi_candidates(direction="output")
	cands = _dedupe_candidates(cands)
	cands = _filter_openable(cands, direction="output")
	cands = _filter_active_endpoints(cands, direction="output")

	default_cand = _pick_default_candidate(cands, direction="output")
	devices: list[AudioDevice] = []
	if default_cand is not None:
		# Keep the system default entry but also retain the named device so
		# explicit devices (e.g. Razer) aren't hidden behind the "System default" label.
		devices.append(AudioDevice(backend="sounddevice", device=default_cand.index, label="System default"))
	else:
		devices.append(AudioDevice(backend="sounddevice", device="default", label="System default"))

	for c in cands:
		devices.append(AudioDevice(backend="sounddevice", device=c.index, label=c.name))

	_audio_enum_log("audio enum outputs final_count=%s", len(devices))
	for d in devices:
		_audio_enum_log("audio enum outputs final backend=%s device=%s label=%s", d.backend, d.device, d.label)
	return devices


def _try_create_player(preferred: Optional[AudioDevice] = None) -> Tuple[Optional[MediaPlayer], Optional[str]]:
	"""Try to create a microphone capture player.

	If a preferred device is provided, try it first, then fall back to common
	defaults.
	"""
	if preferred is not None:
		try:
			player = MediaPlayer(preferred.device, format=preferred.backend)
			return player, preferred.backend
		except Exception:
			pass

	# PulseAudio is typical on desktop Linux
	try:
		player = MediaPlayer("default", format="pulse")
		return player, "pulse"
	except Exception:
		pass

	# ALSA fallback
	try:
		player = MediaPlayer("default", format="alsa")
		return player, "alsa"
	except Exception:
		pass

	return None, None


@dataclass
class LocalAudio:
	"""Owns the underlying media player so its track stays alive."""

	player: Optional[MediaPlayer]
	track: Optional[MediaStreamTrack]
	backend: Optional[str] = None

	@classmethod
	def create(
		cls,
		preferred_input: Optional[AudioDevice] = None,
		*,
		activity_config: Optional[AudioActivityConfig] = None,
		on_activity: Optional[Callable[[bool, int, str], None]] = None,
	) -> "LocalAudio":
		# Windows-first: capture via sounddevice to get real device switching.
		if _is_windows() and sd is not None and np is not None:
			device = None
			if preferred_input is not None and preferred_input.backend == "sounddevice":
				if preferred_input.device != "default":
					device = preferred_input.device
			try:
				track = SoundDeviceAudioTrack(device=device)
				track = cast(
					MediaStreamTrack,
					wrap_with_audio_activity_log(
						track,
						label="TX backend=sounddevice",
						config=activity_config,
						on_activity=on_activity,
					)
					or track,
				)
				logger.info("local audio backend=sounddevice track=%s", bool(track))
				return cls(player=None, track=track, backend="sounddevice")
			except Exception as e:
				logger.warning("sounddevice capture init failed: %s", e)

		player, backend = _try_create_player(preferred_input)
		track = player.audio if player else None
		track = wrap_with_audio_activity_log(
			track,
			label=f"TX backend={backend or 'unknown'}",
			config=activity_config,
			on_activity=on_activity,
		)
		logger.info("local audio backend=%s track=%s", backend, bool(track))
		return cls(player=player, track=track, backend=backend)

	def close(self) -> None:
		"""Best-effort stop for the underlying ffmpeg process."""
		p = self.player
		t = self.track
		self.player = None
		self.track = None
		try:
			if t is not None:
				t.stop()
		except Exception:
			pass
		if p is None:
			return
		try:
			stop = getattr(p, "_stop", None)
			if callable(stop):
				stop()
		except Exception:
			pass


@dataclass
class RemoteAudioSink:
	"""Consumes remote audio tracks.

	If playback to the default device isn't supported, falls back to discarding.
	"""

	output: Optional[AudioDevice] = None
	_recorder: Optional[Any] = None
	_task: Optional[asyncio.Task[None]] = None
	_started: bool = False

	async def start(self, track: MediaStreamTrack) -> None:
		if self._started:
			return

		# Windows-first: play via sounddevice.
		if _is_windows() and sd is not None and np is not None:
			import numpy as np_local  # type: ignore
			assert np_local is not None

			device = None
			if self.output is not None and self.output.backend == "sounddevice":
				if self.output.device != "default":
					device = self.output.device

			stream = sd.RawOutputStream(
				samplerate=48000,
				channels=1,
				dtype="int16",
				blocksize=960,
				device=device,
			)
			stream.start()
			sink = f"sounddevice:{device if device is not None else 'default'}"
			logger.info("remote audio sink=%s", sink)

			async def _pump() -> None:
				try:
					while True:
						frame = await track.recv()
						if not isinstance(frame, av.AudioFrame):
							continue
						aframe = cast(av.AudioFrame, frame)

						rate = int(getattr(aframe, "sample_rate", 48000) or 48000)
						if rate != 48000:
							# Keep MVP simple: assume 48kHz WebRTC audio.
							logger.debug("remote audio sample_rate=%s (expected 48000)", rate)

						arr = aframe.to_ndarray()
						# Shape can be (channels, samples) for planar.
						if arr.ndim == 2 and arr.shape[0] in (1, 2) and arr.shape[0] < arr.shape[1]:
							arr = arr.T
						if arr.ndim == 2 and arr.shape[1] > 1:
							arr = arr.mean(axis=1, keepdims=True)
						if arr.ndim == 1:
							arr = arr.reshape((-1, 1))
						# Convert to int16.
						if arr.dtype != np_local.int16:
							arr = np_local.clip(arr, -32768, 32767).astype(np_local.int16, copy=False)
						data = arr.tobytes(order="C")
						stream.write(data)
				except asyncio.CancelledError:
					pass
				except Exception as e:
					logger.info("remote audio pump stopped: %s", e)
				finally:
					try:
						stream.stop()
						stream.close()
					except Exception:
						pass

			self._task = asyncio.create_task(_pump(), name="remote-audio-pump")
			self._started = True
			return

		# Best-effort playback to chosen output, else to system output.
		# Note: this depends on ffmpeg having appropriate sink support.
		sink = "blackhole"
		device = "default"
		if self.output is not None:
			device = self.output.device
			try:
				recorder = MediaRecorder(device, format=self.output.backend)
				sink = f"{self.output.backend}:{device}"
			except Exception:
				recorder = None
		else:
			recorder = None

		if recorder is None:
			try:
				recorder = MediaRecorder("default", format="pulse")
				sink = "pulse:default"
			except Exception:
				try:
					recorder = MediaRecorder("default", format="alsa")
					sink = "alsa:default"
				except Exception:
					recorder = MediaBlackhole()
					sink = "blackhole"

		logger.info("remote audio sink=%s track_kind=%s", sink, getattr(track, "kind", None))

		recorder.addTrack(track)
		await recorder.start()
		self._recorder = recorder
		self._started = True

	async def stop(self) -> None:
		if self._task is not None:
			self._task.cancel()
			try:
				await self._task
			except Exception:
				pass
			self._task = None
		if not self._started or not self._recorder:
			return
		try:
			await self._recorder.stop()
		finally:
			self._recorder = None
			self._started = False

