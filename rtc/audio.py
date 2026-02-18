"""Audio helpers for aiortc.

MVP scope:
- Create a local microphone audio track (best-effort on Linux).
- Provide a best-effort sink for remote audio (playback if possible, else discard).
"""

from __future__ import annotations

import asyncio
import array
import logging
import os
import platform
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


def _run_cmd_lines(argv: list[str], timeout_sec: float = 0.8) -> list[str]:
	try:
		p = subprocess.run(
			argv,
			check=False,
			stdout=subprocess.PIPE,
			stderr=subprocess.DEVNULL,
			text=True,
			timeout=timeout_sec,
		)
	except Exception:
		return []
	if not p.stdout:
		return []
	return [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]


def _pactl_default(kind: str) -> Optional[str]:
	"""Return PulseAudio default source/sink name, if available."""
	# kind: "Source" or "Sink"
	if shutil.which("pactl") is None:
		return None
	for ln in _run_cmd_lines(["pactl", "info"], timeout_sec=0.8):
		prefix = f"Default {kind}:"
		if ln.startswith(prefix):
			value = ln[len(prefix) :].strip()
			return value or None
	return None


def list_audio_inputs() -> list[AudioDevice]:
	"""Best-effort list of microphone capture devices."""
	devices: list[AudioDevice] = []

	# Windows-first: PortAudio device list
	if _is_windows() and sd is not None:
		try:
			devices.append(AudioDevice(backend="sounddevice", device="default", label="System default"))
			for idx, dev in enumerate(sd.query_devices()):
				try:
					if int(dev.get("max_input_channels", 0)) <= 0:
						continue
				except Exception:
					continue
				name = str(dev.get("name", f"Device {idx}"))
				devices.append(AudioDevice(backend="sounddevice", device=idx, label=name))
			return devices
		except Exception:
			# Fall through to linux probing.
			pass

	# PulseAudio / PipeWire Pulse compatibility
	if shutil.which("pactl") is not None:
		default_src = _pactl_default("Source")
		for ln in _run_cmd_lines(["pactl", "list", "short", "sources"], timeout_sec=1.2):
			parts = ln.split()  # index, name, driver, ...
			if len(parts) >= 2:
				name = parts[1]
				label = name
				if default_src and name == default_src:
					label = f"{name} (default)"
				devices.append(AudioDevice(backend="pulse", device=name, label=label))

	# ALSA fallback
	if shutil.which("arecord") is not None:
		for ln in _run_cmd_lines(["arecord", "-L"], timeout_sec=1.2):
			# arecord -L includes commentary lines; device ids are non-indented
			if ln.startswith(" ") or ln.startswith("\t"):
				continue
			name = ln.strip()
			if not name:
				continue
			devices.append(AudioDevice(backend="alsa", device=name, label=name))

	# Always allow "default" as a safe option.
	if not any(d.backend == "pulse" and d.device == "default" for d in devices):
		devices.insert(0, AudioDevice(backend="pulse", device="default", label="System default"))
	return devices


def list_audio_outputs() -> list[AudioDevice]:
	"""Best-effort list of speaker/headphone output devices."""
	devices: list[AudioDevice] = []

	if _is_windows() and sd is not None:
		try:
			devices.append(AudioDevice(backend="sounddevice", device="default", label="System default"))
			for idx, dev in enumerate(sd.query_devices()):
				try:
					if int(dev.get("max_output_channels", 0)) <= 0:
						continue
				except Exception:
					continue
				name = str(dev.get("name", f"Device {idx}"))
				devices.append(AudioDevice(backend="sounddevice", device=idx, label=name))
			return devices
		except Exception:
			pass

	if shutil.which("pactl") is not None:
		default_sink = _pactl_default("Sink")
		for ln in _run_cmd_lines(["pactl", "list", "short", "sinks"], timeout_sec=1.2):
			parts = ln.split()
			if len(parts) >= 2:
				name = parts[1]
				label = name
				if default_sink and name == default_sink:
					label = f"{name} (default)"
				devices.append(AudioDevice(backend="pulse", device=name, label=label))

	if shutil.which("aplay") is not None:
		for ln in _run_cmd_lines(["aplay", "-L"], timeout_sec=1.2):
			if ln.startswith(" ") or ln.startswith("\t"):
				continue
			name = ln.strip()
			if not name:
				continue
			devices.append(AudioDevice(backend="alsa", device=name, label=name))

	if not any(d.backend == "pulse" and d.device == "default" for d in devices):
		devices.insert(0, AudioDevice(backend="pulse", device="default", label="System default"))
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

