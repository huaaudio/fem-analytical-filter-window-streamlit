"""Accessible, synchronized audio comparison for the public demo.

The preferred renderer uses Streamlit Components v2, which keeps the player
code scoped to a shadow root and lets it inherit Streamlit's theme variables.
Older Streamlit releases fall back to a compact list of native ``st.audio``
players so audio is never hidden behind a version-dependent component API.
"""

from __future__ import annotations

import base64
import io
import math
import wave
from collections.abc import Mapping
from typing import Any

import numpy as np
import streamlit as st

try:  # Components v2 was introduced after the original app was written.
    from streamlit.components.v2 import component as _declare_component
except (AttributeError, ImportError):  # pragma: no cover - version dependent
    _declare_component = None


_COMPONENT_HTML = r"""
<section class="audio-compare" data-audio-compare-root aria-labelledby="audio-compare-title">
  <header class="audio-compare__header">
    <div>
      <h3 id="audio-compare-title">Listening comparison</h3>
      <p>Switch versions while the sound plays to compare the same moment.</p>
    </div>
    <p class="audio-compare__listening-note">Headphones recommended</p>
  </header>

  <div class="audio-compare__versions" role="radiogroup" aria-label="Audio version"></div>

  <div class="audio-compare__selection" aria-live="polite" aria-atomic="true">
    <span class="audio-compare__swatch" aria-hidden="true"></span>
    <div class="audio-compare__selection-copy">
      <strong class="audio-compare__selected-label"></strong>
      <span class="audio-compare__description"></span>
    </div>
    <span class="audio-compare__level"></span>
  </div>

  <div class="audio-compare__transport">
    <button class="audio-compare__play" type="button" disabled></button>
    <div class="audio-compare__timeline">
      <label class="sr-only" for="audio-compare-seek">Playback position</label>
      <input
        id="audio-compare-seek"
        class="audio-compare__seek"
        type="range"
        min="0"
        max="0.01"
        step="0.01"
        value="0"
        disabled
      />
      <div class="audio-compare__times" aria-hidden="true">
        <span class="audio-compare__current-time">0:00</span>
        <span class="audio-compare__duration">0:00</span>
      </div>
    </div>
  </div>

  <p class="audio-compare__message" role="status" aria-live="polite"></p>
  <div class="audio-compare__media" aria-hidden="true"></div>
</section>
"""


_COMPONENT_CSS = r"""
:host {
  color-scheme: light dark;
}

* {
  box-sizing: border-box;
}

.audio-compare {
  --player-original: #364152;
  --player-bare: #52616f;
  --player-analytical: #007c83;
  --player-fem: #b84e32;
  --player-infinite: #7a8793;
  --player-model: var(--st-primary-color, #007c83);
  --player-accent: var(--player-original);

  width: 100%;
  color: var(--st-text-color, #102a43);
  background: var(--st-background-color, #ffffff);
  border: 1px solid var(--st-border-color, #d8e2ea);
  border-radius: max(var(--st-base-radius, 0.75rem), 0.75rem);
  padding: clamp(1rem, 2.8vw, 1.5rem);
  font-family: var(--st-font, "Inter", system-ui, sans-serif);
  font-size: var(--st-base-font-size, 1rem);
  box-shadow: 0 0.5rem 1.5rem rgba(16, 42, 67, 0.06);
}

.audio-compare__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1.25rem;
}

.audio-compare__header h3,
.audio-compare__header p,
.audio-compare__listening-note,
.audio-compare__message {
  margin: 0;
}

.audio-compare__header h3 {
  color: var(--st-heading-color, var(--st-text-color, #102a43));
  font-family: var(--st-heading-font, var(--st-font, "Inter", system-ui, sans-serif));
  font-size: 1.125rem;
  font-weight: 700;
  line-height: 1.3;
}

.audio-compare__header > div > p {
  max-width: 40rem;
  margin-top: 0.25rem;
  color: color-mix(in srgb, var(--st-text-color, #102a43) 72%, transparent);
  font-size: 0.925rem;
  line-height: 1.5;
}

.audio-compare__listening-note {
  flex: 0 0 auto;
  padding-top: 0.125rem;
  color: color-mix(in srgb, var(--st-text-color, #102a43) 70%, transparent);
  font-size: 0.8rem;
  font-weight: 600;
  line-height: 1.4;
}

.audio-compare__versions {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr));
  gap: 0.5rem;
}

.audio-compare__version {
  --track-accent: var(--player-model);

  min-width: 0;
  min-height: 2.75rem;
  padding: 0.6rem 0.75rem;
  overflow-wrap: anywhere;
  color: var(--st-text-color, #102a43);
  background: var(--st-secondary-background-color, #f4f7fa);
  border: 1px solid var(--st-border-color, #d8e2ea);
  border-radius: var(--st-button-radius, 0.55rem);
  font: inherit;
  font-size: 0.875rem;
  font-weight: 650;
  line-height: 1.25;
  text-align: center;
  white-space: normal;
  cursor: pointer;
  transition:
    background-color 150ms ease,
    border-color 150ms ease,
    box-shadow 150ms ease,
    color 150ms ease,
    transform 150ms ease;
}

.audio-compare__version[data-kind="original"] {
  --track-accent: var(--player-original);
}

.audio-compare__version[data-kind="bare"] {
  --track-accent: var(--player-bare);
}

.audio-compare__version[data-kind="analytical"] {
  --track-accent: var(--player-analytical);
}

.audio-compare__version[data-kind="fem"] {
  --track-accent: var(--player-fem);
}

.audio-compare__version[data-kind="infinite"] {
  --track-accent: var(--player-infinite);
}

.audio-compare__version:hover:not([aria-checked="true"]) {
  border-color: color-mix(in srgb, var(--track-accent) 52%, var(--st-border-color, #d8e2ea));
  background: color-mix(in srgb, var(--track-accent) 7%, var(--st-background-color, #ffffff));
}

.audio-compare__version[aria-checked="true"] {
  color: #ffffff;
  background: var(--track-accent);
  border-color: var(--track-accent);
  box-shadow: 0 0.25rem 0.75rem color-mix(in srgb, var(--track-accent) 24%, transparent);
}

.audio-compare__version:focus-visible,
.audio-compare__play:focus-visible,
.audio-compare__seek:focus-visible {
  outline: 3px solid color-mix(in srgb, var(--st-primary-color, #007c83) 36%, transparent);
  outline-offset: 3px;
}

.audio-compare__selection {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: center;
  gap: 0.75rem;
  min-height: 4.25rem;
  margin-top: 1rem;
  padding: 0.75rem 0;
  border-bottom: 1px solid var(--st-border-color, #d8e2ea);
}

.audio-compare__swatch {
  width: 0.65rem;
  height: 0.65rem;
  background: var(--player-accent);
  border-radius: 50%;
}

.audio-compare__selection-copy {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 0.15rem;
}

.audio-compare__selected-label {
  overflow-wrap: anywhere;
  color: var(--st-heading-color, var(--st-text-color, #102a43));
  font-size: 0.95rem;
  line-height: 1.35;
}

.audio-compare__description {
  overflow-wrap: anywhere;
  color: color-mix(in srgb, var(--st-text-color, #102a43) 68%, transparent);
  font-size: 0.825rem;
  line-height: 1.4;
}

.audio-compare__level {
  color: var(--player-accent);
  font-size: 0.825rem;
  font-weight: 700;
  line-height: 1.35;
  text-align: right;
}

.audio-compare__transport {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  align-items: center;
  gap: 1rem;
  padding-top: 1rem;
}

.audio-compare__play {
  display: inline-grid;
  width: 3.25rem;
  height: 3.25rem;
  place-items: center;
  color: #ffffff;
  background: var(--player-accent);
  border: 0;
  border-radius: 50%;
  cursor: pointer;
  transition:
    filter 150ms ease,
    transform 150ms ease;
}

.audio-compare__play:hover:not(:disabled) {
  filter: brightness(0.92);
  transform: translateY(-1px);
}

.audio-compare__play:disabled {
  cursor: wait;
  filter: saturate(0.25);
  opacity: 0.52;
}

.audio-compare__play svg {
  display: block;
  width: 1.25rem;
  height: 1.25rem;
}

.audio-compare__timeline {
  min-width: 0;
}

.audio-compare__seek {
  display: block;
  width: 100%;
  height: 1.5rem;
  margin: 0;
  accent-color: var(--player-accent);
  cursor: pointer;
}

.audio-compare__seek:disabled {
  cursor: wait;
  opacity: 0.55;
}

.audio-compare__times {
  display: flex;
  justify-content: space-between;
  margin-top: -0.1rem;
  color: color-mix(in srgb, var(--st-text-color, #102a43) 62%, transparent);
  font-variant-numeric: tabular-nums;
  font-size: 0.75rem;
  line-height: 1.3;
}

.audio-compare__message {
  display: none;
  margin-top: 0.75rem;
  padding: 0.6rem 0.75rem;
  color: var(--st-red-text-color, #8b1a1a);
  background: var(--st-red-background-color, #fff0f0);
  border-radius: var(--st-button-radius, 0.55rem);
  font-size: 0.825rem;
  line-height: 1.4;
}

.audio-compare__message.is-visible {
  display: block;
}

.audio-compare__media {
  display: none;
}

.sr-only {
  position: absolute !important;
  width: 1px !important;
  height: 1px !important;
  padding: 0 !important;
  overflow: hidden !important;
  clip: rect(0, 0, 0, 0) !important;
  white-space: nowrap !important;
  border: 0 !important;
}

@media (max-width: 40rem) {
  .audio-compare {
    padding: 1rem;
  }

  .audio-compare__header {
    display: block;
    margin-bottom: 1rem;
  }

  .audio-compare__listening-note {
    margin-top: 0.45rem;
  }

  .audio-compare__versions {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.5rem;
  }

  .audio-compare__version {
    width: 100%;
    min-width: 0;
  }

  .audio-compare__selection {
    grid-template-columns: auto minmax(0, 1fr);
  }

  .audio-compare__level {
    grid-column: 2;
    text-align: left;
  }

  .audio-compare__transport {
    gap: 0.75rem;
  }
}

@media (prefers-reduced-motion: reduce) {
  .audio-compare__version,
  .audio-compare__play {
    transition: none;
  }
}
"""


_COMPONENT_JS = r"""
const PLAY_ICON = `
  <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
    <path d="M8.2 5.8a1 1 0 0 1 1.53-.85l9.05 6.2a1 1 0 0 1 0 1.7l-9.05 6.2a1 1 0 0 1-1.53-.85V5.8Z" fill="currentColor"/>
  </svg>`;

const PAUSE_ICON = `
  <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
    <rect x="6.5" y="5" width="4" height="14" rx="1" fill="currentColor"/>
    <rect x="13.5" y="5" width="4" height="14" rx="1" fill="currentColor"/>
  </svg>`;

export default function(component) {
  const { data, parentElement } = component;

  if (typeof parentElement.__metavisionAudioCleanup === "function") {
    parentElement.__metavisionAudioCleanup();
  }

  const root = parentElement.querySelector("[data-audio-compare-root]");
  const tracks = Array.isArray(data?.tracks) ? data.tracks : [];
  if (!root || tracks.length === 0) {
    if (root) {
      root.replaceChildren(document.createTextNode("Audio comparison is unavailable."));
    }
    return;
  }

  const versions = root.querySelector(".audio-compare__versions");
  const selectedLabel = root.querySelector(".audio-compare__selected-label");
  const description = root.querySelector(".audio-compare__description");
  const level = root.querySelector(".audio-compare__level");
  const playButton = root.querySelector(".audio-compare__play");
  const seek = root.querySelector(".audio-compare__seek");
  const seekLabel = root.querySelector('label[for="audio-compare-seek"]');
  const currentTime = root.querySelector(".audio-compare__current-time");
  const duration = root.querySelector(".audio-compare__duration");
  const message = root.querySelector(".audio-compare__message");
  const media = root.querySelector(".audio-compare__media");
  const cleanups = [];
  const audioById = new Map();
  const buttonById = new Map();
  const knownIds = new Set(tracks.map((track) => track.id));
  const savedState = parentElement.__metavisionAudioState || {};

  let selectedId = knownIds.has(savedState.selectedId)
    ? savedState.selectedId
    : tracks[0].id;
  let switchSequence = 0;

  const addListener = (target, event, handler, options) => {
    target.addEventListener(event, handler, options);
    cleanups.push(() => target.removeEventListener(event, handler, options));
  };

  const activeTrack = () => tracks.find((track) => track.id === selectedId) || tracks[0];
  const activeAudio = () => audioById.get(selectedId);
  const finiteNumber = (value, fallback = 0) =>
    Number.isFinite(Number(value)) ? Number(value) : fallback;
  const formatTime = (seconds) => {
    const wholeSeconds = Math.max(0, Math.floor(finiteNumber(seconds)));
    const minutes = Math.floor(wholeSeconds / 60);
    return `${minutes}:${String(wholeSeconds % 60).padStart(2, "0")}`;
  };

  const accentFor = (kind) => {
    const property = {
      original: "--player-original",
      bare: "--player-bare",
      analytical: "--player-analytical",
      fem: "--player-fem",
      infinite: "--player-infinite",
    }[kind] || "--player-model";
    return `var(${property})`;
  };

  const levelText = (track) => {
    if (track.kind === "original") return "Reference level";
    if (track.levelDelta === null || track.levelDelta === undefined) return "";
    const delta = finiteNumber(track.levelDelta);
    if (Math.abs(delta) < 0.05) return "Same level as original";
    return delta < 0
      ? `${Math.abs(delta).toFixed(1)} dB quieter`
      : `${delta.toFixed(1)} dB louder`;
  };

  const announceError = (text) => {
    message.textContent = text;
    message.classList.toggle("is-visible", Boolean(text));
  };

  const persistState = () => {
    const audio = activeAudio();
    parentElement.__metavisionAudioState = {
      selectedId,
      currentTime: finiteNumber(audio?.currentTime),
    };
  };

  const updatePlayButton = () => {
    const audio = activeAudio();
    const track = activeTrack();
    const isPlaying = Boolean(audio && !audio.paused && !audio.ended);
    playButton.innerHTML = isPlaying ? PAUSE_ICON : PLAY_ICON;
    playButton.setAttribute(
      "aria-label",
      `${isPlaying ? "Pause" : "Play"} ${track.label}`,
    );
    playButton.setAttribute("title", isPlaying ? "Pause" : "Play");
  };

  const updateTimeline = () => {
    const audio = activeAudio();
    const track = activeTrack();
    if (!audio) return;

    const total = Number.isFinite(audio.duration)
      ? audio.duration
      : finiteNumber(track.duration);
    const position = Math.min(finiteNumber(audio.currentTime), total || 0);
    seek.max = String(Math.max(total, 0.01));
    seek.value = String(position);
    seek.setAttribute("aria-valuetext", `${formatTime(position)} of ${formatTime(total)}`);
    currentTime.textContent = formatTime(position);
    duration.textContent = formatTime(total);
    persistState();
  };

  const updateSelection = () => {
    const track = activeTrack();
    root.style.setProperty("--player-accent", accentFor(track.kind));
    selectedLabel.textContent = track.label;
    description.textContent = track.description || "";
    level.textContent = levelText(track);
    seekLabel.textContent = `Playback position for ${track.label}`;

    buttonById.forEach((button, id) => {
      const selected = id === selectedId;
      button.setAttribute("aria-checked", String(selected));
      button.tabIndex = selected ? 0 : -1;
    });

    updateTimeline();
    updatePlayButton();
  };

  const setPosition = (audio, position) => {
    const total = Number.isFinite(audio.duration) ? audio.duration : Infinity;
    const safePosition = Math.max(0, Math.min(finiteNumber(position), total));
    try {
      audio.currentTime = safePosition;
    } catch (_) {
      // Metadata can still be loading. The loadedmetadata handler retries.
    }
  };

  const playActive = async () => {
    const audio = activeAudio();
    if (!audio) return;
    announceError("");
    try {
      await audio.play();
    } catch (_) {
      announceError("Playback could not start. Check your browser's audio permissions and try again.");
    }
    updatePlayButton();
  };

  const selectTrack = (nextId, moveFocus = false) => {
    if (!knownIds.has(nextId) || nextId === selectedId) {
      if (moveFocus) buttonById.get(selectedId)?.focus();
      return;
    }

    const outgoing = activeAudio();
    const position = finiteNumber(outgoing?.currentTime);
    const wasPlaying = Boolean(outgoing && !outgoing.paused && !outgoing.ended);
    outgoing?.pause();

    selectedId = nextId;
    switchSequence += 1;
    const thisSwitch = switchSequence;
    const incoming = activeAudio();

    const resumeAtSharedPosition = () => {
      if (thisSwitch !== switchSequence || incoming !== activeAudio()) return;
      setPosition(incoming, position);
      updateSelection();
      if (wasPlaying) void playActive();
    };

    if (incoming.readyState >= HTMLMediaElement.HAVE_METADATA) {
      resumeAtSharedPosition();
    } else {
      incoming.addEventListener("loadedmetadata", resumeAtSharedPosition, { once: true });
      cleanups.push(() => incoming.removeEventListener("loadedmetadata", resumeAtSharedPosition));
      updateSelection();
    }

    if (moveFocus) buttonById.get(selectedId)?.focus();
  };

  tracks.forEach((track) => {
    const audio = document.createElement("audio");
    audio.preload = "auto";
    audio.src = track.src;
    audioById.set(track.id, audio);
    media.appendChild(audio);

    addListener(audio, "loadedmetadata", () => {
      if (track.id !== selectedId) return;
      const restoredPosition = finiteNumber(savedState.currentTime);
      if (restoredPosition > 0 && finiteNumber(audio.currentTime) === 0) {
        setPosition(audio, restoredPosition);
      }
      playButton.disabled = false;
      seek.disabled = false;
      updateTimeline();
    });
    addListener(audio, "canplay", () => {
      if (track.id === selectedId) {
        playButton.disabled = false;
        seek.disabled = false;
      }
    });
    addListener(audio, "timeupdate", () => {
      if (track.id === selectedId) updateTimeline();
    });
    addListener(audio, "play", () => {
      if (track.id === selectedId) updatePlayButton();
    });
    addListener(audio, "pause", () => {
      if (track.id === selectedId) updatePlayButton();
    });
    addListener(audio, "ended", () => {
      if (track.id !== selectedId) return;
      audio.currentTime = 0;
      updateTimeline();
      updatePlayButton();
    });
    addListener(audio, "error", () => {
      if (track.id !== selectedId) return;
      playButton.disabled = true;
      seek.disabled = true;
      announceError(`The ${track.label} audio could not be loaded.`);
    });

    const button = document.createElement("button");
    button.className = "audio-compare__version";
    button.type = "button";
    button.dataset.kind = track.kind;
    button.setAttribute("role", "radio");
    button.setAttribute("aria-checked", String(track.id === selectedId));
    button.tabIndex = track.id === selectedId ? 0 : -1;
    button.textContent = track.label;
    buttonById.set(track.id, button);
    versions.appendChild(button);

    addListener(button, "click", () => selectTrack(track.id));
    addListener(button, "keydown", (event) => {
      const currentIndex = tracks.findIndex((item) => item.id === selectedId);
      let nextIndex = currentIndex;
      if (event.key === "ArrowRight" || event.key === "ArrowDown") {
        nextIndex = (currentIndex + 1) % tracks.length;
      } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
        nextIndex = (currentIndex - 1 + tracks.length) % tracks.length;
      } else if (event.key === "Home") {
        nextIndex = 0;
      } else if (event.key === "End") {
        nextIndex = tracks.length - 1;
      } else {
        return;
      }
      event.preventDefault();
      selectTrack(tracks[nextIndex].id, true);
    });
  });

  addListener(playButton, "click", () => {
    const audio = activeAudio();
    if (!audio) return;
    if (!audio.paused && !audio.ended) {
      audio.pause();
      return;
    }
    void playActive();
  });

  addListener(seek, "input", () => {
    const audio = activeAudio();
    if (!audio) return;
    setPosition(audio, seek.valueAsNumber);
    updateTimeline();
  });

  addListener(seek, "change", () => {
    message.textContent = `Playback position ${formatTime(seek.valueAsNumber)}.`;
    message.classList.remove("is-visible");
  });

  updateSelection();
  const firstAudio = activeAudio();
  if (firstAudio.readyState >= HTMLMediaElement.HAVE_METADATA) {
    setPosition(firstAudio, savedState.currentTime);
    playButton.disabled = false;
    seek.disabled = false;
    updateTimeline();
  }

  const cleanup = () => {
    persistState();
    cleanups.splice(0).forEach((removeListener) => removeListener());
    audioById.forEach((audio) => {
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
    });
    versions.replaceChildren();
    media.replaceChildren();
    if (parentElement.__metavisionAudioCleanup === cleanup) {
      parentElement.__metavisionAudioCleanup = null;
    }
  };

  parentElement.__metavisionAudioCleanup = cleanup;
  return cleanup;
}
"""


if _declare_component is not None:
    try:
        _AUDIO_COMPARE_COMPONENT = _declare_component(
            "metavision_audio_comparison",
            html=_COMPONENT_HTML,
            css=_COMPONENT_CSS,
            js=_COMPONENT_JS,
            isolate_styles=True,
        )
    except (AttributeError, TypeError):  # pragma: no cover - version dependent
        _AUDIO_COMPARE_COMPONENT = None
else:  # pragma: no cover - version dependent
    _AUDIO_COMPARE_COMPONENT = None


_DEFAULT_DESCRIPTIONS = {
    "original": "Unprocessed source audio",
    "bare": "A4-sized host panel without local resonators",
    "analytical": "A4-sized panel · analytical simulation",
    "fem": "A4-sized panel · FEM simulation",
    "infinite": "Idealized infinite-panel simulation",
    "model": "Simulated panel response",
}


def _track_kind(label: str) -> str:
    """Return a visual and semantic category without changing track order."""

    normalized = label.casefold()
    if "original" in normalized or normalized.strip() == "input":
        return "original"
    if "bare" in normalized:
        return "bare"
    if "infinite" in normalized:
        return "infinite"
    if "fem" in normalized or "finite element" in normalized:
        return "fem"
    if "analytical" in normalized or "a4" in normalized:
        return "analytical"
    return "model"


def _prepare_mono_signal(signal: np.ndarray, *, label: str) -> np.ndarray:
    """Convert a numeric mono signal to finite floating-point samples."""

    raw = np.asarray(signal)
    if raw.ndim != 1:
        raise ValueError(f"Signal {label!r} must be a one-dimensional mono array.")
    if raw.size == 0:
        raise ValueError(f"Signal {label!r} must contain at least one sample.")
    if not (
        np.issubdtype(raw.dtype, np.number)
        and not np.issubdtype(raw.dtype, np.complexfloating)
    ):
        raise TypeError(f"Signal {label!r} must contain real numeric samples.")

    if np.issubdtype(raw.dtype, np.unsignedinteger):
        info = np.iinfo(raw.dtype)
        midpoint = (float(info.max) + 1.0) / 2.0
        normalized = (raw.astype(np.float64) - midpoint) / midpoint
    elif np.issubdtype(raw.dtype, np.signedinteger):
        info = np.iinfo(raw.dtype)
        scale = float(max(abs(info.min), info.max))
        normalized = raw.astype(np.float64) / scale
    else:
        normalized = raw.astype(np.float64)

    return np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)


def _wav_data_uri(signal: np.ndarray, sample_rate: int) -> str:
    """Encode a normalized signal as a browser-safe 16-bit PCM WAV data URI."""

    pcm = np.rint(signal.astype(np.float64) * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def _safe_level_delta(level_deltas: Mapping[str, float] | None, label: str) -> float | None:
    if level_deltas is None or label not in level_deltas:
        return None
    try:
        value = float(level_deltas[label])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def build_audio_comparison_payload(
    signals: Mapping[str, np.ndarray],
    sample_rate: int,
    level_deltas_db: Mapping[str, float] | None = None,
    descriptions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the ordered, JSON-safe Components v2 payload.

    Samples are normalized as one group before PCM16 encoding. This avoids
    clipping while preserving the relative level differences listeners are
    meant to compare. Labels and descriptions remain data values; the browser
    renderer inserts them with ``textContent`` instead of interpolating HTML.
    """

    if not signals:
        raise ValueError("At least one audio signal is required.")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise TypeError("sample_rate must be a positive integer.")
    sample_rate = int(sample_rate)
    if sample_rate <= 0 or sample_rate > (2**32 - 1):
        raise ValueError("sample_rate must be between 1 and 4,294,967,295 Hz.")

    source_tracks: list[tuple[object, str, np.ndarray, str, str, float | None]] = []
    for index, (raw_label, raw_signal) in enumerate(signals.items()):
        label = str(raw_label).strip()
        if not label:
            raise ValueError("Audio track labels must not be empty.")
        signal = _prepare_mono_signal(raw_signal, label=label)
        kind = _track_kind(label)
        raw_description = descriptions.get(raw_label) if descriptions is not None else None
        description = (
            str(raw_description).strip()
            if raw_description is not None
            else _DEFAULT_DESCRIPTIONS[kind]
        )
        delta = _safe_level_delta(level_deltas_db, raw_label)
        source_tracks.append((raw_label, label, signal, kind, description, delta))

    group_peak = max(
        (float(np.max(np.abs(signal))) for _, _, signal, _, _, _ in source_tracks),
        default=0.0,
    )
    group_scale = group_peak if group_peak > 1e-12 else 1.0
    id_counts: dict[str, int] = {}
    tracks: list[dict[str, Any]] = []
    for index, (_, label, unscaled_signal, kind, description, delta) in enumerate(source_tracks):
        signal = np.clip(unscaled_signal / group_scale, -1.0, 1.0).astype(
            np.float32,
            copy=False,
        )
        id_counts[kind] = id_counts.get(kind, 0) + 1
        occurrence = id_counts[kind]
        track_id = kind if occurrence == 1 else f"{kind}-{occurrence}"
        tracks.append(
            {
                "id": track_id if kind != "model" else f"model-{index}",
                "label": label,
                "kind": kind,
                "description": description,
                "levelDelta": delta,
                "duration": signal.size / sample_rate,
                "src": _wav_data_uri(signal, sample_rate),
            }
        )
    return {"tracks": tracks, "sampleRate": sample_rate}


def _prepared_tracks_from_payload(
    payload: Mapping[str, Any],
) -> list[tuple[str, np.ndarray, str, float | None]]:
    """Decode payload tracks for the native fallback without changing levels."""

    prepared: list[tuple[str, np.ndarray, str, float | None]] = []
    for track in payload["tracks"]:
        encoded = str(track["src"]).partition(",")[2]
        with wave.open(io.BytesIO(base64.b64decode(encoded)), "rb") as wav_file:
            pcm = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
        signal = pcm.astype(np.float32) / 32767.0
        prepared.append(
            (
                str(track["label"]),
                signal,
                str(track["description"]),
                track["levelDelta"],
            )
        )
    return prepared


def _render_native_fallback(
    prepared_tracks: list[tuple[str, np.ndarray, str, float | None]],
    sample_rate: int,
) -> None:
    st.caption("Play one version at a time and use the same position to compare them.")
    for label, signal, description, delta in prepared_tracks:
        label_column, player_column = st.columns([1.15, 2.85])
        with label_column:
            st.write(label)
            st.caption(description)
            kind = _track_kind(label)
            if kind == "original":
                st.caption("Reference level")
            elif delta is not None:
                qualifier = "quieter" if delta < 0 else "louder"
                if abs(delta) < 0.05:
                    st.caption("Same level as original")
                else:
                    st.caption(f"{abs(delta):.1f} dB {qualifier}")
        with player_column:
            st.audio(signal, sample_rate=sample_rate, format="audio/wav")


def render_audio_comparison(
    signals: dict[str, np.ndarray],
    sample_rate: int,
    level_deltas_db: dict[str, float] | None = None,
    descriptions: dict[str, str] | None = None,
    *,
    key: str = "metavision-audio-comparison",
) -> None:
    """Render one synchronized, keyboard-accessible audio comparison player.

    Track insertion order is preserved. Put ``Original`` first, followed by the
    primary analytical and FEM results; an ``Infinite`` result can be included
    anywhere in the ordered mapping. Labels are used only as text and all audio
    is passed through finite-value normalization before PCM encoding.

    Parameters
    ----------
    signals:
        Ordered mapping of visible track labels to one-dimensional mono arrays.
    sample_rate:
        Shared sample rate in hertz.
    level_deltas_db:
        Optional mapping of track label to dB difference relative to Original.
        Negative values are presented as quieter.
    descriptions:
        Optional mapping of track label to a short plain-language description.
    key:
        Optional Streamlit component key when more than one player is rendered.

    """

    payload = build_audio_comparison_payload(
        signals,
        sample_rate,
        level_deltas_db,
        descriptions,
    )
    if _AUDIO_COMPARE_COMPONENT is None:
        prepared_tracks = _prepared_tracks_from_payload(payload)
        _render_native_fallback(prepared_tracks, int(sample_rate))
        return

    _AUDIO_COMPARE_COMPONENT(
        data=payload,
        key=key,
        width="stretch",
        height="content",
    )


__all__ = ["build_audio_comparison_payload", "render_audio_comparison"]
