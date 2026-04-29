"""
alarm.py
--------
Escalating audio alarm system for driver drowsiness detection.

Three alarm levels fire automatically based on how many consecutive drowsy
frames the detector has seen. Each level is louder and more urgent than
the last so the driver gets a progressive warning rather than a sudden shock.

Alarm levels
------------
Level 0 — NONE     : < 15 consecutive drowsy frames  → silent
Level 1 — WARNING  : 15–29 frames  (~0.5 s at 30fps) → single soft beep every 1.2 s
Level 2 — ALERT    : 30–44 frames  (~1.0 s at 30fps) → rapid double-beep every 0.6 s
Level 3 — CRITICAL : 45+ frames    (~1.5 s at 30fps) → continuous rising siren

Audio backend (tried in order)
-------------------------------
1. pygame  — cross-platform, tones generated in-process with numpy, no WAV file needed
2. winsound — Windows-only built-in, no extra install
3. terminal bell — last resort (often silent in modern terminals)

All audio runs in a daemon thread so it never blocks the video loop.
The alarm stops immediately when the driver becomes alert again.

Usage
-----
    from alarm import AlarmSystem

    alarm = AlarmSystem()
    alarm.start()                          # launch background thread

    # inside your frame loop:
    alarm.update(consecutive_drowsy)       # pass current counter every frame

    alarm.stop()                           # on exit
"""

import sys
import time
import threading
import numpy as np

# ── Try to import audio backends ──────────────────────────────────────────────
try:
    import pygame
    import pygame.sndarray
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

try:
    import winsound
    _WINSOUND_AVAILABLE = True
except ImportError:
    _WINSOUND_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100          # Hz — standard CD quality

# Thresholds: number of consecutive drowsy frames to reach each level
FRAMES_WARNING  = 15         # ~0.5 s at 30fps
FRAMES_ALERT    = 30         # ~1.0 s at 30fps
FRAMES_CRITICAL = 45         # ~1.5 s at 30fps

# Alarm level constants
LEVEL_NONE     = 0
LEVEL_WARNING  = 1
LEVEL_ALERT    = 2
LEVEL_CRITICAL = 3

LEVEL_NAMES = {
    LEVEL_NONE:     "NONE",
    LEVEL_WARNING:  "WARNING",
    LEVEL_ALERT:    "ALERT",
    LEVEL_CRITICAL: "CRITICAL",
}

# Visual colours for each level (BGR for OpenCV)
LEVEL_COLORS = {
    LEVEL_NONE:     (50,  200, 50),    # green
    LEVEL_WARNING:  (0,   200, 255),   # yellow
    LEVEL_ALERT:    (0,   140, 255),   # orange
    LEVEL_CRITICAL: (0,   50,  220),   # red
}

# Banner text for each alarm level
LEVEL_BANNERS = {
    LEVEL_NONE:     "",
    LEVEL_WARNING:  "  WARNING: Signs of drowsiness detected",
    LEVEL_ALERT:    "  ALERT: Pull over and rest soon!",
    LEVEL_CRITICAL: "  CRITICAL: STOP THE VEHICLE NOW !!",
}


# ── Tone generation ───────────────────────────────────────────────────────────

def _make_tone(frequency: float, duration_ms: int, volume: float = 0.7) -> np.ndarray:
    """
    Generate a stereo int16 numpy array for a pure sine tone.
    Used to pre-build pygame Sound objects with zero file I/O.
    """
    n_samples = int(SAMPLE_RATE * duration_ms / 1000)
    t         = np.linspace(0, duration_ms / 1000, n_samples, endpoint=False)

    # Sine wave with a short fade-in/out (5 ms) to remove clicks
    fade_samples = min(int(SAMPLE_RATE * 0.005), n_samples // 4)
    envelope     = np.ones(n_samples)
    envelope[:fade_samples]  = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

    mono   = (np.sin(2 * np.pi * frequency * t) * envelope * volume * 32767).astype(np.int16)
    stereo = np.ascontiguousarray(np.column_stack([mono, mono]))
    return stereo


# ── Alarm patterns ────────────────────────────────────────────────────────────
# Each pattern is a list of (freq_hz, tone_ms, pause_ms) tuples.
# The alarm thread loops through the pattern for the current level.

PATTERNS = {
    LEVEL_WARNING: [
        (880,  250, 950),          # one soft beep, long pause
    ],
    LEVEL_ALERT: [
        (1050, 150, 80),           # double-beep —
        (1050, 150, 520),          # — short pause between beeps, longer between pairs
    ],
    LEVEL_CRITICAL: [              # rising siren pattern
        (900,  80,  30),
        (1000, 80,  30),
        (1100, 80,  30),
        (1200, 80,  30),
        (1300, 80,  30),
        (1400, 80,  30),
        (1500, 80,  30),
        (1600, 120, 180),          # peak hold + pause before repeat
    ],
}


# ── AlarmSystem ───────────────────────────────────────────────────────────────

class AlarmSystem:
    """
    Thread-safe, non-blocking escalating alarm.

    Public interface
    ----------------
    start()                 — launch background audio thread
    stop()                  — stop audio thread and silence everything
    update(n_frames)        — call every video frame with current drowsy counter
    mute / unmute           — toggle silence without stopping the thread
    level                   — read-only property: current LEVEL_* int
    """

    def __init__(self):
        self._level     = LEVEL_NONE
        self._muted     = False
        self._running   = False
        self._thread    = None
        self._lock      = threading.Lock()
        self._backend   = self._init_backend()
        self._sounds    = {}         # pygame Sound cache: (freq, dur_ms) -> Sound

        if self._backend == "pygame":
            self._pregenerate_sounds()

        print(f"[AlarmSystem] Audio backend: {self._backend.upper()}")

    # ── Backend init ──────────────────────────────────────────────────────────

    def _init_backend(self) -> str:
        if _PYGAME_AVAILABLE:
            try:
                pygame.mixer.init(
                    frequency = SAMPLE_RATE,
                    size      = -16,       # signed 16-bit
                    channels  = 2,         # stereo
                    buffer    = 512,       # low-latency buffer
                )
                return "pygame"
            except Exception as e:
                print(f"[AlarmSystem] pygame init failed ({e}), trying winsound.")

        if _WINSOUND_AVAILABLE:
            return "winsound"

        return "bell"

    def _pregenerate_sounds(self):
        """Build pygame Sound objects for every tone used in any pattern."""
        seen = set()
        for pattern in PATTERNS.values():
            for freq, dur_ms, _ in pattern:
                key = (freq, dur_ms)
                if key not in seen:
                    seen.add(key)
                    wave              = _make_tone(freq, dur_ms)
                    self._sounds[key] = pygame.sndarray.make_sound(wave)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Launch the background alarm thread. Call once before your video loop."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="AlarmThread")
        self._thread.start()

    def stop(self):
        """Stop the alarm thread and silence audio."""
        self._running = False
        with self._lock:
            self._level = LEVEL_NONE
        if self._backend == "pygame":
            try:
                pygame.mixer.stop()
            except Exception:
                pass

    def update(self, consecutive_frames: int):
        """
        Called every video frame with the current consecutive-drowsy-frame count.
        Automatically updates the alarm level and starts/stops audio.
        """
        new_level = self._level_for(consecutive_frames)
        with self._lock:
            if new_level != self._level:
                self._level = new_level
                if new_level == LEVEL_NONE and self._backend == "pygame":
                    try:
                        pygame.mixer.stop()
                    except Exception:
                        pass

    def mute(self):
        """Silence the alarm without changing the level or stopping the thread."""
        self._muted = True
        if self._backend == "pygame":
            try:
                pygame.mixer.stop()
            except Exception:
                pass

    def unmute(self):
        """Re-enable audio output."""
        self._muted = False

    def toggle_mute(self):
        if self._muted:
            self.unmute()
        else:
            self.mute()

    @property
    def level(self) -> int:
        with self._lock:
            return self._level

    @property
    def is_muted(self) -> bool:
        return self._muted

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _level_for(n: int) -> int:
        if n >= FRAMES_CRITICAL:
            return LEVEL_CRITICAL
        if n >= FRAMES_ALERT:
            return LEVEL_ALERT
        if n >= FRAMES_WARNING:
            return LEVEL_WARNING
        return LEVEL_NONE

    def _play_tone(self, freq: int, dur_ms: int):
        """Play one tone synchronously inside the alarm thread."""
        if self._muted:
            time.sleep(dur_ms / 1000)
            return

        if self._backend == "pygame":
            sound = self._sounds.get((freq, dur_ms))
            if sound:
                sound.play()
                pygame.time.wait(dur_ms)

        elif self._backend == "winsound":
            # winsound.Beep is blocking — that's fine inside our dedicated thread
            try:
                winsound.Beep(max(37, min(freq, 32767)), dur_ms)
            except Exception:
                time.sleep(dur_ms / 1000)

        else:
            print("\a", end="", flush=True)
            time.sleep(dur_ms / 1000)

    def _loop(self):
        """Background thread: play the pattern for the current alarm level."""
        while self._running:
            with self._lock:
                level = self._level

            if level == LEVEL_NONE:
                time.sleep(0.05)
                continue

            pattern = PATTERNS.get(level, [])
            for freq, dur_ms, pause_ms in pattern:
                # Re-check level between tones so we react fast to state changes
                with self._lock:
                    if self._level != level:
                        break
                if not self._running:
                    return

                self._play_tone(freq, dur_ms)
                time.sleep(pause_ms / 1000)
