"""
Voice command interface for locomotion control.

Port of scripts/house_scene/voice_commander.py (M6 precursor: the human side
of the loop, feeding velocity commands to whatever controller is running):

    Microphone → sounddevice (3 s chunks)
    → Whisper (local) → text
    → Gemini 2.5 Flash (parse NL → vx, vy, vyaw)
    → CommandState (thread-safe)
    → control loop reads CommandState each step

``VoiceCommander`` owns the background thread; ``CommandState`` is the only
object shared with the control loop. The stages are also usable on their own
(``record_audio``, ``transcribe``, ``parse_with_gemini``), e.g. for a text-only
or pre-recorded pipeline.

Dependencies (lazy): pip install openai-whisper sounddevice google-genai
Setup: export GEMINI_API_KEY=<key from aistudio.google.com/app/apikey>
"""

from __future__ import annotations

import json
import os
import threading
import time

import numpy as np

__all__ = ["CommandState", "VoiceCommander", "parse_with_gemini",
           "record_audio", "transcribe"]

# Gemini model and decoding settings for command parsing: near-deterministic
# and short — the reply is a ~5-field JSON object.
GEMINI_MODEL = "gemini-2.5-flash"
_PARSE_TEMPERATURE = 0.1
_PARSE_MAX_TOKENS = 200

# Whisper transcripts shorter than this are noise ("Uh", punctuation).
_MIN_TRANSCRIPT_CHARS = 3

# Pause after an unexpected error so a persistent fault does not spin the
# thread at full speed.
_ERROR_BACKOFF_S = 0.5


# ==========================================================================
#  Command state — shared between voice thread and control loop
# ==========================================================================

class CommandState:
    """Thread-safe (vx, vy, vyaw) velocity command with change tracking.

    Values are clipped to the class bounds on ``set`` so a mis-parsed
    command can never exceed what the locomotion policy was trained for.
    """
    VX_MIN, VX_MAX = -1.0, 1.0
    VY_MIN, VY_MAX = -0.5, 0.5
    VYW_MIN, VYW_MAX = -1.0, 1.0

    def __init__(self, vx: float = 0.5, vy: float = 0.0, vyaw: float = 0.0):
        self._lock = threading.Lock()
        self._vx = float(vx)
        self._vy = float(vy)
        self._vyaw = float(vyaw)
        self._changed = False

    def set(self, vx: float, vy: float, vyaw: float):
        """Store a new command (clipped) and mark the state as changed."""
        vx = float(np.clip(vx, self.VX_MIN, self.VX_MAX))
        vy = float(np.clip(vy, self.VY_MIN, self.VY_MAX))
        vyaw = float(np.clip(vyaw, self.VYW_MIN, self.VYW_MAX))
        with self._lock:
            self._vx, self._vy, self._vyaw = vx, vy, vyaw
            self._changed = True

    def get(self) -> tuple:
        """Current ``(vx, vy, vyaw)``."""
        with self._lock:
            return (self._vx, self._vy, self._vyaw)

    def pop_changed(self) -> bool:
        """Returns True once if state changed since last call."""
        with self._lock:
            c = self._changed
            self._changed = False
            return c


# ==========================================================================
#  Gemini command parser
# ==========================================================================

SYSTEM_PROMPT = """You are a locomotion command interpreter for a quadruped robot (Go2).
Convert natural language instructions into robot velocity commands.

Output ONLY a valid JSON object with these exact fields — no markdown, no explanation:
{
  "vx":          float,   // forward velocity m/s, range [-1.0, 1.0]
  "vy":          float,   // lateral velocity m/s, range [-0.5, 0.5], positive=left
  "vyaw":        float,   // yaw rate rad/s, range [-1.0, 1.0], positive=turn left
  "description": string   // one short phrase
}

Speed mappings:
  "stop" / "halt"          → vx=0 vy=0 vyaw=0
  "slow" / "slowly"        → multiply speed by 0.3
  "medium" / default       → multiply speed by 0.6
  "fast" / "quickly"       → multiply speed by 1.0
  "faster"                 → add 0.2 to current vx
  "slower"                 → subtract 0.2 from current vx

Direction:
  "forward" / "straight"   → vx>0, vy=0, vyaw=0
  "backward" / "reverse"   → vx<0, vy=0, vyaw=0
  "left" (lateral)         → vy>0
  "right" (lateral)        → vy<0
  "turn left"              → vyaw>0
  "turn right"             → vyaw<0
  "spin left"              → vx=0, vyaw=0.8
  "spin right"             → vx=0, vyaw=-0.8

If a specific speed in m/s is mentioned, use it directly (clamped to range).
If the command is unclear or unrelated to locomotion, keep current values and
set description to "no change"."""


def parse_with_gemini(text: str, current: tuple, api_key: str | None = None) -> tuple:
    """Parse a natural-language command. Returns (vx, vy, vyaw, description).

    The current command is sent along so relative instructions ("faster")
    resolve; unparseable/irrelevant speech yields description "no change".

    Raises:
        ImportError: google-genai not installed.
        ValueError: no API key.
        json.JSONDecodeError: the model did not return JSON.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai not installed — pip install google-genai") from None

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("No Gemini API key. Set GEMINI_API_KEY or pass api_key=.")

    client = genai.Client(api_key=key)
    user_msg = (f"Current command: vx={current[0]:.2f}, "
                f"vy={current[1]:.2f}, vyaw={current[2]:.2f}\n"
                f'User said: "{text}"')

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Content(
            role="user",
            parts=[types.Part(text=SYSTEM_PROMPT + "\n\n" + user_msg)])],
        config=types.GenerateContentConfig(
            temperature=_PARSE_TEMPERATURE, max_output_tokens=_PARSE_MAX_TOKENS),
    )

    # The prompt forbids markdown, but the model still fences JSON sometimes.
    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)
    return (float(data.get("vx", current[0])),
            float(data.get("vy", current[1])),
            float(data.get("vyaw", current[2])),
            str(data.get("description", "")))


# ==========================================================================
#  Audio recording + Whisper transcription
# ==========================================================================

def record_audio(duration: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """Blocking mono capture from the default microphone (float32, 16 kHz).

    Raises:
        ImportError: sounddevice not installed.
    """
    try:
        import sounddevice as sd
    except ImportError:
        raise ImportError(
            "sounddevice not installed — pip install sounddevice") from None

    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                   channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def transcribe(audio: np.ndarray, model) -> str:
    """Run a loaded Whisper model on a 16 kHz float32 buffer; returns text."""
    # fp16=False: Whisper warns and falls back on CPU anyway; be explicit.
    result = model.transcribe(audio, fp16=False)
    return result["text"].strip()


# ==========================================================================
#  Voice Commander — background thread
# ==========================================================================

class VoiceCommander:
    """
    Records audio in chunks, transcribes with Whisper, parses with Gemini,
    updates CommandState. Daemon thread — dies with the main program.

    Args:
        state: the shared command the control loop reads.
        whisper_model: Whisper size ("tiny" keeps latency low on CPU).
        chunk_duration: seconds of audio per recognition attempt.
        silence_threshold: RMS below which a chunk is skipped unheard.
        api_key: Gemini key (defaults to ``$GEMINI_API_KEY``).
        verbose: print heard text and resulting commands.
    """

    def __init__(self, state: CommandState, whisper_model: str = "tiny",
                 chunk_duration: float = 3.0, silence_threshold: float = 0.01,
                 api_key: str | None = None, verbose: bool = True):
        self.state = state
        self.whisper_model_name = whisper_model
        self.chunk_duration = chunk_duration
        self.silence_threshold = silence_threshold
        self.api_key = api_key
        self.verbose = verbose
        self._stop = threading.Event()
        self._thread = None
        self._whisper = None

    def _load_whisper(self):
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "openai-whisper not installed — pip install openai-whisper") from None
        self._say(f"[voice] Loading Whisper '{self.whisper_model_name}'...")
        self._whisper = whisper.load_model(self.whisper_model_name)
        self._say("[voice] Whisper ready.")

    def _loop(self):
        self._load_whisper()
        self._say(f"\n[voice] Listening (chunk={self.chunk_duration}s, "
                  f"model={self.whisper_model_name})")
        self._say("[voice] Examples: 'go forward at 0.5 metres per second', "
                  "'turn left slowly', 'stop'\n")

        while not self._stop.is_set():
            try:
                audio = record_audio(self.chunk_duration)

                rms = float(np.sqrt(np.mean(audio ** 2)))
                if rms < self.silence_threshold:
                    continue

                text = transcribe(audio, self._whisper)
                if not text or len(text.strip()) < _MIN_TRANSCRIPT_CHARS:
                    continue
                self._say(f'[voice] Heard: "{text}"')

                current = self.state.get()
                vx, vy, vyaw, desc = parse_with_gemini(text, current, self.api_key)
                if desc == "no change":
                    self._say("[voice] No change.")
                    continue

                self.state.set(vx, vy, vyaw)
                self._say(f"[voice] → {desc}")
                self._say(f"        vx={vx:.2f}  vy={vy:.2f}  vyaw={vyaw:.2f}")

            except KeyboardInterrupt:
                break
            except json.JSONDecodeError as e:
                self._say(f"[voice] JSON parse error: {e}")
            except Exception as e:
                # The listener must outlive transient mic/API/network faults;
                # report and keep listening rather than kill the thread.
                self._say(f"[voice] Error: {e}")
                time.sleep(_ERROR_BACKOFF_S)

    def start(self):
        """Start the daemon listener thread (Whisper loads on that thread)."""
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the loop to exit and wait briefly for it."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _say(self, msg: str):
        if self.verbose:
            print(f"  {msg}")
