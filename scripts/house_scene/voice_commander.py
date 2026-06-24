"""
voice_commander.py
------------------
Voice command interface for Go2 locomotion evaluation.

Pipeline:
    Microphone → sounddevice (3s chunk)
    → Whisper tiny (local, ~0.5s latency)
    → Gemini 2.5 Flash free tier (parse NL → vx, vy, vyaw)
    → CommandState (thread-safe)
    → eval loop reads CommandState each step

Dependencies:
    pip install openai-whisper sounddevice numpy google-genai

Setup:
    1. Get a free API key at aistudio.google.com/app/apikey
    2. export GEMINI_API_KEY=your_key_here

Usage:
    from voice_commander import VoiceCommander, CommandState
    state = CommandState(vx=0.5)
    vc    = VoiceCommander(state)
    vc.start()
    # in eval loop:
    vx, vy, vyaw = state.get()
    vc.stop()

Standalone test:
    python voice_commander.py
"""

import os
import threading
import time
import json
import numpy as np


# ==========================================================================
#  Command state — thread-safe shared state between voice thread and sim loop
# ==========================================================================

class CommandState:
    VX_MIN,  VX_MAX  = -1.0,  1.0
    VY_MIN,  VY_MAX  = -0.5,  0.5
    VYW_MIN, VYW_MAX = -1.0,  1.0

    def __init__(self, vx: float = 0.5, vy: float = 0.0, vyaw: float = 0.0):
        self._lock    = threading.Lock()
        self._vx      = float(vx)
        self._vy      = float(vy)
        self._vyaw    = float(vyaw)
        self._changed = False

    def set(self, vx: float, vy: float, vyaw: float):
        vx   = float(np.clip(vx,   self.VX_MIN,  self.VX_MAX))
        vy   = float(np.clip(vy,   self.VY_MIN,  self.VY_MAX))
        vyaw = float(np.clip(vyaw, self.VYW_MIN, self.VYW_MAX))
        with self._lock:
            self._vx      = vx
            self._vy      = vy
            self._vyaw    = vyaw
            self._changed = True

    def get(self) -> tuple:
        with self._lock:
            return (self._vx, self._vy, self._vyaw)

    def pop_changed(self) -> bool:
        """Returns True once if state changed since last call."""
        with self._lock:
            c = self._changed
            self._changed = False
            return c


# ==========================================================================
#  Gemini command parser (free tier)
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


def parse_with_gemini(
    text:    str,
    current: tuple,
    api_key: str = None,
) -> tuple:
    """
    Parse natural language command using Gemini 2.5 Flash (free tier).
    Returns (vx, vy, vyaw, description).
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai not installed — pip install google-genai"
        )

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "No Gemini API key. Set GEMINI_API_KEY env var or pass api_key=."
        )

    client = genai.Client(api_key=key)

    user_msg = (
        f"Current command: vx={current[0]:.2f}, "
        f"vy={current[1]:.2f}, vyaw={current[2]:.2f}\n"
        f"User said: \"{text}\""
    )

    response = client.models.generate_content(
        model    = "gemini-2.5-flash",
        contents = [
            types.Content(
                role  = "user",
                parts = [types.Part(text=SYSTEM_PROMPT + "\n\n" + user_msg)],
            )
        ],
        config = types.GenerateContentConfig(
            temperature      = 0.1,   # low temp for consistent JSON
            max_output_tokens = 200,
        ),
    )

    raw  = response.text.strip()
    # Strip markdown fences if Gemini adds them
    raw  = raw.replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)

    vx   = float(data.get("vx",   current[0]))
    vy   = float(data.get("vy",   current[1]))
    vyaw = float(data.get("vyaw", current[2]))
    desc = str(data.get("description", ""))

    return (vx, vy, vyaw, desc)


# ==========================================================================
#  Audio recording + Whisper transcription
# ==========================================================================

def record_audio(duration: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """Record `duration` seconds from the default microphone."""
    try:
        import sounddevice as sd
    except ImportError:
        raise ImportError("sounddevice not installed — pip install sounddevice")

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate = sample_rate,
        channels   = 1,
        dtype      = "float32",
    )
    sd.wait()
    return audio.flatten()


def transcribe(audio: np.ndarray, model) -> str:
    """Transcribe float32 numpy audio array using a loaded Whisper model."""
    result = model.transcribe(audio, fp16=False)
    return result["text"].strip()


# ==========================================================================
#  Voice Commander — background thread
# ==========================================================================

class VoiceCommander:
    """
    Records audio in 3-second chunks, transcribes with Whisper,
    parses with Gemini, and updates CommandState.
    Runs as a daemon thread — stops automatically when the main program exits.
    """

    def __init__(
        self,
        state:              CommandState,
        whisper_model:      str   = "tiny",   # tiny/base/small
        chunk_duration:     float = 3.0,
        silence_threshold:  float = 0.01,
        api_key:            str   = None,
        verbose:            bool  = True,
    ):
        self.state             = state
        self.whisper_model_name = whisper_model
        self.chunk_duration    = chunk_duration
        self.silence_threshold = silence_threshold
        self.api_key           = api_key
        self.verbose           = verbose
        self._stop             = threading.Event()
        self._thread           = None
        self._whisper          = None

    def _load_whisper(self):
        try:
            import whisper
            if self.verbose:
                print(f"  [voice] Loading Whisper '{self.whisper_model_name}'...")
            self._whisper = whisper.load_model(self.whisper_model_name)
            if self.verbose:
                print(f"  [voice] Whisper ready.")
        except ImportError:
            raise ImportError(
                "openai-whisper not installed — pip install openai-whisper"
            )

    def _loop(self):
        self._load_whisper()

        if self.verbose:
            print(f"\n  [voice] Listening  (chunk={self.chunk_duration}s, "
                  f"model={self.whisper_model_name})")
            print(f"  [voice] Examples:")
            print(f"            'go forward at 0.5 metres per second'")
            print(f"            'turn left slowly'")
            print(f"            'stop'\n")

        while not self._stop.is_set():
            try:
                audio = record_audio(self.chunk_duration)

                # Skip silence
                rms = float(np.sqrt(np.mean(audio ** 2)))
                if rms < self.silence_threshold:
                    continue

                text = transcribe(audio, self._whisper)
                if not text or len(text.strip()) < 3:
                    continue

                if self.verbose:
                    print(f"  [voice] Heard: \"{text}\"")

                current = self.state.get()
                vx, vy, vyaw, desc = parse_with_gemini(
                    text, current, self.api_key
                )

                if desc == "no change":
                    if self.verbose:
                        print(f"  [voice] No change.")
                    continue

                self.state.set(vx, vy, vyaw)
                if self.verbose:
                    print(f"  [voice] → {desc}")
                    print(f"          vx={vx:.2f}  vy={vy:.2f}  vyaw={vyaw:.2f}")

            except KeyboardInterrupt:
                break
            except json.JSONDecodeError as e:
                if self.verbose:
                    print(f"  [voice] JSON parse error: {e}")
            except Exception as e:
                if self.verbose:
                    print(f"  [voice] Error: {e}")
                time.sleep(0.5)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)


# ==========================================================================
#  Standalone test
# ==========================================================================

if __name__ == "__main__":
    print("Voice Commander — standalone test")
    print("Requires: GEMINI_API_KEY environment variable")
    print("Say locomotion commands. Ctrl+C to exit.\n")

    state = CommandState(vx=0.5)
    vc    = VoiceCommander(state, verbose=True)
    vc.start()

    try:
        while True:
            time.sleep(2.0)
            vx, vy, vyaw = state.get()
            print(f"  State: vx={vx:.2f}  vy={vy:.2f}  vyaw={vyaw:.2f}")
    except KeyboardInterrupt:
        print("\nStopping.")
        vc.stop()
