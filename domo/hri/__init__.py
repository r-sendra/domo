"""
Human-robot interaction layer (M6 precursor): voice command interface.
Heavy dependencies (whisper, sounddevice, google-genai) are imported lazily —
the rest of the library never needs them.

    from domo.hri import CommandState, VoiceCommander
    state = CommandState()
    VoiceCommander(state).start()        # background thread
    vx, vy, vyaw = state.get()           # inside the control loop
"""

from .voice import CommandState, VoiceCommander, parse_with_gemini

__all__ = ["CommandState", "VoiceCommander", "parse_with_gemini"]
