"""
Human-robot interaction layer (M6 precursor): voice command interface.
Heavy dependencies (whisper, sounddevice, google-genai) are imported lazily —
the rest of the library never needs them.
"""

from .voice import CommandState, VoiceCommander, parse_with_gemini

__all__ = ["CommandState", "VoiceCommander", "parse_with_gemini"]
