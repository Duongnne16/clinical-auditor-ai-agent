class GeminiService:
    """Disabled-by-default boundary for a future Gemini integration."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def generate(self, prompt: str) -> str:
        if not self.enabled:
            return "Gemini integration is disabled."
        raise NotImplementedError("Live Gemini calls are not implemented yet.")
