# Creator: Sulabh Bansod
# Description: Initializes the services package.
# Use: Exposes core service singletons for application use.

"""Application service layer."""

from app.services.tts_service import TTSService, tts_service

__all__ = ["TTSService", "tts_service"]

