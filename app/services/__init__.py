# Creator: Sulabh Bansod
# Description: Initializes the services package.
# Use: Exposes core service singletons for application use.

"""Application service layer."""

from app.services.opik_validation_service import OPIKValidationService
from app.services.tts_service import TTSService, tts_service

__all__ = ["OPIKValidationService", "TTSService", "tts_service"]

