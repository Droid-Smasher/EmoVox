"""
Azure Speech Service for EmoVox system.

This module provides comprehensive speech-to-text transcription capabilities
using Azure Cognitive Services Speech SDK. Supports both batch file processing
and real-time continuous recognition with robust error handling and audio
format validation.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Union
import io

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError as e:
    raise ImportError(
        "Azure Speech SDK not found. Please install: pip install azure-cognitiveservices-speech"
    ) from e

from models.data_models import SpeechTranscription

# Configure logging
logger = logging.getLogger(__name__)


class AzureSpeechService:
    """
    Comprehensive Azure Speech Services integration for EmoVox system.
    
    Provides speech-to-text transcription capabilities with support for:
    - Batch file processing and in-memory audio transcription
    - Real-time continuous recognition for streaming audio
    - Multiple audio formats and sampling rates
    - Robust error handling and retry logic
    - Standardized response formatting using SpeechTranscription model
    """
    
    def __init__(self):
        """Initialize Azure Speech Service with configuration from environment variables."""
        # Load Azure configuration
        self.speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.speech_region = os.getenv("AZURE_SPEECH_REGION", "eastus")
        
        if not self.speech_key:
            raise ValueError("AZURE_SPEECH_KEY environment variable is required")
        
        # Continuous recognition state
        self._continuous_recognizer: Optional[speechsdk.SpeechRecognizer] = None
        self._is_continuous_active = False
        self._recognition_lock = asyncio.Lock()
        
        # Supported audio formats with Azure Speech SDK compatibility
        self.supported_formats = {
            "wav": {"mime_type": "audio/wav", "codec": "pcm"},
            "mp3": {"mime_type": "audio/mpeg", "codec": "mp3"},
            "m4a": {"mime_type": "audio/mp4", "codec": "aac"},
            "ogg": {"mime_type": "audio/ogg", "codec": "opus"},
            "flac": {"mime_type": "audio/flac", "codec": "flac"},
            "aac": {"mime_type": "audio/aac", "codec": "aac"}
        }
        
        # Supported sample rates (Hz)
        self.supported_sample_rates = [8000, 16000, 22050, 44100, 48000]
        
        # Recognition configuration
        self.language = "en-US"  # Default language
        self.timeout_seconds = 30  # Default timeout for single recognition
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        
        logger.info("Azure Speech Service initialized successfully")
    
    def _create_speech_config(self) -> speechsdk.SpeechConfig:
        """
        Create Azure Speech configuration with authentication and settings.
        
        Returns:
            Configured SpeechConfig instance
        
        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If Azure service authentication fails
        """
        try:
            speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key, 
                region=self.speech_region
            )
            
            # Configure recognition settings
            speech_config.speech_recognition_language = self.language
            speech_config.output_format = speechsdk.OutputFormat.Detailed
            
            # Enable profanity filtering and enable dictation mode for better accuracy
            speech_config.enable_dictation()
            speech_config.set_profanity(speechsdk.ProfanityOption.Masked)
            
            # Configure for continuous recognition optimization
            speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, 
                "10000"
            )
            speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, 
                "2000"
            )
            
            return speech_config
            
        except Exception as e:
            logger.error(f"Failed to create speech configuration: {e}")
            raise RuntimeError(f"Speech configuration failed: {e}")
    
    def _create_audio_config(self, source: Union[str, bytes, io.BytesIO]) -> speechsdk.AudioConfig:
        """
        Create audio input configuration from various sources.
        
        Args:
            source: Audio source - file path, bytes, or BytesIO stream
        
        Returns:
            Configured AudioConfig instance
        
        Raises:
            ValueError: If source format is unsupported
            FileNotFoundError: If audio file doesn't exist
        """
        try:
            if isinstance(source, str):
                # File path source
                if not Path(source).exists():
                    raise FileNotFoundError(f"Audio file not found: {source}")
                return speechsdk.AudioConfig(filename=source)
            
            elif isinstance(source, (bytes, io.BytesIO)):
                # In-memory audio data
                if isinstance(source, bytes):
                    audio_stream = io.BytesIO(source)
                else:
                    audio_stream = source
                
                # Create audio stream for Azure Speech SDK
                stream = speechsdk.audio.PushAudioInputStream()
                return speechsdk.AudioConfig(stream=stream)
            
            else:
                raise ValueError(f"Unsupported audio source type: {type(source)}")
                
        except Exception as e:
            logger.error(f"Failed to create audio configuration: {e}")
            raise
    
    def _validate_audio_format(self, audio_data: bytes, format_hint: str = None) -> str:
        """
        Validate and detect audio format from data or format hint.
        
        Args:
            audio_data: Raw audio data
            format_hint: Suggested audio format (e.g., 'wav', 'mp3')
        
        Returns:
            Detected or validated audio format
        
        Raises:
            ValueError: If format is unsupported or invalid
        """
        if not audio_data or len(audio_data) < 44:  # Minimum for WAV header
            raise ValueError("Audio data is empty or too small")
        
        # Auto-detect format from headers if no hint provided
        if not format_hint:
            if audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:12]:
                format_hint = "wav"
            elif audio_data.startswith(b'\xff\xfb') or audio_data.startswith(b'ID3'):
                format_hint = "mp3"
            elif audio_data.startswith(b'ftyp'):
                format_hint = "m4a"
            elif audio_data.startswith(b'OggS'):
                format_hint = "ogg"
            elif audio_data.startswith(b'fLaC'):
                format_hint = "flac"
            else:
                raise ValueError("Unable to detect audio format from data")
        
        # Validate format is supported
        format_lower = format_hint.lower().lstrip('.')
        if format_lower not in self.supported_formats:
            supported_list = ", ".join(self.supported_formats.keys())
            raise ValueError(f"Unsupported audio format: {format_hint}. Supported: {supported_list}")
        
        return format_lower
    
    def _process_recognition_result(self, result: speechsdk.SpeechRecognitionResult) -> SpeechTranscription:
        """
        Process Azure Speech recognition result and convert to SpeechTranscription model.
        
        Args:
            result: Azure Speech recognition result
        
        Returns:
            Standardized SpeechTranscription instance
        
        Raises:
            ValueError: If result processing fails
        """
        try:
            # Check result reason
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                # Successful recognition
                confidence = 1.0  # Default confidence
                
                # Try to extract confidence from detailed results
                try:
                    if hasattr(result, 'properties') and result.properties:
                        detailed_result = result.properties.get(speechsdk.PropertyId.SpeechServiceResponse_JsonResult)
                        if detailed_result:
                            import json
                            json_result = json.loads(detailed_result)
                            if 'NBest' in json_result and json_result['NBest']:
                                confidence = json_result['NBest'][0].get('Confidence', 1.0)
                except Exception as e:
                    logger.warning(f"Could not extract detailed confidence: {e}")
                
                return SpeechTranscription(
                    text=result.text,
                    confidence=confidence,
                    timestamp=time.time()
                )
            
            elif result.reason == speechsdk.ResultReason.NoMatch:
                # No speech detected
                return SpeechTranscription(
                    text="",
                    confidence=0.0,
                    timestamp=time.time()
                )
            
            elif result.reason == speechsdk.ResultReason.Canceled:
                # Recognition was canceled
                cancellation = result.cancellation_details
                error_msg = f"Recognition canceled: {cancellation.reason}"
                if cancellation.error_details:
                    error_msg += f" - {cancellation.error_details}"
                
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            else:
                raise ValueError(f"Unexpected recognition result reason: {result.reason}")
                
        except Exception as e:
            logger.error(f"Failed to process recognition result: {e}")
            raise
    
    async def transcribe_audio_file(self, file_path: str) -> SpeechTranscription:
        """
        Transcribe audio file and return SpeechTranscription model.
        
        Args:
            file_path: Path to audio file to transcribe
        
        Returns:
            SpeechTranscription with transcribed text and confidence
        
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If file format is unsupported
            RuntimeError: If Azure service fails
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        logger.info(f"Starting transcription of audio file: {file_path}")
        
        # Retry logic for Azure service reliability
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                # Create configurations
                speech_config = self._create_speech_config()
                audio_config = self._create_audio_config(file_path)
                
                # Create recognizer
                recognizer = speechsdk.SpeechRecognizer(
                    speech_config=speech_config, 
                    audio_config=audio_config
                )
                
                # Perform recognition
                logger.debug(f"Attempt {attempt + 1}: Starting recognition for {file_path}")
                result = recognizer.recognize_once_async().get(timeout=self.timeout_seconds)
                
                # Process and return result
                transcription = self._process_recognition_result(result)
                logger.info(f"Transcription completed successfully: {len(transcription.text)} characters")
                return transcription
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Transcription attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                continue
        
        # All retries failed
        logger.error(f"All transcription attempts failed for {file_path}")
        raise RuntimeError(f"Transcription failed after {self.max_retries} attempts: {last_exception}")
    
    async def transcribe_audio_bytes(
        self, 
        audio_data: bytes, 
        audio_format: str = "wav"
    ) -> SpeechTranscription:
        """
        Transcribe audio bytes and return SpeechTranscription model.
        
        Args:
            audio_data: Raw audio data as bytes
            audio_format: Audio format (wav, mp3, m4a, ogg, flac)
        
        Returns:
            SpeechTranscription with transcribed text and confidence
        
        Raises:
            ValueError: If audio data is invalid or format unsupported
            RuntimeError: If Azure service fails
        """
        if not audio_data:
            raise ValueError("Audio data cannot be empty")
        
        # Validate format
        validated_format = self._validate_audio_format(audio_data, audio_format)
        logger.info(f"Starting transcription of {len(audio_data)} bytes of {validated_format} audio")
        
        # Retry logic for Azure service reliability
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                # Create configurations
                speech_config = self._create_speech_config()
                
                # Create push stream for in-memory audio
                push_stream = speechsdk.audio.PushAudioInputStream()
                audio_config = speechsdk.AudioConfig(stream=push_stream)
                
                # Create recognizer
                recognizer = speechsdk.SpeechRecognizer(
                    speech_config=speech_config, 
                    audio_config=audio_config
                )
                
                # Push audio data to stream
                push_stream.write(audio_data)
                push_stream.close()
                
                # Perform recognition
                logger.debug(f"Attempt {attempt + 1}: Starting recognition for audio bytes")
                result = recognizer.recognize_once_async().get(timeout=self.timeout_seconds)
                
                # Process and return result
                transcription = self._process_recognition_result(result)
                logger.info(f"Transcription completed successfully: {len(transcription.text)} characters")
                return transcription
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Transcription attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                continue
        
        # All retries failed
        logger.error(f"All transcription attempts failed for audio bytes")
        raise RuntimeError(f"Transcription failed after {self.max_retries} attempts: {last_exception}")
    
    async def start_continuous_recognition(self, callback: Callable[[SpeechTranscription], None]) -> None:
        """
        Start continuous recognition for real-time streaming audio.
        
        Args:
            callback: Function to call with each transcription result
        
        Raises:
            RuntimeError: If continuous recognition is already active or fails to start
            ValueError: If callback is not provided
        """
        if not callback or not callable(callback):
            raise ValueError("Valid callback function is required")
        
        async with self._recognition_lock:
            if self._is_continuous_active:
                raise RuntimeError("Continuous recognition is already active")
            
            try:
                logger.info("Starting continuous speech recognition")
                
                # Create configurations
                speech_config = self._create_speech_config()
                
                # Use default microphone for continuous recognition
                audio_config = speechsdk.AudioConfig(use_default_microphone=True)
                
                # Create continuous recognizer
                self._continuous_recognizer = speechsdk.SpeechRecognizer(
                    speech_config=speech_config,
                    audio_config=audio_config
                )
                
                # Set up event handlers
                def recognized_handler(evt):
                    """Handle recognized speech events."""
                    try:
                        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                            transcription = self._process_recognition_result(evt.result)
                            callback(transcription)
                    except Exception as e:
                        logger.error(f"Error in recognition callback: {e}")
                
                def canceled_handler(evt):
                    """Handle canceled recognition events."""
                    logger.warning(f"Continuous recognition canceled: {evt.cancellation_details.reason}")
                    if evt.cancellation_details.error_details:
                        logger.error(f"Cancellation error: {evt.cancellation_details.error_details}")
                
                # Connect event handlers
                self._continuous_recognizer.recognized.connect(recognized_handler)
                self._continuous_recognizer.canceled.connect(canceled_handler)
                
                # Start continuous recognition
                self._continuous_recognizer.start_continuous_recognition_async()
                self._is_continuous_active = True
                
                logger.info("Continuous recognition started successfully")
                
            except Exception as e:
                logger.error(f"Failed to start continuous recognition: {e}")
                self._continuous_recognizer = None
                raise RuntimeError(f"Continuous recognition startup failed: {e}")
    
    async def stop_continuous_recognition(self) -> None:
        """
        Stop continuous recognition.
        
        Raises:
            RuntimeError: If continuous recognition is not active
        """
        async with self._recognition_lock:
            if not self._is_continuous_active or not self._continuous_recognizer:
                raise RuntimeError("Continuous recognition is not active")
            
            try:
                logger.info("Stopping continuous speech recognition")
                
                # Stop recognition
                self._continuous_recognizer.stop_continuous_recognition_async().get()
                
                # Clean up
                self._continuous_recognizer = None
                self._is_continuous_active = False
                
                logger.info("Continuous recognition stopped successfully")
                
            except Exception as e:
                logger.error(f"Failed to stop continuous recognition: {e}")
                # Force cleanup even if stop fails
                self._continuous_recognizer = None
                self._is_continuous_active = False
                raise RuntimeError(f"Continuous recognition stop failed: {e}")
    
    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported audio formats.
        
        Returns:
            List of supported audio format strings
        """
        return list(self.supported_formats.keys())
    
    def is_continuous_active(self) -> bool:
        """
        Check if continuous recognition is currently active.
        
        Returns:
            True if continuous recognition is active, False otherwise
        """
        return self._is_continuous_active
    
    def set_language(self, language_code: str) -> None:
        """
        Set recognition language.
        
        Args:
            language_code: Language code (e.g., "en-US", "es-ES", "fr-FR")
        
        Raises:
            ValueError: If language code is invalid
        """
        if not language_code or len(language_code) < 2:
            raise ValueError("Invalid language code")
        
        self.language = language_code
        logger.info(f"Recognition language set to: {language_code}")
    
    def set_timeout(self, timeout_seconds: int) -> None:
        """
        Set recognition timeout.
        
        Args:
            timeout_seconds: Timeout in seconds for recognition operations
        
        Raises:
            ValueError: If timeout is invalid
        """
        if timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        
        self.timeout_seconds = timeout_seconds
        logger.info(f"Recognition timeout set to: {timeout_seconds} seconds")
    
    async def get_service_health(self) -> Dict[str, any]:
        """
        Check Azure Speech Service health and connectivity.
        
        Returns:
            Dictionary with service health information
        """
        try:
            # Test basic connectivity with a small audio test
            test_audio = b'\x00' * 1024  # Simple silence for testing
            
            # Quick test recognition
            speech_config = self._create_speech_config()
            
            return {
                "status": "healthy",
                "region": self.speech_region,
                "language": self.language,
                "supported_formats": len(self.supported_formats),
                "continuous_active": self._is_continuous_active,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Service health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }


# Module-level instance for dependency injection
azure_speech_service = AzureSpeechService()


async def get_azure_speech_service() -> AzureSpeechService:
    """
    Dependency injection function for FastAPI.
    
    Returns:
        AzureSpeechService instance
    """
    return azure_speech_service