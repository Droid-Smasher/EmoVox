"""
Comprehensive tests for Azure Speech Service.

This module tests all functionality of the AzureSpeechService including
transcription methods, continuous recognition, error handling, and
audio format validation.
"""

import asyncio
import os
import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from io import BytesIO

# Import the service under test
from services.azure_speech import AzureSpeechService, azure_speech_service, get_azure_speech_service
from models.data_models import SpeechTranscription


class TestAzureSpeechServiceInitialization:
    """Test Azure Speech Service initialization and configuration."""
    
    def test_initialization_success(self):
        """Test successful service initialization with valid configuration."""
        with patch.dict(os.environ, {
            'AZURE_SPEECH_KEY': 'test_key',
            'AZURE_SPEECH_REGION': 'eastus'
        }):
            service = AzureSpeechService()
            
            assert service.speech_key == 'test_key'
            assert service.speech_region == 'eastus'
            assert service.language == 'en-US'
            assert service.timeout_seconds == 30
            assert service.max_retries == 3
            assert not service.is_continuous_active()
            assert len(service.get_supported_formats()) > 0
    
    def test_initialization_missing_key(self):
        """Test initialization failure when Azure Speech key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="AZURE_SPEECH_KEY environment variable is required"):
                AzureSpeechService()
    
    def test_supported_formats(self):
        """Test that all expected audio formats are supported."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            supported = service.get_supported_formats()
            expected_formats = ['wav', 'mp3', 'm4a', 'ogg', 'flac', 'aac']
            
            for format_type in expected_formats:
                assert format_type in supported
    
    def test_language_configuration(self):
        """Test language setting functionality."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            # Test valid language codes
            service.set_language('es-ES')
            assert service.language == 'es-ES'
            
            service.set_language('fr-FR')
            assert service.language == 'fr-FR'
            
            # Test invalid language codes
            with pytest.raises(ValueError):
                service.set_language('')
            
            with pytest.raises(ValueError):
                service.set_language('x')
    
    def test_timeout_configuration(self):
        """Test timeout setting functionality."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            # Test valid timeouts
            service.set_timeout(60)
            assert service.timeout_seconds == 60
            
            service.set_timeout(10)
            assert service.timeout_seconds == 10
            
            # Test invalid timeouts
            with pytest.raises(ValueError):
                service.set_timeout(0)
            
            with pytest.raises(ValueError):
                service.set_timeout(-5)


class TestAudioFormatValidation:
    """Test audio format validation and detection."""
    
    def test_wav_format_detection(self):
        """Test WAV format detection from audio headers."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            # Mock WAV header
            wav_header = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00'
            wav_data = wav_header + b'\x00' * 100
            
            detected_format = service._validate_audio_format(wav_data)
            assert detected_format == 'wav'
    
    def test_mp3_format_detection(self):
        """Test MP3 format detection from audio headers."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            # Mock MP3 header
            mp3_header = b'\xff\xfb\x90\x00'
            mp3_data = mp3_header + b'\x00' * 100
            
            detected_format = service._validate_audio_format(mp3_data)
            assert detected_format == 'mp3'
    
    def test_format_hint_validation(self):
        """Test format validation with format hints."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            # Valid format with enough data
            audio_data = b'\x00' * 100
            
            # Valid formats
            for format_type in ['wav', 'mp3', 'm4a', 'ogg', 'flac']:
                validated = service._validate_audio_format(audio_data, format_type)
                assert validated == format_type
            
            # Invalid format
            with pytest.raises(ValueError, match="Unsupported audio format"):
                service._validate_audio_format(audio_data, 'xyz')
    
    def test_empty_audio_validation(self):
        """Test validation of empty or insufficient audio data."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            # Empty data
            with pytest.raises(ValueError, match="Audio data is empty or too small"):
                service._validate_audio_format(b'')
            
            # Insufficient data
            with pytest.raises(ValueError, match="Audio data is empty or too small"):
                service._validate_audio_format(b'\x00' * 10)
    
    def test_unknown_format_detection(self):
        """Test behavior with unknown audio format."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            # Unknown format header
            unknown_data = b'\xAB\xCD\xEF\x00' + b'\x00' * 100
            
            with pytest.raises(ValueError, match="Unable to detect audio format"):
                service._validate_audio_format(unknown_data)


class TestSpeechConfigurationCreation:
    """Test Azure Speech SDK configuration creation."""
    
    @patch('services.azure_speech.speechsdk.SpeechConfig')
    def test_speech_config_creation(self, mock_speech_config):
        """Test speech configuration creation with proper settings."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key', 'AZURE_SPEECH_REGION': 'westus'}):
            service = AzureSpeechService()
            mock_config = Mock()
            mock_speech_config.return_value = mock_config
            
            result = service._create_speech_config()
            
            # Verify SpeechConfig was called with correct parameters
            mock_speech_config.assert_called_once_with(
                subscription='test_key',
                region='westus'
            )
            
            # Verify configuration methods were called
            assert mock_config.enable_dictation.called
            assert mock_config.set_profanity.called
            assert mock_config.set_property.called
            assert result == mock_config
    
    def test_audio_config_file_source(self):
        """Test audio configuration creation from file source."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            # Test with non-existent file
            with pytest.raises(FileNotFoundError):
                service._create_audio_config('/non/existent/file.wav')
    
    def test_audio_config_bytes_source(self):
        """Test audio configuration creation from bytes source."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            with patch('services.azure_speech.speechsdk.AudioConfig') as mock_audio_config:
                service = AzureSpeechService()
                audio_bytes = b'\x00' * 1000
                
                result = service._create_audio_config(audio_bytes)
                
                assert mock_audio_config.called
    
    def test_audio_config_invalid_source(self):
        """Test audio configuration with invalid source type."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            with pytest.raises(ValueError, match="Unsupported audio source type"):
                service._create_audio_config(12345)  # Invalid type


class TestTranscriptionMethods:
    """Test transcription functionality with mocked Azure SDK."""
    
    @pytest.mark.asyncio
    @patch('services.azure_speech.speechsdk.SpeechRecognizer')
    @patch('services.azure_speech.speechsdk.AudioConfig')
    async def test_transcribe_audio_file_success(self, mock_audio_config, mock_recognizer):
        """Test successful file transcription."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            # Mock successful recognition result
            mock_result = Mock()
            mock_result.reason = Mock()
            mock_result.reason.__eq__ = lambda self, other: other.name == 'RecognizedSpeech'
            mock_result.text = "Hello world"
            
            # Mock recognizer
            mock_recognizer_instance = Mock()
            mock_async_result = Mock()
            mock_async_result.get.return_value = mock_result
            mock_recognizer_instance.recognize_once_async.return_value = mock_async_result
            mock_recognizer.return_value = mock_recognizer_instance
            
            # Mock file existence
            with patch('pathlib.Path.exists', return_value=True):
                # Mock the enum comparison
                with patch('services.azure_speech.speechsdk.ResultReason') as mock_reason:
                    mock_reason.RecognizedSpeech = Mock()
                    mock_result.reason = mock_reason.RecognizedSpeech
                    
                    result = await service.transcribe_audio_file('/fake/audio.wav')
                    
                    assert isinstance(result, SpeechTranscription)
                    assert result.text == "Hello world"
                    assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_file_not_found(self):
        """Test transcription with non-existent file."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            with pytest.raises(FileNotFoundError):
                await service.transcribe_audio_file('/non/existent/file.wav')
    
    @pytest.mark.asyncio
    @patch('services.azure_speech.speechsdk.SpeechRecognizer')
    @patch('services.azure_speech.speechsdk.AudioConfig')
    async def test_transcribe_audio_bytes_success(self, mock_audio_config, mock_recognizer):
        """Test successful bytes transcription."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            # Mock successful recognition result
            mock_result = Mock()
            mock_result.text = "Transcribed text"
            
            # Mock recognizer
            mock_recognizer_instance = Mock()
            mock_async_result = Mock()
            mock_async_result.get.return_value = mock_result
            mock_recognizer_instance.recognize_once_async.return_value = mock_async_result
            mock_recognizer.return_value = mock_recognizer_instance
            
            # Mock push stream
            with patch('services.azure_speech.speechsdk.audio.PushAudioInputStream'):
                with patch('services.azure_speech.speechsdk.ResultReason') as mock_reason:
                    mock_reason.RecognizedSpeech = Mock()
                    mock_result.reason = mock_reason.RecognizedSpeech
                    
                    # Test with WAV data
                    wav_header = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00'
                    wav_data = wav_header + b'\x00' * 100
                    
                    result = await service.transcribe_audio_bytes(wav_data, 'wav')
                    
                    assert isinstance(result, SpeechTranscription)
                    assert result.text == "Transcribed text"
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_bytes_empty_data(self):
        """Test transcription with empty audio data."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            with pytest.raises(ValueError, match="Audio data cannot be empty"):
                await service.transcribe_audio_bytes(b'', 'wav')
    
    @pytest.mark.asyncio
    @patch('services.azure_speech.speechsdk.SpeechRecognizer')
    async def test_transcription_retry_logic(self, mock_recognizer):
        """Test retry logic on transcription failures."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            service.max_retries = 2
            service.retry_delay = 0.1  # Fast retry for testing
            
            # Mock recognizer that always fails
            mock_recognizer_instance = Mock()
            mock_recognizer_instance.recognize_once_async.side_effect = RuntimeError("Azure service error")
            mock_recognizer.return_value = mock_recognizer_instance
            
            # Test with file (mock file existence)
            with patch('pathlib.Path.exists', return_value=True):
                with pytest.raises(RuntimeError, match="Transcription failed after 2 attempts"):
                    await service.transcribe_audio_file('/fake/audio.wav')


class TestContinuousRecognition:
    """Test continuous recognition functionality."""
    
    @pytest.mark.asyncio
    async def test_start_continuous_recognition_success(self):
        """Test successful start of continuous recognition."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            # Mock recognizer
            with patch('services.azure_speech.speechsdk.SpeechRecognizer') as mock_recognizer:
                mock_recognizer_instance = Mock()
                mock_recognizer_instance.recognized = Mock()
                mock_recognizer_instance.canceled = Mock()
                mock_recognizer_instance.start_continuous_recognition_async = Mock()
                mock_recognizer.return_value = mock_recognizer_instance
                
                # Mock audio config
                with patch('services.azure_speech.speechsdk.AudioConfig'):
                    callback = Mock()
                    
                    await service.start_continuous_recognition(callback)
                    
                    assert service.is_continuous_active()
                    assert mock_recognizer_instance.start_continuous_recognition_async.called
    
    @pytest.mark.asyncio
    async def test_start_continuous_recognition_invalid_callback(self):
        """Test start continuous recognition with invalid callback."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            # Test with None callback
            with pytest.raises(ValueError, match="Valid callback function is required"):
                await service.start_continuous_recognition(None)
            
            # Test with non-callable
            with pytest.raises(ValueError, match="Valid callback function is required"):
                await service.start_continuous_recognition("not_a_function")
    
    @pytest.mark.asyncio
    async def test_start_continuous_recognition_already_active(self):
        """Test start continuous recognition when already active."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            service._is_continuous_active = True  # Simulate active state
            
            callback = Mock()
            
            with pytest.raises(RuntimeError, match="Continuous recognition is already active"):
                await service.start_continuous_recognition(callback)
    
    @pytest.mark.asyncio
    async def test_stop_continuous_recognition_success(self):
        """Test successful stop of continuous recognition."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            # Simulate active continuous recognition
            mock_recognizer = Mock()
            mock_recognizer.stop_continuous_recognition_async.return_value.get.return_value = None
            service._continuous_recognizer = mock_recognizer
            service._is_continuous_active = True
            
            await service.stop_continuous_recognition()
            
            assert not service.is_continuous_active()
            assert service._continuous_recognizer is None
            assert mock_recognizer.stop_continuous_recognition_async.called
    
    @pytest.mark.asyncio
    async def test_stop_continuous_recognition_not_active(self):
        """Test stop continuous recognition when not active."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            with pytest.raises(RuntimeError, match="Continuous recognition is not active"):
                await service.stop_continuous_recognition()


class TestResultProcessing:
    """Test recognition result processing."""
    
    def test_process_successful_result(self):
        """Test processing of successful recognition result."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            # Mock successful result
            mock_result = Mock()
            mock_result.text = "Hello world"
            mock_result.properties = None
            
            with patch('services.azure_speech.speechsdk.ResultReason') as mock_reason:
                mock_reason.RecognizedSpeech = Mock()
                mock_result.reason = mock_reason.RecognizedSpeech
                
                transcription = service._process_recognition_result(mock_result)
                
                assert isinstance(transcription, SpeechTranscription)
                assert transcription.text == "Hello world"
                assert transcription.confidence == 1.0
    
    def test_process_no_match_result(self):
        """Test processing of no match result."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            # Mock no match result
            mock_result = Mock()
            
            with patch('services.azure_speech.speechsdk.ResultReason') as mock_reason:
                mock_reason.NoMatch = Mock()
                mock_result.reason = mock_reason.NoMatch
                
                transcription = service._process_recognition_result(mock_result)
                
                assert isinstance(transcription, SpeechTranscription)
                assert transcription.text == ""
                assert transcription.confidence == 0.0
    
    def test_process_canceled_result(self):
        """Test processing of canceled result."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            # Mock canceled result
            mock_result = Mock()
            mock_cancellation = Mock()
            mock_cancellation.reason = "ServiceError"
            mock_cancellation.error_details = "Network error"
            mock_result.cancellation_details = mock_cancellation
            
            with patch('services.azure_speech.speechsdk.ResultReason') as mock_reason:
                mock_reason.Canceled = Mock()
                mock_result.reason = mock_reason.Canceled
                
                with pytest.raises(ValueError, match="Recognition canceled"):
                    service._process_recognition_result(mock_result)


class TestServiceHealth:
    """Test service health and monitoring functionality."""
    
    @pytest.mark.asyncio
    async def test_service_health_check(self):
        """Test service health check functionality."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key', 'AZURE_SPEECH_REGION': 'eastus'}):
            service = AzureSpeechService()
            
            health = await service.get_service_health()
            
            assert 'status' in health
            assert 'region' in health
            assert 'language' in health
            assert 'supported_formats' in health
            assert 'continuous_active' in health
            assert 'timestamp' in health
            
            assert health['region'] == 'eastus'
            assert health['language'] == 'en-US'
            assert health['continuous_active'] is False


class TestDependencyInjection:
    """Test FastAPI dependency injection functionality."""
    
    @pytest.mark.asyncio
    async def test_get_azure_speech_service(self):
        """Test dependency injection function."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = await get_azure_speech_service()
            
            assert isinstance(service, AzureSpeechService)
            assert service.speech_key == 'test_key'
    
    def test_module_level_instance(self):
        """Test module-level service instance."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            # Import should create the instance
            from services.azure_speech import azure_speech_service
            
            assert isinstance(azure_speech_service, AzureSpeechService)


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""
    
    @pytest.mark.asyncio
    @patch('services.azure_speech.speechsdk.SpeechRecognizer')
    async def test_azure_authentication_error(self, mock_recognizer):
        """Test handling of Azure authentication errors."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'invalid_key'}):
            service = AzureSpeechService()
            
            # Mock authentication error
            mock_recognizer_instance = Mock()
            mock_recognizer_instance.recognize_once_async.side_effect = Exception("Authentication failed")
            mock_recognizer.return_value = mock_recognizer_instance
            
            with patch('pathlib.Path.exists', return_value=True):
                with pytest.raises(RuntimeError, match="Transcription failed"):
                    await service.transcribe_audio_file('/fake/audio.wav')
    
    @pytest.mark.asyncio
    async def test_network_connectivity_error(self):
        """Test handling of network connectivity issues."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            # Mock network error during configuration
            with patch.object(service, '_create_speech_config', side_effect=RuntimeError("Network error")):
                with patch('pathlib.Path.exists', return_value=True):
                    with pytest.raises(RuntimeError, match="Transcription failed"):
                        await service.transcribe_audio_file('/fake/audio.wav')
    
    def test_configuration_error_handling(self):
        """Test handling of configuration errors."""
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': 'test_key'}):
            service = AzureSpeechService()
            
            # Mock configuration failure
            with patch('services.azure_speech.speechsdk.SpeechConfig', side_effect=Exception("Config error")):
                with pytest.raises(RuntimeError, match="Speech configuration failed"):
                    service._create_speech_config()


# Integration test placeholder
class TestIntegration:
    """Integration tests (require actual Azure credentials)."""
    
    @pytest.mark.skip(reason="Requires actual Azure Speech Service credentials")
    @pytest.mark.asyncio
    async def test_real_azure_transcription(self):
        """Test with real Azure Speech Service (manual test only)."""
        # This test should only be run manually with real credentials
        # and a real audio file for end-to-end validation
        pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])