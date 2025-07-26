"""
Comprehensive Pydantic data models for EmoVox multimodal emotion analysis system.

This module contains all the data schemas used throughout the application for
emotion analysis, risk assessment, API communication, and data persistence.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import time

from pydantic import BaseModel, Field, field_validator, ConfigDict


class RiskLevel(str, Enum):
    """Risk level enumeration for mental health assessment."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class EmotionCategory(str, Enum):
    """Standard emotion categories for analysis."""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    NEUTRAL = "neutral"
    ANXIOUS = "anxious"
    DEPRESSED = "depressed"
    STRESSED = "stressed"


class SentimentCategory(str, Enum):
    """Sentiment categories for text analysis."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class MessageType(str, Enum):
    """WebSocket message types."""
    AUDIO_DATA = "audio_data"
    VIDEO_DATA = "video_data"
    ANALYSIS_RESULT = "analysis_result"
    ERROR = "error"
    STATUS = "status"
    HEARTBEAT = "heartbeat"


# =============================
# Emotion Analysis Models
# =============================

class FacialEmotionResult(BaseModel):
    """
    Result model for facial emotion analysis.
    
    Contains the dominant emotion detected, confidence scores for all emotions,
    overall confidence, and timestamp information.
    """
    model_config = ConfigDict(
        str_to_lower=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    dominant_emotion: str = Field(
        ...,
        description="The most prominent emotion detected in the facial analysis"
    )
    emotions: Dict[str, float] = Field(
        ...,
        description="Confidence scores for each emotion category (0.0 to 1.0)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence score for the facial emotion detection"
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp when the analysis was performed"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if analysis failed"
    )
    
    @field_validator('emotions')
    @classmethod
    def validate_emotion_scores(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that all emotion scores are between 0 and 1."""
        for emotion, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Emotion score for {emotion} must be between 0.0 and 1.0")
        return v


class AudioEmotionResult(BaseModel):
    """
    Result model for audio emotion analysis.
    
    Contains emotion detection results from audio analysis including
    the dominant emotion, confidence scores, and model information.
    """
    model_config = ConfigDict(
        str_to_lower=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    dominant_emotion: str = Field(
        ...,
        description="The most prominent emotion detected in the audio analysis"
    )
    emotion_scores: Dict[str, float] = Field(
        ...,
        description="Confidence scores for each emotion category (0.0 to 1.0)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence score for the audio emotion detection"
    )
    model: str = Field(
        ...,
        description="Name/version of the model used for analysis"
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp when the analysis was performed"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if analysis failed"
    )
    
    @field_validator('emotion_scores')
    @classmethod
    def validate_emotion_scores(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that all emotion scores are between 0 and 1."""
        for emotion, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Emotion score for {emotion} must be between 0.0 and 1.0")
        return v


# =============================
# Breathing Analysis Models
# =============================

class BreathingMetrics(BaseModel):
    """
    Breathing analysis metrics derived from audio or sensor data.
    
    Contains breathing rate, variability, signal quality metrics,
    and irregularity detection for mental health assessment.
    """
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    breathing_rate: float = Field(
        ...,
        ge=0.0,
        description="Breaths per minute detected from the audio signal"
    )
    breathing_variability: float = Field(
        ...,
        ge=0.0,
        description="Measure of breathing pattern variability (higher = more irregular)"
    )
    signal_quality: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Quality score of the breathing signal (0.0 to 1.0)"
    )
    irregularity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score indicating breathing irregularity (0.0 = regular, 1.0 = highly irregular)"
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp when the analysis was performed"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if analysis failed"
    )


# =============================
# Risk Assessment Models
# =============================

class RiskAssessment(BaseModel):
    """
    Mental health risk assessment based on multimodal analysis.
    
    Combines results from facial emotion, audio analysis, and breathing
    metrics to provide an overall risk assessment.
    """
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    risk_level: RiskLevel = Field(
        ...,
        description="Overall risk level assessment"
    )
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Numerical risk score (0.0 = no risk, 1.0 = high risk)"
    )
    contributing_factors: List[str] = Field(
        default_factory=list,
        description="List of factors contributing to the risk assessment"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the risk assessment"
    )
    modality_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Individual risk scores from each analysis modality"
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp when the assessment was performed"
    )
    
    @field_validator('modality_scores')
    @classmethod
    def validate_modality_scores(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that all modality scores are between 0 and 1."""
        for modality, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Modality score for {modality} must be between 0.0 and 1.0")
        return v


# =============================
# API Request/Response Models
# =============================

class AnalysisRequest(BaseModel):
    """
    Request model for emotion analysis API endpoint.
    
    Contains optional audio and video data along with session information
    for multimodal emotion analysis.
    """
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    audio_data: Optional[bytes] = Field(
        None,
        description="Raw audio data in supported format (WAV, MP3, etc.)"
    )
    video_data: Optional[bytes] = Field(
        None,
        description="Raw video data in supported format (MP4, AVI, etc.)"
    )
    session_id: Optional[str] = Field(
        None,
        description="Unique session identifier for tracking analysis sessions"
    )
    
    @field_validator('audio_data', 'video_data')
    @classmethod
    def validate_data_not_empty(cls, v: Optional[bytes]) -> Optional[bytes]:
        """Validate that data is not empty if provided."""
        if v is not None and len(v) == 0:
            raise ValueError("Data cannot be empty if provided")
        return v


class AnalysisResponse(BaseModel):
    """
    Response model for emotion analysis API endpoint.
    
    Contains results from all analysis modalities and processing metadata.
    """
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    facial_emotion: Optional[FacialEmotionResult] = Field(
        None,
        description="Facial emotion analysis results"
    )
    audio_emotion: Optional[AudioEmotionResult] = Field(
        None,
        description="Audio emotion analysis results"
    )
    breathing_metrics: Optional[BreathingMetrics] = Field(
        None,
        description="Breathing pattern analysis results"
    )
    risk_assessment: Optional[RiskAssessment] = Field(
        None,
        description="Overall risk assessment based on all modalities"
    )
    processing_time: float = Field(
        ...,
        ge=0.0,
        description="Total processing time in seconds"
    )


# =============================
# WebSocket Models
# =============================

class WebSocketMessage(BaseModel):
    """
    Generic WebSocket message model for real-time communication.
    
    Used for sending various types of messages between client and server
    during real-time analysis sessions.
    """
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    message_type: MessageType = Field(
        ...,
        description="Type of the WebSocket message"
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Message payload data"
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp when the message was created"
    )


class StreamingData(BaseModel):
    """
    Model for streaming audio/video data via WebSocket.
    
    Contains chunked audio or video data for real-time analysis
    along with session identification.
    """
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    audio_chunk: Optional[bytes] = Field(
        None,
        description="Chunk of audio data for streaming analysis"
    )
    video_frame: Optional[bytes] = Field(
        None,
        description="Video frame data for streaming analysis"
    )
    session_id: str = Field(
        ...,
        min_length=1,
        description="Unique session identifier"
    )
    
    @field_validator('audio_chunk', 'video_frame')
    @classmethod
    def validate_data_not_empty(cls, v: Optional[bytes]) -> Optional[bytes]:
        """Validate that data is not empty if provided."""
        if v is not None and len(v) == 0:
            raise ValueError("Data chunk cannot be empty if provided")
        return v


# =============================
# Database Models
# =============================

class SessionRecord(BaseModel):
    """
    Database model for storing analysis session information.
    
    Contains session metadata, file paths, and analysis results
    for persistent storage and retrieval.
    """
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    session_id: str = Field(
        ...,
        min_length=1,
        description="Unique session identifier"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Session creation timestamp"
    )
    audio_path: Optional[str] = Field(
        None,
        description="File path to stored audio data"
    )
    video_path: Optional[str] = Field(
        None,
        description="File path to stored video data"
    )
    analysis_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Complete analysis results stored as JSON"
    )


# =============================
# Azure Service Models
# =============================

class SpeechTranscription(BaseModel):
    """
    Model for Azure Speech Services transcription results.
    
    Contains transcribed text with confidence metrics and timing information.
    """
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    text: str = Field(
        ...,
        description="Transcribed text from speech"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the transcription"
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp when transcription was completed"
    )


class TextSentiment(BaseModel):
    """
    Model for Azure Text Analytics sentiment analysis results.
    
    Contains sentiment classification and confidence scores for
    positive, negative, and neutral sentiment categories.
    """
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    sentiment: SentimentCategory = Field(
        ...,
        description="Overall sentiment classification"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in sentiment analysis"
    )
    positive_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for positive sentiment"
    )
    negative_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for negative sentiment"
    )
    neutral_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for neutral sentiment"
    )
    
    @field_validator('positive_score', 'negative_score', 'neutral_score')
    @classmethod
    def validate_sentiment_scores_sum(cls, v: float, info) -> float:
        """Validate individual sentiment scores are valid."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Sentiment scores must be between 0.0 and 1.0")
        return v


# =============================
# Utility Models
# =============================

class ErrorResponse(BaseModel):
    """
    Standard error response model for API endpoints.
    """
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    error: str = Field(
        ...,
        description="Error message"
    )
    error_code: Optional[str] = Field(
        None,
        description="Specific error code for programmatic handling"
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp when error occurred"
    )


class HealthCheckResponse(BaseModel):
    """
    Health check response model for service monitoring.
    """
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    status: str = Field(
        ...,
        description="Health status (healthy, unhealthy, degraded)"
    )
    version: str = Field(
        ...,
        description="Application version"
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp of health check"
    )
    dependencies: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of external dependencies"
    )