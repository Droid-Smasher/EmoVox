"""
EmoVox FastAPI Main Application

This module contains the main FastAPI application for the EmoVox multimodal emotion analysis system.
It provides REST API endpoints and WebSocket connections for real-time emotion analysis.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from dotenv import load_dotenv

from models.data_models import HealthCheckResponse, ErrorResponse


# Load environment variables from .env file
load_dotenv()


# =============================
# Configuration
# =============================

class Config:
    """Application configuration loaded from environment variables."""
    
    # Application settings
    APP_TITLE: str = os.getenv("API_TITLE", "EmoVox API")
    APP_VERSION: str = os.getenv("API_VERSION", "1.0.0")
    APP_DESCRIPTION: str = os.getenv(
        "API_DESCRIPTION", 
        "Multimodal emotion and breathing analysis system combining facial emotion recognition, "
        "audio emotion detection, and breathing pattern analysis for comprehensive mental health monitoring."
    )
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    RELOAD: bool = os.getenv("RELOAD", "false").lower() == "true"
    
    # Storage paths
    LOCAL_STORAGE_PATH: str = os.getenv("LOCAL_STORAGE_PATH", "./data")
    AUDIO_STORAGE_PATH: str = os.getenv("AUDIO_STORAGE_PATH", "./data/audio")
    VIDEO_STORAGE_PATH: str = os.getenv("VIDEO_STORAGE_PATH", "./data/video")
    MODELS_STORAGE_PATH: str = os.getenv("MODELS_STORAGE_PATH", "./data/models")
    
    # File upload settings
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
    
    # CORS settings
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:8080", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080"
    ]
    
    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv(
        "LOG_FORMAT", 
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


config = Config()


# =============================
# Logging Configuration
# =============================

def setup_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper()),
        format=config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("emovox.log", mode="a")
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)


# =============================
# Directory Creation
# =============================

def create_data_directories() -> None:
    """Create necessary data directories if they don't exist."""
    directories = [
        config.LOCAL_STORAGE_PATH,
        config.AUDIO_STORAGE_PATH,
        config.VIDEO_STORAGE_PATH,
        config.MODELS_STORAGE_PATH
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logging.info(f"Created/verified directory: {directory}")


# =============================
# Lifespan Events
# =============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    
    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup events
    logging.info("Starting EmoVox API...")
    
    try:
        # Set up logging
        setup_logging()
        logging.info("Logging configuration completed")
        
        # Create data directories
        create_data_directories()
        logging.info("Data directories created/verified")
        
        # Log configuration
        logging.info(f"Server configuration: {config.HOST}:{config.PORT}")
        logging.info(f"Debug mode: {config.DEBUG}")
        logging.info(f"Max file size: {config.MAX_FILE_SIZE_MB}MB")
        
        logging.info("EmoVox API startup completed successfully")
        
    except Exception as e:
        logging.error(f"Failed to start EmoVox API: {str(e)}")
        raise
    
    yield
    
    # Shutdown events
    logging.info("Shutting down EmoVox API...")
    logging.info("EmoVox API shutdown completed")


# =============================
# FastAPI Application
# =============================

app = FastAPI(
    title=config.APP_TITLE,
    version=config.APP_VERSION,
    description=config.APP_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# =============================
# Middleware Configuration
# =============================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if config.DEBUG else config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Logging Middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all incoming requests and responses."""
    start_time = time.time()
    
    # Log request
    logging.info(
        f"Request: {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'}"
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logging.info(
        f"Response: {response.status_code} "
        f"processed in {process_time:.4f}s"
    )
    
    # Add processing time header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# =============================
# Exception Handlers
# =============================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions with consistent error format."""
    error_response = ErrorResponse(
        error=exc.detail,
        error_code=f"HTTP_{exc.status_code}"
    )
    
    logging.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle request validation errors."""
    error_details = []
    for error in exc.errors():
        error_details.append(f"{'.'.join(str(x) for x in error['loc'])}: {error['msg']}")
    
    error_response = ErrorResponse(
        error=f"Validation error: {'; '.join(error_details)}",
        error_code="VALIDATION_ERROR"
    )
    
    logging.warning(f"Validation Error: {error_response.error}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    error_response = ErrorResponse(
        error="Internal server error occurred",
        error_code="INTERNAL_ERROR"
    )
    
    logging.error(f"Unexpected error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump()
    )


# =============================
# Core Endpoints
# =============================

@app.get("/", response_model=Dict[str, Any])
async def root() -> Dict[str, Any]:
    """
    Root endpoint providing API status and basic information.
    
    Returns:
        Dict containing API status, version, and available endpoints
    """
    return {
        "message": "EmoVox API is running",
        "version": config.APP_VERSION,
        "status": "healthy",
        "description": "Multimodal emotion and breathing analysis system",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "upload": "/upload (coming soon)",
            "websocket": "/ws/stream (coming soon)",
            "analysis": "/analysis (coming soon)"
        },
        "timestamp": time.time()
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint for service monitoring.
    
    Returns:
        HealthCheckResponse with current service status
    """
    # Check data directories
    directories_status = {}
    for name, path in [
        ("audio", config.AUDIO_STORAGE_PATH),
        ("video", config.VIDEO_STORAGE_PATH),
        ("models", config.MODELS_STORAGE_PATH)
    ]:
        directories_status[f"{name}_dir"] = "healthy" if Path(path).exists() else "unhealthy"
    
    # Determine overall status
    overall_status = "healthy"
    if any(status == "unhealthy" for status in directories_status.values()):
        overall_status = "degraded"
    
    return HealthCheckResponse(
        status=overall_status,
        version=config.APP_VERSION,
        dependencies=directories_status
    )


# =============================
# Router Preparation (Future Implementation)
# =============================

# These routers will be implemented in separate modules and included here
# app.include_router(upload_router, prefix="/upload", tags=["upload"])
# app.include_router(websocket_router, prefix="/ws", tags=["websocket"])
# app.include_router(analysis_router, prefix="/analysis", tags=["analysis"])


# =============================
# Application Entry Point
# =============================

def main() -> None:
    """
    Main function to run the FastAPI application with Uvicorn.
    
    This function is called when the script is run directly.
    It configures and starts the Uvicorn server with the appropriate settings.
    """
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD,
        log_level=config.LOG_LEVEL.lower(),
        access_log=True,
        use_colors=True,
        # File upload size limit
        limit_max_requests=1000,
        timeout_keep_alive=5
    )


if __name__ == "__main__":
    main()
