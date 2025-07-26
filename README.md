# EmoVox API

EmoVox is a high-performance REST API for multimodal emotion and breathing analysis. It combines facial emotion recognition (FER), voice emotion recognition (VER), and breathing pattern analysis to provide a comprehensive system for mental health monitoring and emotional state assessment.

The system is built with FastAPI and leverages a suite of machine learning and signal processing libraries to analyze video, audio, and physiological data in real-time or from uploaded files.

## Core Features

- **Multimodal Analysis:** Integrates three distinct analysis modalities for a holistic assessment:
    - **Facial Emotion Recognition:** Detects emotions from video streams or files using the `DeepFace` library.
    - **Audio Emotion & Speech Analysis:** Analyzes vocal tones and speech patterns for emotional content using `librosa`, `torchaudio`, and Azure Speech Services.
    - **Breathing Pattern Analysis:** Processes physiological data to analyze breathing rates and patterns using `neurokit2`.
- **Real-time Processing:** Provides WebSocket endpoints for real-time data streaming and analysis.
- **REST Endpoints:** Offers clear, documented RESTful endpoints for file uploads, status checks, and retrieving analysis results.
- **Configurable Storage:** Supports configurable local storage for audio, video, and model files.
- **Robust & Scalable:** Built on an asynchronous framework (FastAPI) suitable for handling concurrent requests and I/O-bound operations efficiently.

## Technology Stack

### Backend
- **Framework:** FastAPI
- **Web Server:** Uvicorn

### Machine Learning & Data Processing
- **Deep Learning:** TensorFlow, PyTorch
- **Transformers:** Hugging Face Transformers
- **Facial Recognition:** DeepFace
- **Audio Processing:** Librosa, torchaudio, SciPy
- **Physiological Signal Processing:** NeuroKit2
- **Data Handling:** NumPy

### Cloud & Services
- **Microsoft Azure:** Integrates with Azure Speech Services for advanced speech-to-text and sentiment analysis.

### Development & Tooling
- **Dependency Management:** uv
- **Environment Configuration:** python-dotenv

## API Endpoints

The following are the core endpoints provided by the application:

- `GET /`: Root endpoint with basic API information and status.
- `GET /health`: Health check endpoint to monitor service status and dependencies.
- `POST /upload`: (Coming Soon) For uploading audio/video files for analysis.
- `GET /analysis`: (Coming Soon) To retrieve the results of an analysis task.
- `WS /ws/stream`: (Coming Soon) WebSocket for real-time, bidirectional communication and data streaming.

## Configuration

The application is configured via environment variables defined in a `.env` file. A comprehensive list of variables can be found in the `Config` class within `main.py`.

Key variables include:
- `PORT`: The port on which the server runs.
- `DEBUG`: Enables or disables debug mode.
- `LOCAL_STORAGE_PATH`: The base path for storing data files.
- `AUDIO_STORAGE_PATH`: Directory for storing audio files.
- `VIDEO_STORAGE_PATH`: Directory for storing video files.
- `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`: Credentials for Azure Cognitive Services.

## Setup and Installation

1.  **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd hackathon-project
    ```

2.  **Create a virtual environment and install dependencies:**
    This project uses `uv` for package management.
    ```sh
    python -m venv .venv
    source .venv/bin/activate
    uv pip install -r requirements.txt 
    # Or, if starting from scratch with pyproject.toml
    # uv pip install -e .
    ```

3.  **Configure environment variables:**
    Create a `.env` file in the project root and populate it with the necessary configuration.
    ```env
    # Sample .env
    API_TITLE="EmoVox API"
    PORT=8000
    DEBUG=true
    RELOAD=true
    LOCAL_STORAGE_PATH="./data"
    # Add other variables as needed...
    ```

4.  **Run the application:**
    The application can be started by running the `main.py` script directly.
    ```sh
    python main.py
    ```
    The server will be accessible at `http://localhost:8000`. API documentation is available at `http://localhost:8000/docs`.