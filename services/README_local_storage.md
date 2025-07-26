# LocalStorageService Documentation

## Overview

The `LocalStorageService` is a comprehensive async storage service for the EmoVox system that handles file management and SQLite database operations. It's designed for high concurrent load with proper connection pooling and error handling.

## Features

### Database Management
- **SQLite Database**: Manages `sessions.db` with sessions table
- **Async Operations**: All database operations use `aiosqlite` for non-blocking I/O
- **Connection Pooling**: Semaphore-controlled concurrent connections (max 10)
- **WAL Mode**: Enabled for better concurrent read/write access
- **Automatic Indexing**: Performance-optimized indexes on frequently queried columns

### File Storage
- **Unique Filenames**: Generated using timestamp + session_id + UUID
- **Date Organization**: Files stored in subdirectories by date (YYYY-MM-DD)
- **Format Support**: 
  - Audio: `.wav`, `.mp3`, `.m4a`, `.aac`, `.ogg`
  - Video: `.mp4`, `.webm`, `.avi`, `.mov`, `.mkv`
- **Size Validation**: Configurable limits (default: 50MB audio, 200MB video)
- **Atomic Operations**: Safe file operations with proper error handling

### Session Management
- **CRUD Operations**: Create, read, update session records
- **Analysis Storage**: JSON storage of multimodal analysis results
- **Recent Sessions**: Efficient querying of recent sessions
- **Cleanup Operations**: Automated cleanup of old files and sessions

## Database Schema

```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    audio_path TEXT,
    video_path TEXT,
    analysis_results TEXT  -- JSON string
);

CREATE INDEX idx_sessions_created_at ON sessions(created_at DESC);
```

## Configuration

Environment variables for configuration:

```bash
# Storage paths
LOCAL_STORAGE_PATH=./data
AUDIO_STORAGE_PATH=./data/audio
VIDEO_STORAGE_PATH=./data/video
DATABASE_PATH=./data/sessions.db

# File size limits (bytes)
MAX_AUDIO_SIZE=52428800  # 50MB
MAX_VIDEO_SIZE=209715200 # 200MB
```

## Usage Examples

### Basic Setup

```python
from services.local_storage import get_local_storage_service

# Get service instance (dependency injection)
storage = await get_local_storage_service()
```

### Session Management

```python
# Create new session
session = await storage.create_session("session-123")

# Save audio file
audio_path = await storage.save_audio_file(
    audio_data=audio_bytes,
    session_id="session-123",
    format="wav"
)

# Update session with file paths
await storage.update_session_file_paths(
    session_id="session-123",
    audio_path=audio_path
)

# Update with analysis results
analysis_results = {
    "facial_emotion": {"dominant_emotion": "happy", "confidence": 0.85},
    "risk_assessment": {"risk_level": "low", "risk_score": 0.15}
}
await storage.update_session_analysis("session-123", analysis_results)

# Retrieve session
session = await storage.get_session("session-123")
```

### File Operations

```python
# Check file existence
exists = await storage.file_exists("/path/to/file")

# Get file size
size = await storage.get_file_size("/path/to/file")

# Delete file
deleted = await storage.delete_file("/path/to/file")
```

### Storage Statistics

```python
stats = await storage.get_storage_stats()
print(f"Total sessions: {stats['total_sessions']}")
print(f"Total storage: {stats['total_size']} bytes")
```

### Cleanup Operations

```python
# Clean up files older than 30 days
cleanup_stats = await storage.cleanup_old_files(days_old=30)
print(f"Cleaned up {cleanup_stats['sessions_deleted']} sessions")
print(f"Freed {cleanup_stats['bytes_freed']} bytes")
```

## API Reference

### Core Methods

#### `initialize_database() -> None`
Initialize SQLite database and create tables if they don't exist.

#### `create_session(session_id: str) -> SessionRecord`
Create a new session record in the database.

#### `save_audio_file(audio_data: bytes, session_id: str, format: str) -> str`
Save audio file and return relative file path.

#### `save_video_file(video_data: bytes, session_id: str, format: str) -> str`
Save video file and return relative file path.

#### `update_session_analysis(session_id: str, analysis_results: Dict) -> None`
Update session with analysis results.

#### `get_session(session_id: str) -> Optional[SessionRecord]`
Retrieve session data by session ID.

#### `list_recent_sessions(limit: int = 10) -> List[SessionRecord]`
Get recent sessions ordered by creation time.

### Utility Methods

#### `get_file_size(file_path: str) -> int`
Get file size in bytes.

#### `file_exists(file_path: str) -> bool`
Check if file exists.

#### `delete_file(file_path: str) -> bool`
Delete file safely.

#### `get_storage_stats() -> Dict[str, any]`
Get comprehensive storage usage statistics.

#### `cleanup_old_files(days_old: int = 30) -> Dict[str, int]`
Clean up old files and sessions.

## Error Handling

The service provides comprehensive error handling:

- **`ValueError`**: Invalid input parameters, unsupported formats, file size limits
- **`FileNotFoundError`**: File operations on non-existent files
- **`IOError`**: File I/O operation failures
- **`RuntimeError`**: Database operation failures

All errors are logged with appropriate detail levels.

## Performance Considerations

- **Async Operations**: All I/O operations are non-blocking
- **Connection Pooling**: Limited concurrent database connections
- **Efficient Indexing**: Database indexes on frequently queried columns
- **Date-based Organization**: Hierarchical file structure for better filesystem performance
- **WAL Mode**: SQLite Write-Ahead Logging for concurrent access

## Testing

Run the comprehensive test suite:

```bash
python test_local_storage.py
```

The test covers:
- Database initialization
- Session CRUD operations
- File storage operations
- Error handling scenarios
- Utility method functionality
- Storage statistics
- Cleanup operations

## Dependencies

- `aiosqlite>=0.20.0`: Async SQLite operations
- `aiofiles>=24.1.0`: Async file operations
- `pydantic`: Data validation (SessionRecord model)

## Thread Safety

The service is designed for concurrent use:
- Semaphore-controlled database connections
- Atomic file operations
- Proper exception handling in concurrent scenarios
- WAL mode for better concurrent database access