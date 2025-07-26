"""
Comprehensive local storage service for EmoVox system.

This module provides file management and SQLite database operations for storing
audio/video files and session metadata. Designed for high concurrent load with
async/await patterns throughout.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiofiles
import aiosqlite
from pydantic import ValidationError

from models.data_models import SessionRecord

# Configure logging
logger = logging.getLogger(__name__)


class LocalStorageService:
    """
    Async-capable local storage service for EmoVox system.
    
    Handles file storage operations for audio/video files and SQLite database
    operations for session management with proper connection pooling and
    error handling.
    """
    
    def __init__(self):
        """Initialize the local storage service with configuration."""
        # Load configuration from environment variables
        self.storage_path = Path(os.getenv("LOCAL_STORAGE_PATH", "./data"))
        self.audio_path = Path(os.getenv("AUDIO_STORAGE_PATH", "./data/audio"))
        self.video_path = Path(os.getenv("VIDEO_STORAGE_PATH", "./data/video"))
        self.database_path = Path(os.getenv("DATABASE_PATH", "./data/sessions.db"))
        
        # File size limits (in bytes)
        self.max_audio_size = int(os.getenv("MAX_AUDIO_SIZE", 50 * 1024 * 1024))  # 50MB
        self.max_video_size = int(os.getenv("MAX_VIDEO_SIZE", 200 * 1024 * 1024))  # 200MB
        
        # Supported file formats
        self.supported_audio_formats = {".wav", ".mp3", ".m4a", ".aac", ".ogg"}
        self.supported_video_formats = {".mp4", ".webm", ".avi", ".mov", ".mkv"}
        
        # Database connection pool settings
        self._db_semaphore = asyncio.Semaphore(10)  # Max 10 concurrent connections
        self._initialized = False
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for path in [self.storage_path, self.audio_path, self.video_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    async def initialize_database(self) -> None:
        """
        Initialize SQLite database and create tables if they don't exist.
        
        Creates the sessions table with proper schema and indexes for performance.
        """
        try:
            async with self._db_semaphore:
                async with aiosqlite.connect(self.database_path) as db:
                    # Enable WAL mode for better concurrent access
                    await db.execute("PRAGMA journal_mode=WAL")
                    await db.execute("PRAGMA foreign_keys=ON")
                    await db.execute("PRAGMA synchronous=NORMAL")
                    
                    # Create sessions table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS sessions (
                            session_id TEXT PRIMARY KEY,
                            created_at TIMESTAMP NOT NULL,
                            audio_path TEXT,
                            video_path TEXT,
                            analysis_results TEXT
                        )
                    """)
                    
                    # Create indexes for performance
                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_sessions_created_at 
                        ON sessions(created_at DESC)
                    """)
                    
                    await db.commit()
                    
            self._initialized = True
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise RuntimeError(f"Database initialization failed: {e}")
    
    def _generate_unique_filename(self, session_id: str, file_format: str) -> str:
        """
        Generate a secure, unique filename with timestamp and session ID.
        
        Args:
            session_id: Unique session identifier
            file_format: File extension (e.g., '.wav', '.mp4')
        
        Returns:
            Unique filename string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{timestamp}_{session_id}_{unique_id}{file_format}"
    
    def _get_date_directory(self, base_path: Path) -> Path:
        """
        Get date-based subdirectory (YYYY-MM-DD format).
        
        Args:
            base_path: Base storage path
        
        Returns:
            Path to date-based subdirectory
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        date_dir = base_path / date_str
        date_dir.mkdir(exist_ok=True)
        return date_dir
    
    async def save_audio_file(self, audio_data: bytes, session_id: str, format: str) -> str:
        """
        Save audio file with unique filename and return file path.
        
        Args:
            audio_data: Raw audio data as bytes
            session_id: Unique session identifier
            format: Audio file format (e.g., 'wav', 'mp3')
        
        Returns:
            Relative file path to saved audio file
        
        Raises:
            ValueError: If format is not supported or file is too large
            IOError: If file save operation fails
        """
        # Validate format
        file_ext = f".{format.lower().lstrip('.')}"
        if file_ext not in self.supported_audio_formats:
            raise ValueError(f"Unsupported audio format: {format}")
        
        # Validate file size
        if len(audio_data) > self.max_audio_size:
            raise ValueError(f"Audio file too large: {len(audio_data)} bytes > {self.max_audio_size}")
        
        # Generate unique filename and path
        filename = self._generate_unique_filename(session_id, file_ext)
        date_dir = self._get_date_directory(self.audio_path)
        file_path = date_dir / filename
        
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(audio_data)
            
            # Return relative path from storage root
            relative_path = file_path.relative_to(self.storage_path)
            logger.info(f"Audio file saved: {relative_path}")
            return str(relative_path)
            
        except Exception as e:
            logger.error(f"Failed to save audio file: {e}")
            raise IOError(f"Audio file save failed: {e}")
    
    async def save_video_file(self, video_data: bytes, session_id: str, format: str) -> str:
        """
        Save video file with unique filename and return file path.
        
        Args:
            video_data: Raw video data as bytes
            session_id: Unique session identifier
            format: Video file format (e.g., 'mp4', 'webm')
        
        Returns:
            Relative file path to saved video file
        
        Raises:
            ValueError: If format is not supported or file is too large
            IOError: If file save operation fails
        """
        # Validate format
        file_ext = f".{format.lower().lstrip('.')}"
        if file_ext not in self.supported_video_formats:
            raise ValueError(f"Unsupported video format: {format}")
        
        # Validate file size
        if len(video_data) > self.max_video_size:
            raise ValueError(f"Video file too large: {len(video_data)} bytes > {self.max_video_size}")
        
        # Generate unique filename and path
        filename = self._generate_unique_filename(session_id, file_ext)
        date_dir = self._get_date_directory(self.video_path)
        file_path = date_dir / filename
        
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(video_data)
            
            # Return relative path from storage root
            relative_path = file_path.relative_to(self.storage_path)
            logger.info(f"Video file saved: {relative_path}")
            return str(relative_path)
            
        except Exception as e:
            logger.error(f"Failed to save video file: {e}")
            raise IOError(f"Video file save failed: {e}")
    
    async def create_session(self, session_id: str) -> SessionRecord:
        """
        Create a new session record in the database.
        
        Args:
            session_id: Unique session identifier
        
        Returns:
            Created SessionRecord instance
        
        Raises:
            RuntimeError: If database operation fails
            ValueError: If session already exists
        """
        if not self._initialized:
            await self.initialize_database()
        
        try:
            async with self._db_semaphore:
                async with aiosqlite.connect(self.database_path) as db:
                    # Check if session already exists
                    cursor = await db.execute(
                        "SELECT session_id FROM sessions WHERE session_id = ?",
                        (session_id,)
                    )
                    if await cursor.fetchone():
                        raise ValueError(f"Session {session_id} already exists")
                    
                    # Create new session
                    created_at = datetime.utcnow()
                    await db.execute("""
                        INSERT INTO sessions (session_id, created_at, audio_path, video_path, analysis_results)
                        VALUES (?, ?, NULL, NULL, '{}')
                    """, (session_id, created_at))
                    
                    await db.commit()
                    
            session_record = SessionRecord(
                session_id=session_id,
                created_at=created_at,
                audio_path=None,
                video_path=None,
                analysis_results={}
            )
            
            logger.info(f"Session created: {session_id}")
            return session_record
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {e}")
            raise RuntimeError(f"Session creation failed: {e}")
    
    async def update_session_analysis(self, session_id: str, analysis_results: Dict) -> None:
        """
        Update session with analysis results.
        
        Args:
            session_id: Unique session identifier
            analysis_results: Analysis results dictionary to store
        
        Raises:
            RuntimeError: If database operation fails
            ValueError: If session doesn't exist
        """
        if not self._initialized:
            await self.initialize_database()
        
        try:
            async with self._db_semaphore:
                async with aiosqlite.connect(self.database_path) as db:
                    # Check if session exists
                    cursor = await db.execute(
                        "SELECT session_id FROM sessions WHERE session_id = ?",
                        (session_id,)
                    )
                    if not await cursor.fetchone():
                        raise ValueError(f"Session {session_id} not found")
                    
                    # Update analysis results
                    results_json = json.dumps(analysis_results)
                    await db.execute("""
                        UPDATE sessions 
                        SET analysis_results = ?
                        WHERE session_id = ?
                    """, (results_json, session_id))
                    
                    await db.commit()
                    
            logger.info(f"Session analysis updated: {session_id}")
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to update session analysis {session_id}: {e}")
            raise RuntimeError(f"Session update failed: {e}")
    
    async def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """
        Retrieve session data by session ID.
        
        Args:
            session_id: Unique session identifier
        
        Returns:
            SessionRecord instance if found, None otherwise
        
        Raises:
            RuntimeError: If database operation fails
        """
        if not self._initialized:
            await self.initialize_database()
        
        try:
            async with self._db_semaphore:
                async with aiosqlite.connect(self.database_path) as db:
                    cursor = await db.execute("""
                        SELECT session_id, created_at, audio_path, video_path, analysis_results
                        FROM sessions WHERE session_id = ?
                    """, (session_id,))
                    
                    row = await cursor.fetchone()
                    if not row:
                        return None
                    
                    # Parse results
                    analysis_results = json.loads(row[4]) if row[4] else {}
                    
                    return SessionRecord(
                        session_id=row[0],
                        created_at=datetime.fromisoformat(row[1]),
                        audio_path=row[2],
                        video_path=row[3],
                        analysis_results=analysis_results
                    )
                    
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            raise RuntimeError(f"Session retrieval failed: {e}")
    
    async def list_recent_sessions(self, limit: int = 10) -> List[SessionRecord]:
        """
        Get recent sessions ordered by creation time.
        
        Args:
            limit: Maximum number of sessions to return
        
        Returns:
            List of SessionRecord instances
        
        Raises:
            RuntimeError: If database operation fails
        """
        if not self._initialized:
            await self.initialize_database()
        
        try:
            async with self._db_semaphore:
                async with aiosqlite.connect(self.database_path) as db:
                    cursor = await db.execute("""
                        SELECT session_id, created_at, audio_path, video_path, analysis_results
                        FROM sessions 
                        ORDER BY created_at DESC 
                        LIMIT ?
                    """, (limit,))
                    
                    rows = await cursor.fetchall()
                    sessions = []
                    
                    for row in rows:
                        analysis_results = json.loads(row[4]) if row[4] else {}
                        sessions.append(SessionRecord(
                            session_id=row[0],
                            created_at=datetime.fromisoformat(row[1]),
                            audio_path=row[2],
                            video_path=row[3],
                            analysis_results=analysis_results
                        ))
                    
                    return sessions
                    
        except Exception as e:
            logger.error(f"Failed to list recent sessions: {e}")
            raise RuntimeError(f"Session listing failed: {e}")
    
    async def cleanup_old_files(self, days_old: int = 30) -> Dict[str, int]:
        """
        Clean up old files and sessions based on age.
        
        Args:
            days_old: Number of days after which files are considered old
        
        Returns:
            Dictionary with cleanup statistics
        
        Raises:
            RuntimeError: If cleanup operation fails
        """
        if not self._initialized:
            await self.initialize_database()
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        cleanup_stats = {
            "sessions_deleted": 0,
            "audio_files_deleted": 0,
            "video_files_deleted": 0,
            "bytes_freed": 0
        }
        
        try:
            async with self._db_semaphore:
                async with aiosqlite.connect(self.database_path) as db:
                    # Get old sessions
                    cursor = await db.execute("""
                        SELECT session_id, audio_path, video_path
                        FROM sessions 
                        WHERE created_at < ?
                    """, (cutoff_date,))
                    
                    old_sessions = await cursor.fetchall()
                    
                    for session_id, audio_path, video_path in old_sessions:
                        # Delete associated files
                        for file_path in [audio_path, video_path]:
                            if file_path:
                                full_path = self.storage_path / file_path
                                if await self.file_exists(str(full_path)):
                                    file_size = await self.get_file_size(str(full_path))
                                    await self.delete_file(str(full_path))
                                    cleanup_stats["bytes_freed"] += file_size
                                    
                                    if audio_path == file_path:
                                        cleanup_stats["audio_files_deleted"] += 1
                                    else:
                                        cleanup_stats["video_files_deleted"] += 1
                    
                    # Delete session records
                    await db.execute(
                        "DELETE FROM sessions WHERE created_at < ?",
                        (cutoff_date,)
                    )
                    cleanup_stats["sessions_deleted"] = len(old_sessions)
                    
                    await db.commit()
            
            logger.info(f"Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise RuntimeError(f"Cleanup operation failed: {e}")
    
    async def get_file_size(self, file_path: str) -> int:
        """
        Get file size in bytes.
        
        Args:
            file_path: Path to the file
        
        Returns:
            File size in bytes
        
        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file access fails
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            return path.stat().st_size
            
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get file size for {file_path}: {e}")
            raise IOError(f"File size check failed: {e}")
    
    async def file_exists(self, file_path: str) -> bool:
        """
        Check if file exists.
        
        Args:
            file_path: Path to the file
        
        Returns:
            True if file exists, False otherwise
        """
        try:
            return Path(file_path).exists()
        except Exception as e:
            logger.error(f"Failed to check file existence for {file_path}: {e}")
            return False
    
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete file safely.
        
        Args:
            file_path: Path to the file to delete
        
        Returns:
            True if file was deleted successfully, False otherwise
        """
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.info(f"File deleted: {file_path}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False
    
    async def get_storage_stats(self) -> Dict[str, any]:
        """
        Get storage usage statistics.
        
        Returns:
            Dictionary containing storage statistics
        """
        try:
            stats = {
                "total_sessions": 0,
                "audio_files_count": 0,
                "video_files_count": 0,
                "total_audio_size": 0,
                "total_video_size": 0,
                "total_size": 0,
                "oldest_session": None,
                "newest_session": None
            }
            
            # Database statistics
            if self._initialized:
                async with self._db_semaphore:
                    async with aiosqlite.connect(self.database_path) as db:
                        # Count total sessions
                        cursor = await db.execute("SELECT COUNT(*) FROM sessions")
                        row = await cursor.fetchone()
                        stats["total_sessions"] = row[0] if row else 0
                        
                        # Get oldest and newest sessions
                        cursor = await db.execute("""
                            SELECT MIN(created_at), MAX(created_at) FROM sessions
                        """)
                        row = await cursor.fetchone()
                        if row and row[0]:
                            stats["oldest_session"] = row[0]
                            stats["newest_session"] = row[1]
            
            # File system statistics
            for path_name, path_obj in [("audio", self.audio_path), ("video", self.video_path)]:
                if path_obj.exists():
                    for file_path in path_obj.rglob("*"):
                        if file_path.is_file():
                            file_size = file_path.stat().st_size
                            stats[f"total_{path_name}_size"] += file_size
                            stats[f"{path_name}_files_count"] += 1
            
            stats["total_size"] = stats["total_audio_size"] + stats["total_video_size"]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {"error": str(e)}
    
    async def update_session_file_paths(self, session_id: str, audio_path: Optional[str] = None, video_path: Optional[str] = None) -> None:
        """
        Update session with file paths after saving files.
        
        Args:
            session_id: Unique session identifier
            audio_path: Path to audio file (optional)
            video_path: Path to video file (optional)
        
        Raises:
            RuntimeError: If database operation fails
            ValueError: If session doesn't exist
        """
        if not self._initialized:
            await self.initialize_database()
        
        try:
            async with self._db_semaphore:
                async with aiosqlite.connect(self.database_path) as db:
                    # Check if session exists
                    cursor = await db.execute(
                        "SELECT session_id FROM sessions WHERE session_id = ?",
                        (session_id,)
                    )
                    if not await cursor.fetchone():
                        raise ValueError(f"Session {session_id} not found")
                    
                    # Build update query dynamically
                    updates = []
                    params = []
                    
                    if audio_path is not None:
                        updates.append("audio_path = ?")
                        params.append(audio_path)
                    
                    if video_path is not None:
                        updates.append("video_path = ?")
                        params.append(video_path)
                    
                    if updates:
                        params.append(session_id)
                        query = f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = ?"
                        await db.execute(query, params)
                        await db.commit()
                        
            logger.info(f"Session file paths updated: {session_id}")
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to update session file paths {session_id}: {e}")
            raise RuntimeError(f"Session file path update failed: {e}")


# Module-level instance for dependency injection
local_storage_service = LocalStorageService()


async def get_local_storage_service() -> LocalStorageService:
    """
    Dependency injection function for FastAPI.
    
    Returns:
        LocalStorageService instance
    """
    if not local_storage_service._initialized:
        await local_storage_service.initialize_database()
    
    return local_storage_service