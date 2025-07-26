"""
Test script for LocalStorageService functionality.

This script demonstrates and tests the core functionality of the LocalStorageService
including database operations, file storage, and session management.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from services.local_storage import LocalStorageService


async def test_local_storage_service():
    """Test the LocalStorageService functionality."""
    print("🧪 Testing LocalStorageService...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Override environment variables for testing
        os.environ["LOCAL_STORAGE_PATH"] = str(temp_path)
        os.environ["AUDIO_STORAGE_PATH"] = str(temp_path / "audio")
        os.environ["VIDEO_STORAGE_PATH"] = str(temp_path / "video")
        os.environ["DATABASE_PATH"] = str(temp_path / "test_sessions.db")
        
        # Create new service instance with test configuration
        service = LocalStorageService()
        
        print("✅ Step 1: Initialize database")
        await service.initialize_database()
        print(f"   Database created at: {service.database_path}")
        
        print("\n✅ Step 2: Create test session")
        session_id = "test-session-001"
        session = await service.create_session(session_id)
        print(f"   Session created: {session.session_id}")
        print(f"   Created at: {session.created_at}")
        
        print("\n✅ Step 3: Save test audio file")
        # Create dummy audio data
        audio_data = b"fake_audio_data_for_testing" * 100
        audio_path = await service.save_audio_file(audio_data, session_id, "wav")
        print(f"   Audio saved to: {audio_path}")
        
        # Update session with audio path
        await service.update_session_file_paths(session_id, audio_path=audio_path)
        
        print("\n✅ Step 4: Save test video file")
        # Create dummy video data
        video_data = b"fake_video_data_for_testing" * 200
        video_path = await service.save_video_file(video_data, session_id, "mp4")
        print(f"   Video saved to: {video_path}")
        
        # Update session with video path
        await service.update_session_file_paths(session_id, video_path=video_path)
        
        print("\n✅ Step 5: Update session with analysis results")
        analysis_results = {
            "facial_emotion": {
                "dominant_emotion": "happy",
                "confidence": 0.85,
                "timestamp": datetime.now().timestamp()
            },
            "audio_emotion": {
                "dominant_emotion": "neutral",
                "confidence": 0.72,
                "timestamp": datetime.now().timestamp()
            },
            "risk_assessment": {
                "risk_level": "low",
                "risk_score": 0.15,
                "confidence": 0.80
            }
        }
        await service.update_session_analysis(session_id, analysis_results)
        print("   Analysis results updated")
        
        print("\n✅ Step 6: Retrieve session data")
        retrieved_session = await service.get_session(session_id)
        if retrieved_session:
            print(f"   Retrieved session: {retrieved_session.session_id}")
            print(f"   Audio path: {retrieved_session.audio_path}")
            print(f"   Video path: {retrieved_session.video_path}")
            print(f"   Analysis results keys: {list(retrieved_session.analysis_results.keys())}")
        
        print("\n✅ Step 7: Test utility methods")
        # Test file operations
        full_audio_path = service.storage_path / audio_path
        file_exists = await service.file_exists(str(full_audio_path))
        file_size = await service.get_file_size(str(full_audio_path))
        print(f"   Audio file exists: {file_exists}")
        print(f"   Audio file size: {file_size} bytes")
        
        print("\n✅ Step 8: Get storage statistics")
        stats = await service.get_storage_stats()
        print("   Storage statistics:")
        for key, value in stats.items():
            print(f"     • {key}: {value}")
        
        print("\n✅ Step 9: List recent sessions")
        recent_sessions = await service.list_recent_sessions(limit=5)
        print(f"   Found {len(recent_sessions)} recent sessions")
        for session in recent_sessions:
            print(f"     • {session.session_id} (created: {session.created_at})")
        
        print("\n✅ Step 10: Test error handling")
        try:
            # Try to create duplicate session
            await service.create_session(session_id)
        except ValueError as e:
            print(f"   ✓ Caught expected error: {e}")
        
        try:
            # Try unsupported format
            await service.save_audio_file(b"test", session_id, "xyz")
        except ValueError as e:
            print(f"   ✓ Caught expected error: {e}")
        
        print("\n✅ Step 11: Test file deletion")
        deleted = await service.delete_file(str(full_audio_path))
        print(f"   Audio file deleted: {deleted}")
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"   Test directory: {temp_dir}")
        print(f"   Database file: {service.database_path}")


async def main():
    """Main test execution function."""
    try:
        await test_local_storage_service()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())