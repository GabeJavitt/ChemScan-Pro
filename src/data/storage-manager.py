"""
File system storage manager for chemical label images.
"""
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import uuid


class StorageManager:
    """Manages file system storage for chemical label images."""
    
    def __init__(self, base_path: str = "images"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_image(self, image_path: str, original_filename: str) -> Tuple[str, dict]:
        """
        Save an image file to organized storage.
        
        Args:
            image_path: Path to the source image file
            original_filename: Original filename of the image
            
        Returns:
            Tuple of (stored_file_path, metadata_dict)
        """
        # Create year/month directory structure
        now = datetime.now()
        year_month = now.strftime("%Y/%m")
        storage_dir = self.base_path / year_month
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        file_extension = Path(original_filename).suffix.lower()
        unique_id = str(uuid.uuid4())[:8]
        new_filename = f"chemical_{unique_id}{file_extension}"
        
        # Full path for stored file
        stored_path = storage_dir / new_filename
        
        # Copy file to storage location
        shutil.copy2(image_path, stored_path)
        
        # Get file metadata
        file_size = stored_path.stat().st_size
        relative_path = str(stored_path.relative_to(self.base_path))
        
        metadata = {
            'stored_path': str(stored_path),
            'relative_path': relative_path,
            'original_filename': original_filename,
            'new_filename': new_filename,
            'file_size': file_size,
            'storage_date': now.isoformat()
        }
        
        return str(stored_path), metadata
    
    def get_image_path(self, relative_path: str) -> Optional[str]:
        """
        Get full path to stored image from relative path.
        
        Args:
            relative_path: Relative path stored in database
            
        Returns:
            Full path to image file, or None if not found
        """
        full_path = self.base_path / relative_path
        return str(full_path) if full_path.exists() else None
    
    def delete_image(self, relative_path: str) -> bool:
        """
        Delete an image file from storage.
        
        Args:
            relative_path: Relative path to the image file
            
        Returns:
            True if file was deleted, False if not found
        """
        full_path = self.base_path / relative_path
        try:
            if full_path.exists():
                full_path.unlink()
                return True
            return False
        except Exception as e:
            print(f"Error deleting file {full_path}: {e}")
            return False
    
    def get_storage_stats(self) -> dict:
        """Get storage statistics."""
        total_files = 0
        total_size = 0
        
        for file_path in self.base_path.rglob("*"):
            if file_path.is_file():
                total_files += 1
                total_size += file_path.stat().st_size
        
        return {
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'base_path': str(self.base_path.absolute())
        }


# Example usage
if __name__ == "__main__":
    storage = StorageManager()
    
    # Example: Save a test image
    # stored_path, metadata = storage.save_image("test_label.jpg", "sodium_chloride.jpg")
    # print(f"Stored at: {stored_path}")
    # print(f"Metadata: {metadata}")
    
    # Get storage stats
    stats = storage.get_storage_stats()
    print(f"Storage stats: {stats}")