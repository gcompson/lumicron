import os
import shutil
import sys
from pathlib import Path

def run_archive(target_name, external_mount="/Volumes/4TB_SSD"):
    """
    Moves heavy data to external SSD and symlinks back to internal.
    """
    internal_root = Path.home() / "Projects/UAP_Data" / target_name
    external_root = Path(external_mount) / "Lumicron_Archive" / target_name
    
    if not internal_root.exists():
        print(f"Error: Internal project {target_name} not found.")
        return

    if not Path(external_mount).exists():
        print(f"Error: External SSD not found at {external_mount}.")
        return

    # Folders to migrate
    heavy_folders = ["01_RAW", "02_FRAMES"]
    
    os.makedirs(external_root, exist_ok=True)
    
    for folder in heavy_folders:
        src = internal_root / folder
        dst = external_root / folder
        
        if src.is_symlink():
            print(f"Skipping {folder}: Already symlinked.")
            continue
            
        if src.exists():
            print(f"Migrating {folder} to external storage...")
            # Move the actual data
            shutil.move(str(src), str(dst))
            # Create the symbolic link
            os.symlink(dst, src)
            print(f"SUCCESS: {folder} is now linked to external SSD.")
        else:
            print(f"Notice: {folder} does not exist in {target_name}. Skipping.")

    print(f"\nArchive Complete for {target_name}. Internal SSD pressure relieved.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python archive.py <target_name>")
    else:
        run_archive(sys.argv[1])
