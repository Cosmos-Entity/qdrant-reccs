import os
import json
import tarfile
import shutil
from pathlib import Path

def fix_segment_config(segment_path):
    """Fix the configuration in a segment file to use on_disk storage"""
    print(f"Processing segment: {os.path.basename(segment_path)}")
    
    # Create a temporary directory for modification
    temp_dir = Path("temp_segment")
    temp_dir.mkdir(exist_ok=True)
    
    # Extract the tar file
    with tarfile.open(segment_path, 'r') as tar:
        tar.extractall(temp_dir)
    
    # Fix the segment configuration
    segment_json_path = temp_dir / "snapshot" / "files" / "segment.json"
    if segment_json_path.exists():
        with open(segment_json_path, 'r') as f:
            config = json.load(f)
            print("Original config:", json.dumps(config, indent=2))
            
            # Replace 'mmap' with 'on_disk' in the configuration
            if isinstance(config, dict):
                def replace_mmap(obj):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if isinstance(v, str) and v == 'mmap':
                                obj[k] = 'on_disk'
                            elif isinstance(v, (dict, list)):
                                replace_mmap(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            if isinstance(item, (dict, list)):
                                replace_mmap(item)
                
                replace_mmap(config)
                print("\nModified config:", json.dumps(config, indent=2))
                
                # Save the modified configuration
                with open(segment_json_path, 'w') as f:
                    json.dump(config, f, indent=2)
    
    # Create new tar file
    backup_path = Path(str(segment_path) + '.backup')
    if segment_path.exists():
        shutil.move(segment_path, backup_path)
    
    with tarfile.open(segment_path, 'w') as tar:
        for file_path in temp_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(temp_dir)
                tar.add(file_path, arcname=arcname)
    
    # Clean up
    shutil.rmtree(temp_dir)
    print(f"Processed {os.path.basename(segment_path)}")

def main():
    # Process both collections
    collections = ['extracted_users_data', 'extracted_elements_data']
    
    for collection in collections:
        print(f"\nProcessing collection: {collection}")
        segments_dir = Path(collection) / "0" / "segments"
        
        if not segments_dir.exists():
            print(f"Directory not found: {segments_dir}")
            continue
            
        for segment_file in segments_dir.glob('*.tar'):
            fix_segment_config(segment_file)
            
        # Also fix the main collection config
        config_path = Path(collection) / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                print("\nOriginal collection config:", json.dumps(config, indent=2))
                
                # Update storage type in collection config
                if 'params' in config:
                    if config['params'].get('on_disk', None) == 'mmap':
                        config['params']['on_disk'] = True
                    
                print("\nModified collection config:", json.dumps(config, indent=2))
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)

if __name__ == "__main__":
    main() 