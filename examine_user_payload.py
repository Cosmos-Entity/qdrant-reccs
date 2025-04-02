import os
import tarfile
import json
import struct
from collections import defaultdict

def examine_payload(segment_path, users_found, field_stats):
    """Examine the payload data in a segment file"""
    print(f"\nExamining payload in: {os.path.basename(segment_path)}")
    
    with tarfile.open(segment_path, 'r') as tar:
        # Try to find and read payload data
        payload_files = [m for m in tar.getmembers() if 'payload_storage/page_0.dat' in m.name]
        for payload_file in payload_files:
            print(f"\nReading payload file: {payload_file.name}")
            f = tar.extractfile(payload_file)
            if f:
                data = f.read()
                print(f"Payload size: {len(data)} bytes")
                
                # Try to decode some of the payload data
                try:
                    # Skip header (first 16 bytes)
                    payload_data = data[16:]
                    # Try to find JSON-like structures
                    start = 0
                    while start < len(payload_data) and users_found[0] < 100:
                        # Look for JSON start
                        try:
                            json_start = payload_data.index(b'{', start)
                            # Look for matching closing brace
                            brace_count = 1
                            pos = json_start + 1
                            while brace_count > 0 and pos < len(payload_data):
                                if payload_data[pos] == ord('{'):
                                    brace_count += 1
                                elif payload_data[pos] == ord('}'):
                                    brace_count -= 1
                                pos += 1
                            
                            if brace_count == 0:
                                json_data = payload_data[json_start:pos]
                                try:
                                    # Try to decode and parse JSON
                                    decoded = json.loads(json_data.decode('utf-8'))
                                    if 'user_id' in decoded:
                                        users_found[0] += 1
                                        print(f"\nUser {users_found[0]}:")
                                        print(json.dumps(decoded, indent=2))
                                        
                                        # Track all fields and their types
                                        for field, value in decoded.items():
                                            field_stats[field]['count'] += 1
                                            value_type = type(value).__name__
                                            field_stats[field]['types'].add(value_type)
                                            if isinstance(value, (list, dict)):
                                                field_stats[field]['example'] = value
                                            else:
                                                field_stats[field]['example'] = str(value)
                                except:
                                    pass
                                start = pos
                            else:
                                start = json_start + 1
                        except ValueError:
                            break
                except Exception as e:
                    print(f"Error reading payload data: {e}")
        
        # Also try to read the db_backup files which might contain metadata
        db_files = [m for m in tar.getmembers() if 'db_backup/shared_checksum' in m.name and m.name.endswith('.sst')]
        for db_file in db_files[:5]:  # Only check first 5 files to save time
            print(f"\nChecking database file: {db_file.name}")
            f = tar.extractfile(db_file)
            if f:
                data = f.read()
                # Try to find JSON-like structures
                try:
                    start = 0
                    found = 0
                    while start < len(data) and found < 5:  # Look for up to 5 JSON objects
                        try:
                            json_start = data.index(b'{', start)
                            # Look for matching closing brace
                            brace_count = 1
                            pos = json_start + 1
                            while brace_count > 0 and pos < len(data):
                                if data[pos] == ord('{'):
                                    brace_count += 1
                                elif data[pos] == ord('}'):
                                    brace_count -= 1
                                pos += 1
                            
                            if brace_count == 0:
                                json_data = data[json_start:pos]
                                try:
                                    # Try to decode and parse JSON
                                    decoded = json.loads(json_data.decode('utf-8'))
                                    print("\nFound metadata:")
                                    print(json.dumps(decoded, indent=2))
                                    found += 1
                                    
                                    # Track metadata fields
                                    for field, value in decoded.items():
                                        field_stats[f"metadata_{field}"]['count'] += 1
                                        value_type = type(value).__name__
                                        field_stats[f"metadata_{field}"]['types'].add(value_type)
                                        if isinstance(value, (list, dict)):
                                            field_stats[f"metadata_{field}"]['example'] = value
                                        else:
                                            field_stats[f"metadata_{field}"]['example'] = str(value)
                                except:
                                    pass
                                start = pos
                            else:
                                start = json_start + 1
                        except ValueError:
                            break
                except Exception as e:
                    print(f"Error reading database file: {e}")

def main():
    # Path to the segments directory
    segments_dir = "extracted_users_data/0/segments"
    
    # Counter for users found (using list to make it mutable)
    users_found = [0]
    
    # Track field statistics
    field_stats = defaultdict(lambda: {'count': 0, 'types': set(), 'example': None})
    
    # Examine each segment file until we find 100 users
    for filename in os.listdir(segments_dir):
        if filename.endswith('.tar'):
            segment_path = os.path.join(segments_dir, filename)
            examine_payload(segment_path, users_found, field_stats)
            if users_found[0] >= 100:
                break

    print(f"\nTotal users found: {users_found[0]}")
    
    # Print field statistics
    print("\nField Statistics:")
    print("=" * 50)
    for field, stats in field_stats.items():
        print(f"\nField: {field}")
        print(f"Count: {stats['count']}")
        print(f"Types: {', '.join(stats['types'])}")
        print(f"Example: {stats['example']}")

if __name__ == "__main__":
    main() 