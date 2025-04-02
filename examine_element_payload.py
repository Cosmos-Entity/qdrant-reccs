import os
import json
import tarfile
from collections import defaultdict

def examine_payload(tar_path, elements_found, field_stats):
    try:
        with tarfile.open(tar_path, 'r') as tar:
            # Look for payload storage files
            payload_files = [f for f in tar.getnames() if 'payload_storage/page_' in f and f.endswith('.dat')]
            
            for payload_file in payload_files:
                print(f"\nReading payload file: {payload_file}")
                f = tar.extractfile(payload_file)
                if f is None:
                    continue
                    
                data = f.read()
                print(f"Payload size: {len(data)} bytes")
                
                # Process the payload data
                try:
                    # Split by null bytes and try to decode each chunk
                    chunks = data.split(b'\x00')
                    for chunk in chunks:
                        if not chunk.strip():
                            continue
                            
                        try:
                            # Try to decode as JSON
                            decoded = json.loads(chunk)
                            
                            # Check if this matches our expected schema
                            if isinstance(decoded, dict) and 'result' in decoded:
                                # Process each element in the result array
                                for element in decoded['result']:
                                    if not isinstance(element, dict) or 'id' not in element or 'payload' not in element:
                                        continue
                                        
                                    # Track field statistics for the payload
                                    payload = element['payload']
                                    for field, value in payload.items():
                                        field_type = type(value).__name__
                                        field_stats[field]['count'] += 1
                                        if field_type not in field_stats[field]['types']:
                                            field_stats[field]['types'].add(field_type)
                                        if len(field_stats[field]['examples']) < 3:  # Keep up to 3 examples
                                            field_stats[field]['examples'].append(value)
                                    
                                    elements_found[0] += 1
                                    print(f"\nComplete Element {elements_found[0]}:")
                                    print(json.dumps(element, indent=2))
                                    
                                    # Also check for color analysis fields
                                    if 'dominant_colors' in payload:
                                        print("\nColor Analysis:")
                                        print(json.dumps(payload['dominant_colors'], indent=2))
                                    
                                    # Check for safety scores
                                    safety_scores = {k: v for k, v in payload.items() if 'Score' in k}
                                    if safety_scores:
                                        print("\nSafety Scores:")
                                        print(json.dumps(safety_scores, indent=2))
                                    
                                    if elements_found[0] >= 100:
                                        return
                                
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            print(f"Error processing chunk: {str(e)}")
                            continue
                except Exception as e:
                    print(f"Error processing payload data: {str(e)}")
                    continue
    except Exception as e:
        print(f"Error opening tar file {tar_path}: {str(e)}")
        return

def main():
    elements_found = [0]  # Using list to make it mutable
    field_stats = defaultdict(lambda: {'count': 0, 'types': set(), 'examples': []})
    
    # Process segments directory
    segments_dir = "extracted_elements_data/0/segments"
    print(f"\nExamining segments in: {segments_dir}")
    
    # Look for .tar files in the segments directory
    for file in os.listdir(segments_dir):
        if file.endswith('.tar') and not file.endswith('.tar.backup'):
            tar_path = os.path.join(segments_dir, file)
            print(f"\nProcessing segment file: {file}")
            examine_payload(tar_path, elements_found, field_stats)
            
            if elements_found[0] >= 100:
                break
    
    print(f"\nTotal complete elements found: {elements_found[0]}")
    
    print("\nField Statistics:")
    print("=" * 50 + "\n")
    
    # Sort fields by count
    sorted_fields = sorted(field_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    for field, stats in sorted_fields:
        print(f"Field: {field}")
        print(f"Count: {stats['count']}")
        print(f"Types: {', '.join(stats['types'])}")
        print("Examples:")
        for example in stats['examples']:
            if isinstance(example, (list, dict)):
                print(f"  {json.dumps(example)[:100]}...")
            else:
                print(f"  {example}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main() 