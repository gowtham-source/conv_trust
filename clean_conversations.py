import os
import json
from pathlib import Path
import shutil

# Define paths
base_path = Path(r'c:\Users\gowth\Downloads\conv_trust')
conversations_dir = base_path / 'data' / 'conversations'
metadata_dir = base_path / 'data' / 'metadata'

# Ensure metadata directory exists
metadata_dir.mkdir(parents=True, exist_ok=True)

print(f"Processing files in: {conversations_dir}")
print(f"Saving metadata to: {metadata_dir}")

processed_count = 0
moved_count = 0
error_count = 0

# Iterate through files in the conversations directory
for file_path in conversations_dir.glob('*.json'):
    processed_count += 1
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check if 'turns' key exists and is an empty list
        if 'turns' in data and isinstance(data['turns'], list) and not data['turns']:
            # Check if 'metadata' key exists
            if 'metadata' in data:
                metadata = data['metadata']
                # Define the path for the new metadata file
                metadata_file_path = metadata_dir / file_path.name

                # Write the metadata to the new file
                with open(metadata_file_path, 'w', encoding='utf-8') as mf:
                    json.dump(metadata, mf, indent=2)

                print(f"  - Extracted metadata from '{file_path.name}' to '{metadata_file_path}'")
                moved_count +=1

                # Optional: Remove the original file after extracting metadata
                # Uncomment the next line if you want to delete the original file
                file_path.unlink()
                print(f"  - Removed original file: '{file_path.name}'")
            else:
                 print(f"  - Skipping '{file_path.name}': 'metadata' key not found.")
        # else:
        #     # Optional: Log files that are kept (non-empty turns)
        #     print(f"  - Keeping '{file_path.name}': 'turns' list is not empty.")

    except json.JSONDecodeError:
        print(f"  - Error: Could not decode JSON from '{file_path.name}'")
        error_count += 1
    except Exception as e:
        print(f"  - Error processing file '{file_path.name}': {e}")
        error_count += 1

print(f"\nProcessing complete.")
print(f"Total files processed: {processed_count}")
print(f"Metadata files created: {moved_count}")
print(f"Errors encountered: {error_count}")
