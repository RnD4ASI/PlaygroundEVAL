import json
import os

def load_json(filepath: str) -> dict:
    """
    Loads a JSON file from the given filepath.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading {filepath}: {e}")
        raise

def save_json(data: dict, filepath: str, indent: int = 4):
    """
    Saves a dictionary to a JSON file.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
        print(f"Data successfully saved to {filepath}")
    except Exception as e:
        print(f"An unexpected error occurred while saving to {filepath}: {e}")
        raise

if __name__ == '__main__':
    # Test functions
    test_dir = "_test_data_file_utils"
    os.makedirs(test_dir, exist_ok=True)

    sample_data = {"name": "Test JSON", "version": 1, "items": [1, 2, 3]}
    test_filepath = os.path.join(test_dir, "test.json")

    print(f"Attempting to save data to {test_filepath}")
    save_json(sample_data, test_filepath)

    print(f"Attempting to load data from {test_filepath}")
    loaded_data = load_json(test_filepath)

    assert loaded_data == sample_data
    print("JSON load/save test successful.")

    # Test non-existent file
    try:
        load_json(os.path.join(test_dir,"non_existent.json"))
    except FileNotFoundError:
        print("Correctly handled FileNotFoundError for load_json.")

    # Clean up
    os.remove(test_filepath)
    os.rmdir(test_dir)
    print(f"Cleaned up test directory and file: {test_dir}")
