import pickle

# --- CORRECTED: Use a raw string (r"...") ---
MAPPING_PATH = r"C:/Users/admin/Desktop/Chess_project/models/move_to_int"

try:
    with open(MAPPING_PATH, "rb") as f:
        move_to_int = pickle.load(f)

    num_classes_original = len(move_to_int)
    print(f"The value for NUM_CLASSES_ORIGINAL is: {num_classes_original}")

except FileNotFoundError:
    print(f"Error: Could not find the mapping file at '{MAPPING_PATH}'.")
    print("Please ensure the path is correct and the file exists.")
except Exception as e:
    print(f"An error occurred loading the file: {e}")