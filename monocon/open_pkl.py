import pickle

# Specify the path to your pickle file
pickle_file_path = '/home/matan/Projects/MonoCon/outputs/pretrained_results/output.pkl'  # Replace with your actual file path

# Open the pickle file in binary read mode ('rb')
with open(pickle_file_path, 'rb') as file:
    # Load the contents of the pickle file
    data = pickle.load(file)

# Now, `data` contains the Python object stored in the pickle file
print(data)