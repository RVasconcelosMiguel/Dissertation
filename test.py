import os

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path for the outputs folder
outputs_dir = os.path.join(current_dir, "outputs")

# Create the outputs directory if it doesn't exist
os.makedirs(outputs_dir, exist_ok=True)