import os

# Create output folder
current_dir = os.path.dirname(os.path.abspath(__file__))
outputs_dir_1 = os.path.join(current_dir, "outputs/head/results")
os.makedirs(outputs_dir_1, exist_ok=True)
outputs_dir_2 = os.path.join(current_dir, "outputs/fine/results")
os.makedirs(outputs_dir_2, exist_ok=True)