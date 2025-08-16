import numpy as np

# Function to read data from a file
def read_data(file_path):
    with open(file_path, 'r') as file:
        # Assuming the values are space-separated on each line
        data = file.read().strip().split()
        return np.array(data, dtype=float)

# Specify the path to your data file
file_path = '/home/xln/PycharmProjects/PredictLeafNode/input/Training_data/sift1M_learn/leafsize10K/pred_probs.txt'  # Replace with your file path

# Load the data from the file
data = read_data(file_path)

# Reshaping the data into rows (assuming the data is to be divided into rows of length 10 for this example)
row_length = 10000  # Adjust this based on your data's required row length
data_reshaped = data.reshape( row_length,-1)

# Check dimensions and sum of each row
for i, row in enumerate(data_reshaped):
    row_sum = np.sum(row)
    print(f"Row {i + 1}: Dimension = {row.shape}, Sum = {row_sum:.6f}, Close to 1? {'Yes' if np.isclose(row_sum, 1) else 'No'}")
