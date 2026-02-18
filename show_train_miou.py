import os
import pandas as pd
import matplotlib.pyplot as plt

print("Hello, World!")

# Define the path to the csv file (place training_log.csv in this script's directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "training_log.csv")

# Read the CSV file
df = pd.read_csv(csv_path)

# Plot Loss vs Epoch
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['loss'], color='b')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Plot mIoU vs Epoch
plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['mIoU'], color='g')
plt.title('mIoU vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('mIoU')
plt.grid(True)

plt.tight_layout()
plt.show()