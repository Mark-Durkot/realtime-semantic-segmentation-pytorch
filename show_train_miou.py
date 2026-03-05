import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training loss and mIoU from CSV log.")
    parser.add_argument(
        "--file_name",
        type=str,
        default="training_log_vvd_1.csv",
        help="CSV file name or absolute path.",
    )
    return parser.parse_args()


args = parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = args.file_name
if not os.path.isabs(csv_path):
    csv_path = os.path.join(script_dir, csv_path)

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