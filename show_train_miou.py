import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training loss and mIoU from CSV log.")
    parser.add_argument(
        "--file_name",
        type=str,
        default="training_log_cityscapes_1.csv",
        help="CSV file name or absolute path.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cityscapes",
        help="Dataset name.",
    )
    return parser.parse_args()


args = parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = args.file_name
if not os.path.isabs(csv_path):
    csv_path = os.path.join(script_dir, csv_path)

df = pd.read_csv(csv_path)

loss_col = 'train_loss' if 'train_loss' in df.columns else 'loss'
miou_col = 'val_mIoU' if 'val_mIoU' in df.columns else 'mIoU'

# Plot Loss vs Epoch
plt.figure(figsize=(12, 5))
plt.suptitle(args.dataset_name, fontsize=14, fontweight='bold')

plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['train_loss'], color='b', label='Train loss')
plt.plot(df['epoch'], df['val_loss'], color='r', label='Val loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot mIoU vs Epoch
plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['train_mIoU'], color='b', label='Train mIoU')
plt.plot(df['epoch'], df['val_mIoU'], color='r', label='Val mIoU')
plt.title('mIoU')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('mIoU')
plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()