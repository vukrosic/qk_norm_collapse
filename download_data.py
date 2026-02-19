from datasets import load_dataset
import os

print('Downloading 1B Pretraining Data...')
# Load the dataset from Hugging Face
ds = load_dataset('vukrosic/blueberry-1B-pretrain')

# Ensure the directory exists
output_dir = 'processed_data/pretrain_1B'
os.makedirs(output_dir, exist_ok=True)

# Save to disk
print(f'Saving to {output_dir}...')
ds.save_to_disk(output_dir)
print('✅ Full Data Ready!')
