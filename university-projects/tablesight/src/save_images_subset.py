import os
import json
from datasets import load_dataset
from tqdm import tqdm

def save_images_for_hard_examples():
    subset_path = "data/processed/hard_examples_subset.jsonl"
    image_output_dir = "data/processed/images"
    os.makedirs(image_output_dir, exist_ok=True)

    print("📖 Loading hard examples manifest...")
    with open(subset_path, 'r') as f:
        target_filenames = {json.loads(line)['imgid'] for line in f}
    
    print(f"🎯 Targeted images: {len(target_filenames)}")

    print("⏳ Accessing cached dataset...")
    # Add index to the dataset just like we did in download script
    dataset = load_dataset("apoidea/fintabnet-html", "en", split="train")
    dataset = dataset.map(lambda ex, i: {"imgid": f"fintabnet_{i:06d}", **ex}, with_indices=True)

    print("🖼️  Extracting and saving images...")
    for sample in tqdm(dataset):
        if sample['imgid'] in target_filenames:
            img_path = os.path.join(image_output_dir, f"{sample['imgid']}.png")
            sample['image'].save(img_path)

    print("-" * 50)
    print(f"✅ Subset images saved successfully to: {image_output_dir}")

if __name__ == "__main__":
    save_images_for_hard_examples()
