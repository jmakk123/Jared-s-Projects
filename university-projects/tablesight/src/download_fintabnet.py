import os
from datasets import load_dataset

def download_and_prepare_dataset():
    os.makedirs('data/raw', exist_ok=True)
    output_path = "data/raw/FinTabNet_train.jsonl"
    
    print("🚀 Fetching FinTabNet and generating unique IDs...")
    
    try:
        dataset = load_dataset("apoidea/fintabnet-html", "en", split="train")
        
        # 💡 THE FIX: Add a unique ID based on row index
        print("🆔 Adding unique IDs to each row...")
        dataset = dataset.map(lambda ex, i: {"imgid": f"fintabnet_{i:06d}", **ex}, with_indices=True)
        
        print("🧹 Removing binary image data for text export...")
        if "image" in dataset.column_names:
            dataset = dataset.remove_columns("image")
            
        print(f"💾 Saving to {output_path}...")
        dataset.to_json(output_path, orient="records", lines=True, force_ascii=False)
        
        print("-" * 50)
        print("🎉 Dataset successfully saved with custom imgid!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    download_and_prepare_dataset()
