import json
import os
import time

def extract_hierarchical_tables(input_path, output_path):
    print(f"🔍 Starting extraction from: {input_path}")
    print(f"💾 Saving hard examples to: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    start_time = time.time()
    total_lines = 0
    extracted_lines = 0

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        # Streaming: read line by line, no memory overload
        for line in f_in:
            total_lines += 1
            try:
                data = json.loads(line)
                
                # 💡 THE FIX: Use the new schema key "html_table"
                html_str = data.get('html_table', '').lower()

                # Logic: Check for complex attributes > 1
                has_colspan = 'colspan="' in html_str and 'colspan="1"' not in html_str
                has_rowspan = 'rowspan="' in html_str and 'rowspan="1"' not in html_str

                if has_colspan or has_rowspan:
                    f_out.write(line)
                    extracted_lines += 1

                # Print progress for massive files
                if total_lines % 10000 == 0:
                    print(f"⏳ Processed {total_lines} lines... Found {extracted_lines} hard examples.")

            except json.JSONDecodeError:
                continue

    elapsed_time = time.time() - start_time
    print("-" * 50)
    print("✅ Extraction Complete!")
    print(f"📄 Total tables processed: {total_lines}")
    print(f"🎯 Hard examples extracted: {extracted_lines} ({(extracted_lines/total_lines)*100 if total_lines > 0 else 0:.2f}%)")
    print(f"⏱️  Time elapsed: {elapsed_time:.2f} seconds")
    print(f"📁 Output saved to: {output_path}")

if __name__ == "__main__":
    INPUT_FILE = "data/raw/FinTabNet_train.jsonl"
    OUTPUT_FILE = "data/processed/hard_examples_subset.jsonl"
    
    extract_hierarchical_tables(INPUT_FILE, OUTPUT_FILE)
