import pandas as pd
import cv2
import numpy as np

def test_environment():
    print("🚀 System environment test initiated...")
    print("-" * 40)
    print(f"✅ Pandas loaded successfully | Version: {pd.__version__}")
    print(f"✅ OpenCV loaded successfully | Version: {cv2.__version__}")
    print(f"✅ Numpy  loaded successfully | Version: {np.__version__}")
    print("-" * 40)
    print("🎉 Member B environment configuration perfect! Ready to ingest FinTabNet data.")

if __name__ == "__main__":
    test_environment()