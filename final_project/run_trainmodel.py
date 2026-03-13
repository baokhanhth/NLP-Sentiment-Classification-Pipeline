import os
from src.train import train_model
from src.config import MODEL_DIR

if __name__ == "__main__":
    build_temp_dir = os.path.join(MODEL_DIR, "build_temp")

    if not os.path.exists(build_temp_dir):
        raise FileNotFoundError(
            f"Folder build_temp không tồn tại: {build_temp_dir}\n"
            f"Bạn cần chạy build_model.py trước!"
        )

    print("===== RUN TRAIN =====")
    train_model(build_temp_dir)
