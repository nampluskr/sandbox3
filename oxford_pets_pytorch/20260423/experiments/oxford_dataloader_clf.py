import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from src.datasets.oxford_pets import get_dataloader

def main():
    data_dir = "/home/namu/myspace/NAMU/datasets/oxford_pets"

    print(f"\n>> Classification:")
    cls_dataloader = get_dataloader(data_dir, "train", task="classification")
    cls_batch = next(iter(cls_dataloader))
    images = cls_batch["image"]
    labels = cls_batch["label"]
    print(f"Image: {images.shape}, {images.dtype}")
    print(f"Label: {labels.shape}, {labels.dtype}")


if __name__ == "__main__":
    main()
