import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from src.datasets.oxford_pets import get_dataloader, OxfordPetsRegression

# def main():
    # data_dir = "/home/namu/myspace/NAMU/datasets/oxford_pets"

    # print(f"\n>> Regression:")
    # reg_dataloader = get_dataloader(data_dir, "test", task="regression")
    # reg_batch = next(iter(reg_dataloader))
    # images = reg_batch["image"]
    # labels = reg_batch["label"]
    # coords = reg_batch["coord"]

    # print(f"Images: {images.shape}, {images.dtype}")
    # print(f"Labels: {labels.shape}, {labels.dtype}")
    # print(f"Coords: {coords.shape}, {coords.dtype}")
    # for coord in coords:
    #     print(coord)


def main():
    data_dir = "/home/namu/myspace/NAMU/datasets/oxford_pets"

    print(f"\n>> Regression:")
    dataset = OxfordPetsRegression(data_dir, "test", transform=None)
    print(f"Dataset length: {len(dataset)}")  # 🔍 이 값이 0이면 문제!

    if len(dataset) == 0:
        print("No samples loaded. Checking first few lines in test.txt...")
        split_file = os.path.join(data_dir, 'annotations', 'test.txt')
        with open(split_file) as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                filename = parts[0]
                xml_path = os.path.join(data_dir, 'annotations', 'xmls', f'{filename}.xml')
                exists = os.path.exists(xml_path)
                print(f"{filename}: xml exists = {exists}")

if __name__ == "__main__":
    main()
