
### `src/utils.py`
```python
import shutil

def random_sample_files(source_dir, dest_dir, sampling_prob):
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    if len(files) == 0:
        return []
    
    indices = np.random.permutation(len(files))
    num_select = int(len(files) * sampling_prob)
    selected_indices = indices[:num_select]
    
    os.makedirs(dest_dir, exist_ok=True)
    moved_files = []
    for i in selected_indices:
        file_name = files[i]
        src_path = os.path.join(source_dir, file_name)
        dst_path = os.path.join(dest_dir, file_name)
        shutil.move(src_path, dst_path)
        moved_files.append(file_name)
    
    return moved_files
```

```python
from src.utils import random_sample_files

file_list = random_sample_files(
    source_dir="E:\\datasets\\smartdoc_2015\\smart_doc_extracted\\images",
    dest_dir="E:\\datasets\\smartdoc_2015\\smart_doc_extracted_selected\\images",
    sampling_prob=0.2   # 19876
)
print(len(file_list))
```

```python
import pandas as pd
import os

csv_path = "E:\\datasets\\smartdoc_2015\\smart_doc_extracted\\frame_data.csv"
df = pd.read_csv(csv_path)

data_dir = "E:\\datasets\\smartdoc_2015\\smart_doc_extracted_selected\\images"
filenames = set(os.listdir(data_dir))

filtered_df = df[df['frame_filename'].isin(filenames)]
output_csv_path = "E:\\datasets\\smartdoc_2015\\smart_doc_extracted_selected\\frame_data_selected.csv"
filtered_df.to_csv(output_csv_path, index=False)

print(f"Selected: {len(filtered_df)}")  # 
```

### `anotations/create_anotations_smartdoc_selected.ipynb`
```python
data_dir = "/home/namu/myspace/NAMU/datasets/smart_doc_extracted_selected"
csv_path = os.path.join(data_dir, "frame_data_selected.csv")
create_anotations(csv_path, "anotations_smartdoc_selected.csv")

df = pd.read_csv("anotations_smartdoc_selected.csv")
df
```

```python
from filter import filter_valid_images

data_dir = "/home/namu/myspace/NAMU/datasets/smart_doc_extracted_selected"
image_dir = os.path.join(data_dir, "images")
df = filter_valid_images("anotations_smartdoc_selected.csv", image_dir)
df.to_csv("anotations_smartdoc_selected_filtered.csv", index=False)
```

### `configs/default.yaml`
```yaml
# Paths
backbone_dir: /home/namu/myspace/NAMU/backbones
dataset_dir: /home/namu/myspace/NAMU/datasets
root_dir: /home/namu/myspace/NAMU/clones/polygon_regression

# Datasets
oxford:
  image_dir: ${dataset_dir}/oxford_pets/images
  csv_path: ${root_dir}/anotations/anotations_oxford_filtered.csv
  sampling: 1.0
  weight: 3.0

midv2020:
  image_dir: ${dataset_dir}/midv2020_processed/images
  csv_path: ${root_dir}/anotations/anotations_midv2020_filtered.csv
  sampling: 0.1
  weight: 2.0

smartdoc:
  image_dir: ${dataset_dir}/smart_doc_extracted_selected/images
  csv_path: ${root_dir}/anotations/anotations_smartdoc_selected_filtered.csv
  sampling: 0.5
  weight: 1.0
```

## Train / Test Split

```python
def load_image(img_path):
    print(f"Load {img_path}")
    return np.array(Image.open(img_path).convert("RGB"))

def img_info(img):
    print(f"img: {img.shape}, [{img.min()}, {img.max()}]")

def show_image(img):
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.show()

def load_coords(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    points = {"p1": None, "p2": None, "p3": None, "p4": None}
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label in points:
            x = float(shape["points"][0][0])
            y = float(shape["points"][0][1])
            points[label] = [x, y]

    return points["p1"] + points["p2"] + points["p3"] + points["p4"]
```

```python
# train
train_img_dir = "E:\\fringe_images\\train_images"
train_json_dir = "E:\\fringe_images\\train_labels"

train_img_paths = glob(os.path.join(train_img_dir, "data*.*"))
train_json_paths = glob(os.path.join(train_json_dir, "data*.json"))

print(f"train images: {len(train_img_paths)}")
print(f"train labels: {len(train_json_paths)}")

img_paths = train_img_paths
img_names = [os.path.basename(path) for path in img_paths]
json_paths = train_json_paths

coords_list = []
for json_path in json_paths:
    coords = load_coords(json_path)
    coords_list.append(coords)
    
df = pd.DataFrame(coords_list, columns=["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"])
df.insert(0, "image_name", img_names)
df.to_csv("E:\\fringe_images\\anotations_data_train.csv")
df
```

```python
# test
test_img_dir = "E:\\fringe_images\\test_images"
test_json_dir = "E:\\fringe_images\\test_labels"

test_img_paths = glob(os.path.join(test_img_dir, "data*.*"))
test_json_paths = glob(os.path.join(test_json_dir, "data*.json"))

print(f"train images: {len(test_img_paths)}")
print(f"train labels: {len(test_json_paths)}")

img_paths = test_img_paths
img_names = [os.path.basename(path) for path in img_paths]
json_paths = test_json_paths

coords_list = []
for json_path in json_paths:
    coords = load_coords(json_path)
    coords_list.append(coords)
    
df = pd.DataFrame(coords_list, columns=["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"])
df.insert(0, "image_name", img_names)
df.to_csv("E:\\fringe_images\\anotations_data_test.csv")
df
```

## Base Dataset / Dataloader (with transform)

```python
class BaseDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        df = pd.read_csv(csv_path)
        self.samples = []
        for _, row in df.iterrows():
            self.samples.append({
                "image_path": os.path.join(image_dir, row["image_name"]),
                "bbox": [
                    row["x1"], row["y1"],  # tl (top left)
                    row["x2"], row["y2"],  # tr (top right)
                    row["x3"], row["y3"],  # br (bottom right)
                    row["x4"], row["y4"]   # bl (bottom left)
                ]
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(self.image_dir, sample["image_path"])
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        image = tv_tensors.Image(image)

        bbox = sample["bbox"]
        coords = torch.zeros((2, 4), dtype=torch.float32)

        if len(bbox) == 8:
            coords[0] = torch.tensor(bbox[:4], dtype=torch.float32)
            coords[1] = torch.tensor(bbox[4:], dtype=torch.float32)

        if self.transform:
            boxes = tv_tensors.BoundingBoxes(coords, format="XYXY", canvas_size=(h, w))
            image, coords = self.transform(image, boxes)

        coords = sort_clockwise(coords.view(-1))
        img_h, img_w = image.shape[-2:]
        coords_norm = coords / torch.tensor([img_w, img_h] * 4, dtype=torch.float32)
        return {
            "image": image,
            "coord": coords,
            "coord_norm": coords_norm,
        }


class SplitDataset(BaseDataset):
    def __init__(self, split, image_dir, csv_path, transform=None,
                 sampling=1.0, test_size=0.2, seed=42):
        if split not in ["train", "test"]:
            raise ValueError(f"split must be 'train' or 'test': {split}")

        self.image_dir = image_dir
        self.transform = transform

        df = pd.read_csv(csv_path)
        coord_cols = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
        df = df[
            df[coord_cols].notna().all(axis=1) &
            (df[coord_cols] > 0).all(axis=1) &
            (df[coord_cols] < 1e6).all(axis=1) &
            (~df[coord_cols].isin([float('inf'), float('-inf')]).any(axis=1))
        ].reset_index(drop=True)
        df = df.sample(frac=sampling, random_state=seed).reset_index(drop=True)

        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
        df = train_df if split == "train" else test_df
        df = df.reset_index(drop=True)

        self.samples = []
        for _, row in df.iterrows():
            self.samples.append({
                "image_path": os.path.join(image_dir, row["image_name"]),
                "bbox": [
                    row["x1"], row["y1"],  # tl (top left)
                    row["x2"], row["y2"],  # tr (top right)
                    row["x3"], row["y3"],  # br (bottom right)
                    row["x4"], row["y4"]   # bl (bottom left)
                ]
            })
```

```python
def get_base_dataloader(image_dir, csv_path, transform, 
    batch_size=16, shuffle=True, drop_last=True, **kwargs):
    dataset = BaseDataset(image_dir, csv_path, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=8,
        # persistent_workers=(split == "train"),
        # prefetch_factor=2,
    )


def get_split_dataloader(split, image_dir, csv_path, sampling=1.0, batch_size=16, image_size=256,
                   test_size=0.2, seed=42, **kwargs):
    pass

def get_combined_dataloader(dataset_configs, split, batch_size=16, image_size=256, **kwargs):
    pass
```
