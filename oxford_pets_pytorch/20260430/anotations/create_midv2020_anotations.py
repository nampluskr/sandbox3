import os
import pandas as pd
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, footprint_rectangle
from PIL import Image
from tqdm import tqdm

def get_coords(mask_path):
    mask = np.array(Image.open(mask_path).convert("L"))
    H, W = mask.shape
    if mask.max() > 1:
        binary = mask > 128
    else:
        binary = mask > 0 

    selem = footprint_rectangle((5, 5))
    binary = binary_closing(binary, selem)
    labeled = label(binary)
    regions = regionprops(labeled)

    if not regions:
        return []

    largest = max(regions, key=lambda r: r.area)
    region_mask = (labeled == largest.label)

    ys, xs = np.where(region_mask)
    if len(xs) == 0:
        return []

    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    sums = pts[:, 0] + pts[:, 1] 
    diffs = pts[:, 1] - pts[:, 0]

    tl = pts[np.argmin(sums)]
    br = pts[np.argmax(sums)]
    tr = pts[np.argmin(diffs)]
    bl = pts[np.argmax(diffs)]

    def polygon_area(points):
        x, y = points[:, 0], points[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    quad = np.array([tl, tr, br, bl])
    poly_area = polygon_area(quad)
    img_area = H * W
    ratio = poly_area / img_area

    if ratio < 0.1 or ratio > 0.95:
        return []

    return [float(v) for v in quad.flatten().tolist()]


def create_anotations(mask_paths, output_path):
    data = []
    for mask_path in tqdm(mask_paths, desc="mask files"):
        image_name = os.path.splitext(os.path.basename(mask_path))[0] + ".jpg"
        coords = get_coords(mask_path)
        data.append([image_name] + coords)

    df = pd.DataFrame(data, columns=['image_name', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])
    df.to_csv(output_path, index=False)
    print(f"{output_path} is saved.")

    
def get_mask_paths(data_dir):
    mask_dir = os.path.join(data_dir, "masks")
    mask_paths = [os.path.join(mask_dir, f) for f in sorted(os.listdir(mask_dir))]
    return mask_paths

data_dir = "E:\\datasets\\midv_2020\\midv2020_processed"
mask_paths = get_mask_paths(data_dir)

print(len(mask_paths))
print(mask_paths[0])
print(get_coords(mask_paths[0]))

data_dir = "E:\\datasets\\midv_2020\\midv2020_processed"
mask_paths = get_mask_paths(data_dir)
create_anotations(mask_paths, "anotations_midv2020.csv")

df = pd.read_csv("anotations_midv2020.csv")
df
