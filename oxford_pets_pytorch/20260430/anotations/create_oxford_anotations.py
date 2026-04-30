import os
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET

def get_coords(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin = float(bndbox.find("xmin").text) - 1
            ymin = float(bndbox.find("ymin").text) - 1
            xmax = float(bndbox.find("xmax").text) - 1
            ymax = float(bndbox.find("ymax").text) - 1

            if xmin < 0 or ymin < 0 or xmax <= xmin or ymax <= ymin:
                continue

            x1, y1 = xmin, ymin  # 좌상단
            x2, y2 = xmax, ymin  # 우상단
            x3, y3 = xmax, ymax  # 우하단
            x4, y4 = xmin, ymax  # 좌하단
            return [x1, y1, x2, y2, x4, y4, x3, y3]

    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
    return []


def create_anotations(mask_paths, output_path):
    data = []
    for mask_path in tqdm(mask_paths, desc="xml files"):
        image_name = os.path.splitext(os.path.basename(mask_path))[0] + ".jpg"
        coords = get_coords(mask_path)
        data.append([image_name] + coords)

    df = pd.DataFrame(data, columns=['image_name', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])
    df.to_csv(output_path, index=False)
    print(f"{output_path} is saved.")


def get_mask_paths(data_dir):
    mask_dir = os.path.join(data_dir, 'annotations', 'xmls')
    mask_paths = [os.path.join(mask_dir, f) for f in sorted(os.listdir(mask_dir))]
    return mask_paths

data_dir = "E:\\datasets\\oxford_pets"
mask_paths = get_mask_paths(data_dir)

print(len(mask_paths))
print(mask_paths[0])
print(get_coords(mask_paths[0]))

data_dir = "E:\\datasets\\oxford_pets"
mask_paths = get_mask_paths(data_dir)
create_anotations(mask_paths, "anotations_oxford.csv")

df = pd.read_csv("anotations_oxford.csv")
df
