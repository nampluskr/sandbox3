import os
import pandas as pd

def create_anotations(csv_path, output_path):
    df = pd.read_csv(csv_path)
    data = []
    for name, group in df.groupby('frame_filename'):
        coords = {}
        for _, row in group.iterrows():
            coords[row['name']] = (row['x'], row['y'])
        try:
            x1, y1 = coords['tl']  # top-left
            x2, y2 = coords['tr']  # top-right
            x3, y3 = coords['br']  # bottom-right
            x4, y4 = coords['bl']  # bottom-left

        except KeyError:
            continue
        
        data.append([name, x1, y1, x2, y2, x3, y3, x4, y4])
    
    result_df = pd.DataFrame(data, columns=['image_name', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])
    result_df.to_csv(output_path, index=False)
    print(f"{output_path} is saved.")

data_dir = "E:\\datasets\\smartdoc_2015\\smart_doc_extracted"
csv_path = os.path.join(data_dir, "frame_data.csv")
print(os.listdir(data_dir))

create_anotations(csv_path, "anotations_smartdoc.csv")

df = pd.read_csv("anotations_smartdoc.csv")
df

