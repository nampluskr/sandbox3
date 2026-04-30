import os
import pandas as pd

def filter_valid_images(csv_path, image_dir, image_name_col="image_name", coord_cols=None):
    df = pd.read_csv(csv_path)
    print(f"원본 데이터 개수: {len(df)}")

    df["image_path"] = df[image_name_col].apply(lambda x: os.path.join(image_dir, x))
    df["exists"] = df["image_path"].apply(os.path.exists)
    missing_count = len(df) - df["exists"].sum()
    
    if missing_count > 0:
        print(f"경고: {missing_count}개의 이미지 파일이 존재하지 않음")
        df = df[df["exists"]].copy()
    else:
        print("모든 이미지 파일이 존재함")

    if coord_cols:
        valid_coords = (
            df[coord_cols].notna().all(axis=1) &
            (df[coord_cols] > 0).all(axis=1) &
            (df[coord_cols] < 1e6).all(axis=1) &
            ~df[coord_cols].isin([float('inf'), float('-inf')]).any(axis=1)
        )
        invalid_coord_count = len(df) - valid_coords.sum()
        if invalid_coord_count > 0:
            print(f"경고: {invalid_coord_count}개의 좌표가 유효하지 않아 제거됨")
            df = df[valid_coords].copy()

    df = df.drop(columns=["image_path", "exists"], errors="ignore")
    df = df.reset_index(drop=True)
    
    print(f"최종 유효한 데이터 개수: {len(df)}")
    return df

image_dir = os.path.join(data_dir, "images")
df = filter_valid_images("anotations_smartdoc.csv", image_dir)
df
