import glob
import shutil

negativeDataset: list[str] = glob.glob("negativeDataset/*jpg")

for fname in negativeDataset:
    label = fname.split("/")[-1].split("_")[0]
    print(fname, label)
    shutil.move(fname, f"dataset/{label}")

negativeDataset: list[str] = glob.glob("positiveDataset/*jpg")

for fname in negativeDataset:
    label = fname.split("/")[-1].split("_")[0]
    print(fname, label)
    shutil.move(fname, f"dataset/{label}")

