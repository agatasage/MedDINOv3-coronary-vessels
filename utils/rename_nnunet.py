import os
import shutil

#### rename from XXX_0001.png ---> XXX.png

src_dir = r"C:\Users\asage\Documents\.ABM\nnUNet\nnUNet_raw\Dataset501_ARCADE\imagesTs"
dst_dir = r"C:\Users\asage\Documents\.ABM\nnUNet\nnUNet_raw\Dataset501_ARCADE\imagesTs"

os.makedirs(dst_dir, exist_ok=True)

files = sorted([f for f in os.listdir(src_dir) if f.endswith(".png")])

for i, fname in enumerate(files, start=1):
    new_name = f"case_{i:04d}.png"

    shutil.copy(
        os.path.join(src_dir, fname),
        os.path.join(dst_dir, new_name)
    )

print("All done.")