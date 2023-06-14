from pathlib import Path
import numpy as np
from PIL import Image



if __name__ == "__main__":
    data_dir = Path('data/sparse_reconstruction_and_nerf_data')
    label_max = 0
    for scene_dir in data_dir.iterdir():
        thing_dir = data_dir / scene_dir.name / 'segmentations' / 'thing'
        stuff_dir = data_dir / scene_dir.name / 'segmentations' / 'stuff'

        thing_list = []
        for file in thing_dir.iterdir():
            panoptic = np.array(Image.open(str(file)))
            thing_list.append(panoptic)

        stuff_list = []
        for file in stuff_dir.iterdir():
            panoptic = np.array(Image.open(str(file)))
            stuff_list.append(panoptic)

        thing_list = np.stack(thing_list, axis=0)
        stuff_list = np.stack(stuff_list, axis=0)

        # Seems that thing class id is separate set from stuff class id
        print(f"Scene: {scene_dir.name}")
        print(f"Things: {np.unique(thing_list)}")
        print(f"Stuff: {np.unique(stuff_list)}")

        if thing_list.max() > label_max:
            label_max = thing_list.max()
    print(f"label_max: {label_max}")    # 79