import os

import cv2
import tqdm
import glob


if __name__ == "__main__":
    source_folder = "/media/dereyly/svision_ssd/cartoon-dataset/watercolor_mix"
    target_folder = "/home/misha/watercolor_mix"

    os.makedirs(target_folder, exist_ok=True)
    errors = 0
    for source_path in tqdm.tqdm(glob.glob(os.path.join(source_folder, "*.jpg"))):
        try:
            filename = os.path.basename(source_path)
            target_path = os.path.join(target_folder, filename)

            image = cv2.imread(source_path)
            cv2.imwrite(target_path, image)
        except Exception as e:
            print(e)
            errors += 1
            continue
    
    print(errors)

