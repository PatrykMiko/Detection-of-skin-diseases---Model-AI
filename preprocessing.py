import os
import cv2
import numpy as np
import kagglehub

#download the dataset
downloaded_path = kagglehub.dataset_download("ahmedxc4/skin-ds")
contents = os.listdir(downloaded_path)
if len(contents) == 1 and os.path.isdir(os.path.join(downloaded_path, contents[0])):
    SOURCE_BASE_PATH = os.path.join(downloaded_path, contents[0])
else:
    SOURCE_BASE_PATH = downloaded_path
OUTPUT_BASE_PATH = os.path.join(os.getcwd(), "dataset_processed")

TARGET_SIZE = (480, 480)

#Dull Razor
def apply_dull_razor(img):
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
    ret, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
    dst = cv2.inpaint(img, mask, 6, cv2.INPAINT_TELEA)
    return dst


def resize(img):
    img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
    return img_resized

#CLAHE
def apply_clahe_color(img):
    # Conversion BGR --> Lab
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    # Split into channels L, a, b
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    # Using CLAHE for channel L
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    l_channel_clahe = clahe.apply(l_channel)
    l_channel_final = np.clip(l_channel_clahe.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    # Connecting the channels back together
    img_lab_clahe = cv2.merge((l_channel_final, a_channel, b_channel))
    # Conversion Lab --> BGR
    img_bgr_clahe = cv2.cvtColor(img_lab_clahe, cv2.COLOR_Lab2BGR)
    return img_bgr_clahe


def process_images(source_dir, output_dir):
    """
    Main processing function that goes through
    train/test/val folders i diseases subfolders.
    """
    for split in ['train', 'test', 'val']:
        split_path = os.path.join(source_dir, split)
        if not os.path.isdir(split_path):
            print(f"Omitted: {split_path} (the folder was not found)")
            continue

        print(f"Folder processing: {split}")

        # Loop through disease folders
        for disease_class in os.listdir(split_path):
            class_path_in = os.path.join(split_path, disease_class)
            class_path_out = os.path.join(output_dir, split, disease_class)

            if not os.path.isdir(class_path_in):
                continue

            os.makedirs(class_path_out, exist_ok=True)
            print(f"Class processing: {disease_class}")

            # Image file loop
            for filename in os.listdir(class_path_in):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path_in = os.path.join(class_path_in, filename)
                    img_path_out = os.path.join(class_path_out, filename)

                    try:
                        # 1. Loading an image
                        img = cv2.imread(img_path_in, cv2.IMREAD_COLOR)
                        if img is None:
                            print(f"Error while loading {filename}")
                            continue
                        # 2. Dull Razor
                        img_no_hair = apply_dull_razor(img)
                        # 3. Resize
                        img_resized_blurred = resize(img_no_hair)
                        # 4. CLAHE
                        img_final = apply_clahe_color(img_resized_blurred)
                        # 5. Saving the final image
                        cv2.imwrite(img_path_out, img_final)

                    except Exception as e:
                        print(f"Error in processing {filename}: {e}")

            print(f"  End of class: {disease_class}.")
    print("End of whole preprocessing!")

# Running the script
process_images(SOURCE_BASE_PATH, OUTPUT_BASE_PATH)
