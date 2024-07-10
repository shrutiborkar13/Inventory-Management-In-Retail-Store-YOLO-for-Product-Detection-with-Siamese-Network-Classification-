from ultralytics import YOLO
from glob import glob
import cv2
import os
import numpy as np
def detect_products_func(x):
    # Directory paths
    input_folder = x
    output_folder = input_folder + 'products/'
    model = YOLO('object_extractor.pt')
    os.makedirs(output_folder, exist_ok=True)
    compartment_images = glob(os.path.join(input_folder, '*.jpg'))

    compartment_images_with_boxes = []

    for i, compartment_image_path in enumerate(compartment_images):
        original_image = cv2.imread(compartment_image_path)

        results = model(compartment_image_path)
        for r in results:
            bounding_boxes = r.boxes.xyxy.cpu()

        # Draw bounding boxes on the original image
        for box in bounding_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            original_image = cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        compartment_name = os.path.splitext(os.path.basename(compartment_image_path))[0]
        compartment_folder = os.path.join(output_folder, compartment_name)
        os.makedirs(compartment_folder, exist_ok=True)

        # Save the original image with bounding boxes
        original_image_with_boxes_path = os.path.join(compartment_folder, f'{compartment_name}_with_boxes.jpg')
        cv2.imwrite(original_image_with_boxes_path, original_image)
        compartment_images_with_boxes.append(original_image)

        for j, box in enumerate(bounding_boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            object_image = original_image[y1:y2, x1:x2]
            object_image_path = os.path.join(compartment_folder, f'product_{j + 1}.jpg')
            cv2.imwrite(object_image_path, object_image)
            print(f"Saved product {j + 1} from compartment {i + 1}.")

    print("Cropped products saved in individual folders within:", output_folder)
    return compartment_images_with_boxes