import torch
import cv2
import numpy as np

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
from glob import glob
import cv2
import torch
import os
import numpy as np

# Bounding box coordinates (replace with your actual tensor)
# Format: [x1, y1, x2, y2]
def compartment_extractor(image_path):
    model = YOLO('compartment_extraction.pt')
    # Extract spaces between bounding boxes and save cropped images
    output_folder = './output_images/'  # Change this to your desired folder
    os.makedirs(output_folder, exist_ok=True)
    model = YOLO('compartment_extraction.pt')
    results=model(image_path)
    for r in results:
        bounding_boxes=r.boxes.xyxy.cpu()
    print(bounding_boxes)
    # Load the original image
    image = cv2.imread(image_path)
    image_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Create a folder with the image filename
    output_folder = f'./output_images/{image_filename}/compartments'  # Change this to your desired folder
    os.makedirs(output_folder, exist_ok=True)
    top_box = torch.tensor([0, 0, image.shape[1], 0])  # Top of the image
    bottom_box = torch.tensor([0, image.shape[0], image.shape[1], image.shape[0]])  # Bottom of the image
    bounding_boxes = torch.cat((top_box.unsqueeze(0), bounding_boxes, bottom_box.unsqueeze(0)))
    sorted_indices = torch.argsort(bounding_boxes[:, 1])

    #  Sort bounding boxes based on sorted indices
    sorted_boxes = bounding_boxes[sorted_indices]


    counter=0
    for i in range(len(sorted_boxes) - 1):
        top_box = sorted_boxes[i]
        bottom_box = sorted_boxes[i + 1]

        # Calculate space coordinates between two consecutive boxes
        space_top = int(top_box[1])
        space_bottom = int(bottom_box[3])

        if space_top < space_bottom:
            # Extract the space between the boxes
            space_image = image[space_top:space_bottom, :, :]

            # Check if the space is non-black
            if np.any(space_image):
                counter += 1
                cv2.imwrite(f'{output_folder}/Compartment_{counter}.jpg', space_image)
                print("Cropped space images saved in:", output_folder)

  # Confirm the operation is completed
    
def extraction(image):
    # Replace 'your_folder_path' with the actual path to your folder containing JPG files
    folder_path = './Sample/'
    model = YOLO('compartment_extraction.pt')
    # Use the glob function to get a list of JPG files in the folder
    jpg_files = glob.glob(os.path.join(folder_path, '*.jpg'))

    # Iterate over the JPG files
    for jpg_file in jpg_files:
        print("Processing:", jpg_file)
        print(jpg_file)
        compartment_extractor(str(jpg_file))
