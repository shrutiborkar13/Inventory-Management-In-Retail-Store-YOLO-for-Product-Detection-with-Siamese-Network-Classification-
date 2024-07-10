import streamlit as st
import tensorflow as tf
import os
import cv2
import numpy as np
from PIL import Image
import os
import glob
from Compartment import compartment_extractor
from emptiness import main_func
from detect_products import detect_products_func
from siamese import visualize_prediction,contrastive_loss,classify
# Define a function for compartment extraction
def extract_compartments(image):
    compartments = compartment_extractor(image)
    return compartments

# Define a function for emptiness extraction
def extract_emptiness(image):
    main_func(image)
# Streamlit app
st.title('Image Processing App')
st.write("Choose an image and select an option:")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    option = st.selectbox("Choose an option", ["Compartment Extraction", "Emptiness Extraction","Product Detection", "Product Classification"])

    if st.button("Process"):
        image = np.array(image)

        # Save the uploaded image to the "Sample" folder
        sample_folder = 'Sample'
        os.makedirs(sample_folder, exist_ok=True)
        image_name_without_extension = os.path.splitext(uploaded_image.name)[0]
        image_path = os.path.join(sample_folder, f'{image_name_without_extension}.jpg')
        # image_path = os.path.join(sample_folder, 'uploaded_image.jpg')
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        if option == "Compartment Extraction":
            compartments = extract_compartments(image_path)

                # Find the folder with the image name in the "output_folder"
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            output_folder = './output_folder/'  # Replace with the actual path to your "output_folder"
            image_folder = os.path.join(r"C:\Users\Adwait\Desktop\Final_project\SKU110K_fixed\output_images", image_name)+"/compartments/"
            print(image_folder)
            if os.path.exists(image_folder):
                    compartment_images = glob.glob(os.path.join(image_folder, '*.jpg'))
                    for i, compartment_image_path in enumerate(compartment_images):
                        compartment_image = Image.open(compartment_image_path)
                        st.image(compartment_image, caption=f"Compartment {i + 1}", use_column_width=True)
            else:
                    st.write("No compartments found in 'output_folder' for the uploaded image.")
        elif option == "Product Detection":
            st.title("Compartment Images with Bounding Boxes")
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            image_folder = os.path.join(r"C:\Users\Adwait\Desktop\Final_project\SKU110K_fixed\output_images", image_name+"/compartments/")
            compartment_images_with_boxes = detect_products_func(image_folder)

            for i, image in enumerate(compartment_images_with_boxes):
                st.image(image, caption=f"Compartment {i + 1} with Bounding Boxes", use_column_width=True)
        elif option == "Emptiness Extraction":
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            output_folder = './output_folder/'  # Replace with the actual path to your "output_folder"
            image_folder = os.path.join(r"C:\Users\Adwait\Desktop\Final_project\SKU110K_fixed\output_images", image_name+"/compartments/")
            extract_emptiness(image_folder)
            print(image_folder)
            if os.path.exists(image_folder):
                    compartment_images = glob.glob(os.path.join(image_folder, '*.png'))
                    for i, compartment_image_path in enumerate(compartment_images):
                        compartment_image = Image.open(compartment_image_path)
                        st.image(compartment_image, caption=f"Compartment {i + 1}", use_column_width=True)
        elif option =="Product Classification":
                classify()
