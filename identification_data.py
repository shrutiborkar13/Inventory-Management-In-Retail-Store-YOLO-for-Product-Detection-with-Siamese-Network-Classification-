import os
import cv2
import numpy as np
import tensorflow as tf
import json
import csv
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Function to define contrastive loss
def contrastive_loss(y_true, y_pred):
    margin = 1

    positive_loss = tf.keras.backend.square(y_pred)
    negative_loss = tf.keras.backend.square(tf.keras.backend.maximum(margin - y_pred, 0))

    total_loss = y_true * positive_loss + (1 - y_true) * negative_loss
    total_average_loss = tf.keras.backend.mean(total_loss)

    return total_average_loss

# Function to load the trained Siamese model
def load_siamese_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'contrastive_loss': contrastive_loss})

# Function to preprocess an image for inference
def preprocess_image(image_path, target_size=(800, 800)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_array = tf.image.resize(img_array, target_size)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to extract product name from filename
def extract_product_name(filename):
    return os.path.splitext(filename)[0]  # Remove extension

# Function to find the compartment for a given image and save results in CSV
def find_compartment_and_save_csv(image_path, products_folder, siamese_model, threshold=2.1, csv_file_path='products.csv'):
    test_img_array = preprocess_image(image_path)

    compartment_data = []

    for compartment in os.listdir(products_folder):
        compartment_number = compartment.split("_")[-1]
        path = os.path.join(products_folder, compartment)
        for filename in os.listdir(path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                product_image_path = os.path.join(path, filename)
                template_img_array = preprocess_image(product_image_path)

                template_batch = np.zeros((1, 800, 800, 3))
                template_batch[0] = template_img_array
                test_batch = np.zeros((1, 800, 800, 3))
                test_batch[0] = test_img_array

                distance = siamese_model.predict([template_batch, test_batch])[0]

                if distance < threshold:
                        product_name="Dove"
                        matched_product_name = product_name
                        compartment_data.append({'product_image_path': product_image_path,
                                             'compartment_number': compartment_number,
                                             'product_name': matched_product_name})
                else:
                        matched_product_name = "Other"
                        compartment_data.append({'product_image_path': product_image_path,
                                             'compartment_number': compartment_number,
                                             'product_name': matched_product_name})

                    

    # Write results to CSV
    mode = 'w' if not os.path.exists(csv_file_path) else 'a'
    with open(csv_file_path, mode, newline='') as csvfile:
        fieldnames = ['product_image_path', 'product_name', 'compartment_number']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()
        for data in compartment_data:
            writer.writerow(data)

    print(f'Results saved in {csv_file_path}')

# Path to the trained Siamese model
siamese_model_path = r"C:\Users\Adwait\Desktop\Final_project\SKU110K_fixed\dove_800_siamese.h5"

# Path to the "products" folder
products_folder = "output_images/test_28/compartments/products/"  # Change this to the correct path

# Path to the query image
query_image_path = r"output_images\test_28\compartments\products\Compartment_5\product_4.jpg"  # Change this to the correct path

# Load the Siamese model
siamese_model = load_siamese_model(siamese_model_path)

# Find the compartment for the query image and save results in CSV
find_compartment_and_save_csv(query_image_path, products_folder, siamese_model)
