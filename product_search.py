import os
import cv2
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.image import img_to_array, load_img
def contrastive_loss(y_true, y_pred):

    """
    y_pred : Eucledean distance for each pair of images
    y_true : 1 for Genuine-genuine pair, 0 otherwise

    Contrastive loss from Hadsell-et-al.'06
    Source: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    Explanation:
    When ytrue is 1, that means the sample are duplicates of each other,
    so the Euclidean distance (ypred) between their outputs must be minimized.
    So the loss is taken as the square of that Euclidean distance itself - square(y_pred).

    When ytrue is 0, i.e. the samples are not duplicates, then the Euclidean distance
    between them must be maximized, at least to the margin. So the loss to be minimized
    is the difference of the margin and the Euclidean distance - (margin - y_pred).
    If the Euclidean distance (ypred) is already greater than the margin,
    then nothing is to be learned, so the loss is made to be zero in
    that case by saying maximum(margin - y_pred, 0).
    """

    margin = 1

    #Loss when pairs are genuine-genuine
    positive_loss = tf.keras.backend.square(y_pred)
    #Loss when pairs are genuine-fake
    negative_loss = tf.keras.backend.square(tf.keras.backend.maximum(margin - y_pred, 0))

    #Total loss
    total_loss = y_true * positive_loss + (1 - y_true) * negative_loss

    #Calculate average loss
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

# Function to find the compartment for a given image
def find_compartment(image_path, products_folder, siamese_model, threshold=2.1):
    test_img_array = preprocess_image(image_path)

    compartment_number = []
    min_distance = float('inf')

    for compartment in os.listdir(products_folder):
        print(compartment)
        path=products_folder+"/"+compartment+"/"
        print(path)
        for filename in os.listdir(path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                product_image_path = os.path.join(path, filename)
                print(product_image_path)
                template_img_array = preprocess_image(product_image_path)

                template_batch = np.zeros((1, 800, 800, 3))
                template_batch[0] = template_img_array
                test_batch = np.zeros((1, 800, 800, 3))
                test_batch[0] = test_img_array

                distance = siamese_model.predict([template_batch, test_batch])[0]
                if distance< 2.4:
                    matched_product_name = "Dove"
                else:
                    matched_product_name="Degree"
                
                if distance< 2.4 and compartment.split("_")[-1] not in compartment_number:
                    compartment_number.append(compartment.split("_")[-1])
    return compartment_number



# Path to the trained Siamese model
siamese_model_path = r"C:\Users\Adwait\Desktop\Final_project\SKU110K_fixed\shampoo_800_siamese.h5"

# Path to the "products" folder
products_folder = "output_images/test_28/compartments/products/"  # Change this to the correct path

# Path to the query image
query_image_path = r"output_images\test_28\compartments\products\Compartment_5\product_4.jpg"  # Change this to the correct path

# Load the Siamese model
siamese_model = load_siamese_model(siamese_model_path)

# Find the compartment for the query image
result_compartment = find_compartment(query_image_path, products_folder, siamese_model)

if result_compartment is not None:
    print("The query image belongs to compartments:", result_compartment)

    # Store result_compartment in a JSON file
    json_result = {'result_compartment': result_compartment}
    json_file_path = 'result_compartment.json'

    with open(json_file_path, 'w') as json_file:
        json.dump(json_result, json_file)
    
    print(f'Results stored in {json_file_path}')
else:
    print("No matching compartment found.")
