import os
import tensorflow as tf
import numpy as np
def visualize_prediction(img_pairs):
    save_path = './project/matrix_siamese.h5'
    model = tf.keras.models.load_model(save_path, custom_objects={'contrastive_loss':contrastive_loss})
    img_height=800
    img_width=800
    #Load images
    first_img = tf.keras.preprocessing.image.load_img(img_pairs[0], target_size=(img_height, img_width))
    second_img = tf.keras.preprocessing.image.load_img(img_pairs[1], target_size=(img_height, img_width))

    #Convert to array
    first_img_array = tf.keras.preprocessing.image.img_to_array(first_img)
    second_img_array = tf.keras.preprocessing.image.img_to_array(second_img)

    #Convert to a batch
    first_img_array = np.expand_dims(first_img_array, axis=0)
    second_img_array = np.expand_dims(second_img_array, axis=0)

    #Normalize data
    first_img_array_norm = tf.keras.applications.mobilenet.preprocess_input(first_img_array)
    second_img_array_norm = tf.keras.applications.mobilenet.preprocess_input(second_img_array)

    #Model prediction - distance
    distance = model.predict([first_img_array_norm, second_img_array_norm])

    #print('distance :', distance)

    if distance <= 4:
        print('Predicted :', 'Same')

    else:
        print('Predicted :', 'Different')
def contrastive_loss(y_true, y_pred):

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
def classify():
    path = "./project/"
    dir_list = next(os.walk(path))[1]
    dir_list.sort()
    print(dir_list)
    save_path = './project/matrix_siamese.h5'
    model = tf.keras.models.load_model(save_path, custom_objects={'contrastive_loss':contrastive_loss})
    query_image  = []
    positive_images = []
    for i,directory in enumerate (dir_list):
                                if directory == "Pass":
                                  print (path+directory)
                                  #Read all image file names in the directory
                                  a = os.listdir(path+'/'+directory)
                                  a.sort()

                                  #Add path to image name
                                  positive_images = [path+directory + '/'+ x for x in a]
                                if directory == "test" and query_image is not None:
                                  print (path+directory)
                                  #Read all image file names in the directory
                                  a = os.listdir(path+'/'+directory)
                                  a.sort()
                                  #Add path to image name
                                  query_image = [path+directory + '/'+ x for x in a]


                            #Check if we have 160 people's genuine and forged signatures
    print('Number of orignal_border:', len(positive_images))
    print('Number of different_borders:', len(query_image))
    img_width = 800
    img_height = 800
    for i in range (len(positive_images)):
                              print([positive_images[i],query_image[0]])
                              visualize_prediction([positive_images[i],query_image[0]])
