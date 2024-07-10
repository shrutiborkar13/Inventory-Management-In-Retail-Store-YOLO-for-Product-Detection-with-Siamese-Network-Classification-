import cv2
import numpy
import os
from glob import glob
class EmptinessExtraciton:
    
    def __init__(self,save_path):
        print("init called")
        self.save_path = save_path
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
                print("folder created to store emptiness")
  

    def get_emptiness(self,image,i=None, areaofbox=None):
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        image_copy = image.copy()
        grayscale = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # print("grayscale")
        blurred = cv2.GaussianBlur(grayscale,(5,5),0)
        clahe = cv2.createCLAHE(clipLimit=2.0)
        image_h = clahe.apply(grayscale) + 20
        
        kernel = numpy.ones((19,19),numpy.uint8) 
        closing = cv2.morphologyEx(image_h, cv2.MORPH_CLOSE, kernel, iterations=2)
        closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel, iterations=2)
        image = cv2.threshold(closing, 0, 200, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        l_kernel = numpy.array([[0,1,0],[1,-4,1],[0,1,0]])
        temp2 = cv2.filter2D(image,-1,l_kernel)
        
        contours , hierarchy = cv2.findContours(temp2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts_sizes = [cv2.contourArea(cnt) for cnt in contours]
        arr1=[]
        for cnt in contours:
            arr1.append(cnt)
        area_var = float(sum(cnts_sizes))
        
        arr1=tuple(arr1)
        image_copy=cv2.drawContours(image_copy, arr1, -1, (0,0,255), 2)

        dims = image_copy.shape
        h = dims[0]
        w = dims[1]

        total_compartment_area = h*w
        percent_emptiness = (area_var / total_compartment_area)*100
        #per2 = (1 - (areaofbox / total_compartment_area)) * 100
        # print("Percent Emptiness: ",percent_emptiness)   
        # print("Contoured_immage: ",image_copy)
  
        # font 
        font = cv2.FONT_HERSHEY_SIMPLEX 

        # org 
        org = (50, 50) 

        # fontScale 
        fontScale = 1

        # Blue color in BGR 
        color = (0, 0, 255) 

        # Line thickness of 2 px 
        thickness = 2

        # Using cv2.putText() method 
        text="Emptiness:"+str(round(percent_emptiness,2))+"%"
        if self.save_path is not None:
            image = cv2.putText(image_copy,text, org, font,  
                           fontScale, color, thickness, cv2.LINE_AA)
            cv2.imwrite(self.save_path +"/"+ f'compartment_{i}.png',image)
        return round(percent_emptiness,2)

def main_func(x):
    input_folder = x  # Change this to the path of your input folder
    save_folder = "./emptiness/"  # Change this to your save folder name
    print(input_folder)
    # Recursively search for image files in subfolders
    image_paths = glob(os.path.join(input_folder, '**', '*.jpg'), recursive=True)
    print(image_paths)
    counter=0
    for image_path in image_paths:
        counter+=1
        image = cv2.imread(image_path)
        print(image.shape)
        image_name = image_path.split('/')[-1].split('.')[0]
        
        # Construct the save path using the original folder structure
        save_path = os.path.join(x, save_folder)

        emptinessExtractor = EmptinessExtraciton(save_path)
        emptinessExtractor.get_emptiness(image,counter)

        print(f"Emptiness extraction for {image_path} completed and saved in {save_path}")