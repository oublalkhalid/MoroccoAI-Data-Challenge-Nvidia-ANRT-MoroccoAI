### ✨ MoroccoAI-Data-Challenge ✨

## Abstract
  This project is reply to Morocco AI challenge (conference december), which aims to detect the plate with the label name type of "XXXX-ﺏ-YY" , our project Pipeline detect only 19 class called $"0,1,2,3,4,5,6,7,8,9 and ww ,ا, ب ,ج ,ش, و ,المغرب ,ه ,د" . After that take the bounding box and get the text inside the bounding box. To get the text inside the bounding box our strategy is to use 2 neural networks with normalized flow and a Yolo to detect each character. If the character has a projection far from the normal distribution (the network is trained to semi-supervise) the yolo proceeds to propose a new bounding box the final decision presents a strong accuracy. The detection and recognition result put us among the top 10% of this challenge.

<img src="images/workflow.png" alt="workflow">

# Plate Recognition Using YOLO and CNN  ✨

  - [Plate Recognition Using YOLO and CNN](#plate-recognition-using-yolo-and-cnn)
  - [Intorduction](#intorduction)
  - [Process Create This Project](#process-create-this-project)
  - [How to run this program](#how-to-run-this-program)
  - [Preview This Project](#preview-this-project)
  - [Conclustion my own Flow Normalizing model vs easyocr](#conclustion-my-own-cnn-model-vs-easyocr)


## First Step 

1. **Annotation of Moroccan Plates Licenses**
   This data-challenge addresses the problem of ANPR in Morocco licensed vehicles. Based on a small training dataset of more labeled car images, the model should have accurate recognize the plate numbers of Morocco licensed vehicles.
   We have created an annotated database [annotated database](https://drive.google.com/drive/folders/1ZFdMo-CyisVzXsSioRH9KrbbD1J4_BMH?usp=sharing) for this challenge that we make available to other editions, or other activities related to number recognition and Arabic characters. 
   You can label your dataset image using [Labelimg]() or [makesense.ai]( and choose the YOLO format or the file with extenstion *.txt

2. **Create YOLO model for object detection**
   in these project i use YOLOv3  to make object deteaction model. You can learn more about the yolo in the documentation [link here](https://github.com/ultralytics/yolov5)

   > YOLO an acronym for 'You only look once', is an object detection algorithm that divides images into a grid system. Each cell in the grid is responsible for detecting objects within itself.

   if you want to create your own object detection with custom dataset you can watch these helpful tutorial to achieve that in [here](https://www.youtube.com/watch?v=GRtgLlwxpc4)

   **The result of training process**
   ![Screenshot](documentation/results.png)
   **Preview test dataset**
   ![Screenshot](documentation/test.jpg)

3. **Create CNN model for classification**
   A Convolutional Neural Network, also known as CNN or ConvNet, is a class of neural networks that specializes in processing data that has a grid-like topology, such as an image.
   in these project CNN use to read the 3 first charachter text inside the bounding box. and in these project i use data augmentation to make more dataset.


4. **Create Segmentation for Charachter**
   A segmentation is to takes an image as input and outputs a collection of regions (or segments) which can be represented as

   in these project i use some code from these [kaggle notebok](https://www.kaggle.com/foolishboi/license-plate-recognition-final) to create segmentation to my plate

   **The result of my Segmentation**
   ![Screenshot](documentation/segmentation.JPG)

5. **Wrap all the project code**
   with all that model ready and the segmentation to get each charachter in the image ready. i wrap all into one folder to easily organize and i use some code from these [github repo](https://github.com/biplob004/motorcycle_license_plate.git)

## How to run this program
1.  You must have python 3 install in your computer if you haven't python you can download and install your python from [here](https://www.python.org/downloads/)
2.  Clone these github repo with these command
  ```git
  git clone https://github.com/ragilhadi/plate-recognition-yolo-cnn
  ```
3. Run this command to install all the python packages
```python
cd plate-recognition-yolo-cnn
pip install -r requirements.txt #for windows
pip3 install -r requirements.txt #for mac or windows
```
> Note: You must have all the python package to run the project

4. You can run the project from command prompt with theese command

- For **image** input and use **CNN** type this command
```python
python main.py #for windows
python3 main.py #for mac or linux
```

- For live **video input** and use **CNN** type this command
```python
python camera.py #for windows
python3 camera.py #for mac or linux
```

- For **image** input and use **easyocr** type this command
```python
python image_ocr.py #for windows
python3 image_ocr.py #for mac or linux
```

- For **live video** input and use **easyocr** type this command
```python
python camera_ocr.py #for windows
python3 camera_ocr.py #for mac or linux
```


## Preview This Project
- This preview of the image with **CNN** to classify the text

- This preview of the image with **easyocr** to classify the text

## Conclustion my own CNN model vs easyocr
the result of these method to recognize the text inside the plate be depicted in these picture)
