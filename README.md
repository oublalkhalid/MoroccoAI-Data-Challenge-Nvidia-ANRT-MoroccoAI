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
<img src="images/gif_characters.gif" alt="gif_characters">

<img src="images/gif_plates.gif" alt="">

2. **Create YOLO model for object detection**
   in these project i use YOLOv3  to make object deteaction model. You can learn more about the yolo in the documentation [link here](https://github.com/ultralytics/yolov5)

   > YOLO an acronym for 'You only look once', is an object detection algorithm that divides images into a grid system. Each cell in the grid is responsible for detecting objects within itself.

   if you want to create your own object detection with custom dataset you can watch these helpful tutorial to achieve that in [here](https://www.youtube.com/watch?v=GRtgLlwxpc4)

3. **Data - Augementation and Transformation **
  
  This is an unbalanced case of the classes, what it does is perform a log(y) transformation on the data just before the CNN input.
<img src="images/non-eq.png" alt="">
