# Nails-Segmentation

**Dataset Preparation**
1. JPG Image of hands with nails was downloaded from the internet.
2. PNG label of respective images was prepared as labels for training the model. Labels were prepared using adobe photo studio
3. Dataset used to train this model was very small - less than 1000 images even after applying augmentation technique. We can achieve better result by increasing our dataset to around 1 lakh images.

**Data Processing**
Model was tested using label in both format - jpg and png. create data.py was used to convert labels from png to jpg by converting image into RGBA channel where 'A' is for alpha channel of the image which measures the transparency at a particular pixel.
JPG contains information in every pixel and no pixel is opaque while in png the opaque pixel does not contain any informaion. Thus, it is better to use png images in case of labels to train the model.

**Training Model**
Convolutional neural network is used to trained the model. We can increase the accuracy of the model by providing it with huge dataset and tweaking the hyper-parameters of the model.

**Execution**
1. Create a Dataset Folder -  having both images and corresponding labels. Name images as "abc.jpg" and labels as "abcnails.png".
2. Create 2 folder as Augmented Nails and Segmented Nails in the same directory.
3. Use Augmentation.py to increase the dataset and store the output in Augmented Nails and Segmented Nails.
4. Use im_to_numpy.py to convert the images in Augmented Nails and Segmented Nails into numpy array and save it in numpy folder in the same directory.
5. 
