## The Challenge and Project BioMassters

This challenge was chosen from the [DrivenData](https://drivendata.co/) website as the final project for my data science bootcamp - Project BioMassters.


The main goal in this challenge was to estimate the yearly biomass (AGBM) of different sections in Finland's forests using imagery from Sentinel-1 and Sentinel-2 satellites. The ground truth for this challenge was derived from airborne LiDAR surveys and in-situ measurements. 

<img alt="Picture" width="800" height="280" src="https://user-images.githubusercontent.com/114782592/211893572-e2628a2d-7dce-439f-be68-057819913ccf.png">

Aboveground biomass (AGBM) is a widespread measure in the study of carbon release and sequestration by forests. Forests can act as carbon sinks by removing carbon dioxide from the air through photosynthesis, but they can also release carbon dioxide through wildfires, respiration, and decomposition. In order to understand how much carbon a forest contains (its carbon stock) and how it changes (carbon flux), it is important to have an accurate measure of AGBM. In turn, such information allows landowners and policy makers to make better decisions for the conservation of forests.

There are a variety of methods that can be used to estimate AGBM, ranging from destructive sampling, which involves cutting down a representative sample of trees and measuring attributes such as the height and width of their crowns and trunks, to remote sensing methods. Remote sensing methods offer a much faster, less destructive, and more geographically expansive biomass estimate. 

<img width="401" alt="image" src="https://user-images.githubusercontent.com/114782592/211896352-f6d6e02d-caf9-443d-8957-33ea3e37f48a.png">

In particular, LiDAR (Light Detection And Ranging) is one such method that is often used to generate precise three-dimensional information about the Earth's surface. However, airborne LiDAR measurements are time consuming and expensive sources of data to collect. To get accurate biomass, LiDAR must also be calibrated with on-the-ground sampling methods. It is challenging to cover and monitor large areas with continuous acquisitions.

| Sentinel-1 (S1) Satellite images | Sentinel-2 (S2) Satellite Images|
| :-: | :-: |
| ![image](https://user-images.githubusercontent.com/114782592/211898193-5dba2dfa-924d-4916-8454-195ceb87c258.png) | ![image](https://user-images.githubusercontent.com/114782592/211898269-5a9cdca9-d5a0-444c-8b03-851bd33cfcda.png) |


Satellite imagery provided by satellites like Sentinel-1 and Sentinel-2 is more timely and has wider coverage. Sentinel-1 and Sentinel-2 are two of five missions developed by the European Commission and the European Space Agency as a part of the Copernicus program, which is an Earth-observation initiative. These satellites are built to monitor several phenomena such as sea ice, oil spills and ships, winds and waves, and land use changes. The data collected by Sentinel-1 and Sentinel-2 can be highly effective in measuring the most important metrics for forest management and conservation and climate change mitigation, if used correctly.

The challenge can be consulted online in <a href='https://drivendata.org/competitions/99/biomass-estimation/'>https://drivendata.org/competitions/99/biomass-estimation/</a>

## Project Objectives

As the main goal, a ML/DL model must predict the amount of biomass for each forest parcel. This prediction is given in a form of an "image", a 256x256 array of "pixel" numbers where each number represents the amount of biomass in the area represented by each "pixel" (roughly a 10mx10m ground area).

This means our model must predict, for each parcel, an "image" from a group of images, given by satellites S1 and S2, throughout the 12 months of the year:
 
2. Setup a deep learning model that is able to predict an image from one or more images as input, and therefore, estimate the biomass in Finland forest from satellite imagery
3. The model chosen uses a U-NET approach where X images are used by a convolutional network to predict a final image
4. The performance metric used was the **Average Root Mean Square Error (RMSE)**

$$Average RMSE = \frac {\sum_{i=0}^M \sqrt{\frac{1}{N} \sum_{i=0}^N (y_i - \hat{y_i})^2 }}{M}$$

Besides creating a model to predict our biomass estimamtion "images", we also had to apply all knowledge in ML Ops from LeWagon bootcamp:

  1. Create an API that would give one image as output from a group of images as input, using **FastAPI**
  2. Release the API in production Using a **Docker container** and **Google Cloud Run**
  3. Create a website for a demo with the API: load/show a group of input images and show the final predicted image

## Results

The primary objective of this project was to fulfill the deliveries for our bootcamp final project: a working model able to predict our output; a live API; a live website; an automated process to deal with the model training/time constraints.

Here is our demo website with our deep learning model API:


https://user-images.githubusercontent.com/114782592/211946494-e1adf6ff-1d8a-4bc5-9cfd-9ae32563202f.mp4


The second objective was to submit our predictions for the challenge and get a score/rank to see how we would rank among other/more experienced data scientists (and hopefully get the money prize 😁). 

Challenge result: **We managed to rank in #52 among more than 800 participants!!** 

This project aims at providing biomass estimations to answer the challenge from the website drivendata.org. The biomass estimation is nothing but a tool used by environmentalists to calculate the amount of carbon captured in a forest (to estimate not only the ability of the forest to capture carbon, but also to know the potential amount of carbon release, for example, if the forest burns). 

The challenge consists in estimating the biomass of a forest, Finland forest, using satellite imagery. LiDAR 3D models where first used to estimate biomass but these surveys are expensive and time-consuming. Using a LiDAR "image" as a target (each pixel number is actually the biomass estimation for each 10mx10m forest area), our model needs to predict as accurately as possible this same "image" using satellite images as inputs. This will make the biomass estimation much cheaper and faster to obtain.

## Our model
*********************************

<details>
  <summary markdown='span'> See model code</summary>

```bash
def initialize_model(start_neurons = 32) -> Model:
    """
    Initialize the Neural Network for image processing
    """
    print("Initialize model..." )
    input1 = Input(shape=(256,256,4)) #add 4 channels for S1 images
    input2 = Input(shape=(256,256,11)) # add 11 channels for S2 images

    conv1 = Conv2D(start_neurons * 1, (4, 4), activation="relu", padding="same")(input1)
    conv1_1 = Conv2D(start_neurons * 1, (4, 4), activation="relu", padding="same")(input2)
    conv1 = concatenate([conv1, conv1_1])
    conv1 = Conv2D(start_neurons * 1, (4, 4), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(start_neurons * 2, (4, 4), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (4, 4), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

    output_layer = Conv2D(1, (1,1), padding="same", activation="linear")(uconv1)

    model = Model(inputs=[input1, input2], outputs = [output_layer])
```
</details>


![finnish forest - image from drivendata website](https://drivendata-public-assets.s3.amazonaws.com/biomass-finnish-forests.jpg)


