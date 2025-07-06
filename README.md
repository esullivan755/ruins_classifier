 Open AI to Z Submission
### Eric Sullivan
### June 2025


The following is a report detailing the methodology and final detection for a ruins classifier, submitted to the OpenAI to Z Hackathon on Kaggle.

### Model Summary

#### Feature Extraction
- ResNet50 pretrained model + 5 epochs of fine tuning for 6-band images
- Output: 2048 dim feature vectors per ~5km^2 tiles
- Accuracy: 98.25% on test set

#### Clustering & Classifier Strategy
- K=2 to split ruins into "forested" and "urban"
- Top 600 forested ruins selected from distance to cluster center in feature space
- Ruins Forest Classifier trained to detect subtle signs of hidden or covered ruins
- PCA analysis for interpretability and overfit protection


#### Patch-Based Anomaly Detection
- Searched for tiles in promising regions
- Extracted 36 patch tensors per tile
- Used the previously saved ResNet50 on each patch
- 36 x 2048 output features fed into Isolation Forest
- Output scores paired to patch, fed into visualizing functions


### Data Gathering

OpenAI API
Importance: Low
- Only necessary for the last cell in ruins_classifier, this was a hackathon requirement


Google Earth Engine Project
Importance: High
- to create Earth Engine tokens
- can be set up on Google Cloud
- depending on purpose noncommercial free licenses are available as of 7/3/2025



### Notebooks

- Data_Loading_Pipeline.ipynb
- - - Construction of the archeological sites coordinates csv from kaggle data
- - - Reading of the csv and generating NDBI, NDVI, RGB, Elevation images (split up through ten cells as each takes several hours)
- - - Reading a 2nd csv hand labeled as true negatives to train the model
- - - Converting these images into a single tensor per tile through preprocessing (normalizing, resizing, concatenating)
- - - Extracting the tensors from kaggle dataset
- - - Image dataset class
- - - Train/test/val split, plotting function, and resnet training loop
- - - Extracting features from saved resnet weights for best val accuracy, saving to directory

- Ruins_Classifier_Pipeline.ipynb
- - - Image generating functions
- - - Feature loading, K means Clustering + visualization plots
- - - Sorting for top n=600 features in desired cluster
- - - Random forest classifier training, pca + confusion matrix
- - - Isolation forest, feature per patch, and visualizing functions
- - - ChatGPT response + visualization for final submission site


### Datasets

ruins_classifier
- Data for the ruins classifier is in the form of 6,224,224 preprocessed tensors obtained from data_loading or
- N,2048 feature vectors obtained from data_loadings resnet feature extractor
- Both of these data types are too large for git and not listed

data_loading
- csvs for the coordinate data are attached
- from these the rest of the notebook is possible so long as the paths are updated



## Report:

The goal of this project was to build a model that balanced high resolution with lightweight features. The four types of data used are: RGB, elevation, normalized difference vegetation index (NDVI) and 
normalized difference built-up index (NDBI). The indexes and elevation are single band images, while RGB is three bands, giving a total of 6 bands for each image at a pixel resolution of 224x224. These 
were pulled from google earth engine, on the USGS, LANDSAT, and Copernicus datasets. By training on the convolutional neural network ResNet, dimension 2048 features vectors were produced for ~5 square 
kilometer areas. The feature vector size to image area ratio can be fine tuned for speed or detail recognition in this pipeline, for example switching to the 101 layer resnet as opposed to the current 
50 layer network.

50-layer resnet was chosen as a good balance for high volume search in the targeted regions, but still a strong test accuracy. Images were extracted from google earth engine on RGB, Short Wave Infrared 
(SWIR), Near Infrared (NIR), and Elevation. NDVI and NDBI indexes were calculated from NIR & Red, and SWIR & NIR normalized differences, respectively. Resnet was loaded pretrained, but was given 5 
epochs of training and validation from the dataset. The model finished with a test accuracy of 98.25% in classifying a tensor as ruin containing or non-ruin containing. The original data included ~6000 
known ruins from kaggle datasets and research articles listed in the references. Approximately 1500 true negative images were added in, hand labeled by the author. Due to the inverse imbalance (more 
positives than negatives), K means clustering was run on the resnet produced features for the positives to sort the data. With k=2, the data split into more urbanized and more forested archaeological 
sites. Using the Euclidean distances from the cluster center of the forested ruin group, the top 600 forested ruins points were selected to train a random forest classifier.

The rationale here is that the goal is to find subtle and obscure ruins often hidden below vegetation, and so training a model in detecting forest with ruins or forest without ruins will be much more 
applicable. The random forest classifier had a high test accuracy, but with large false positives it cannot be completely trusted. The classifier was then used to give a prediction score on a set of 
potential ruins found through searching areas of interest in the Western Brazilian Amazonas. Several mid-high scoring images were selected from this group, and their tensors were processed into36 
patches. Each patch was given its own feature extraction by the saved resnet weigts, giving 36x2048 resolution. Isolation Forest was run on these patches to detect anomalies. The final scoring patch 
was selected for the random forest positive prediction score of 38-52% which is on the lower end, but this was a very subtle blemish on an otherwise completely normal tile. The classifier was trained 
on large tiles with larger archaeological findings, so this high of a score was still promising given the context. Additionally, without fail the Isolation Forest picked this brown crescent-shaped 
patch for every run, and the ChatGPT response was promising, indicating that there could likely be ruins at this location.

This anomaly was found at latitude -5.3896 and longitude -64.699857, in the Amazonas state of Western Brazil. This area is part of the middle Jurua river basin, and is primarily accessible only through 
air or water. Due its remote nature and the rich history of pre-colombian civilizations, this area is ideal for ruin searching. The terra preta anthropological soil is further evidence of larger 
civilizations once inhabiting the land. Indigenous groups in and around this region belong primarily to the Panoan and Arawá language families.

Ultimately the number one factor to improve performance would be the computation and time to build a more robust training set with more true negatives to filter out false positives. Once a perfect 
ratio of feature vector size to image resolution is achieved, it is only a matter of training data to build a bulletproof model. Expanding on this, different types of data would be beneficial (LiDAR, 
soil and hydrology maps, etc).


### References:

Jacobs, J. Q. (2023). Ancient Human Settlement Patterns in Amazonia. Personal Academic Blog.

https://www.nature.com/articles/s41586-022-04780-4#data-availability

https://peerj.com/articles/15137

https://journal.caa-international.org/articles/10.5334/jcaa.45

- As found from https://www.kaggle.com/code/fafa92/i-follow-rivers-visualizing-7500-known-sites


### LICENSE
Creative Commons, see LICENSE
