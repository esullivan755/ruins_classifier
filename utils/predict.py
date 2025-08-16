# Functions used to predict the probability of a tile containing ruins


#Standard
import io

#OpenAI
import openai

#Data & Math
import numpy as np

#Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest


#Pytorch & Tensorflow
import torch
import torch.nn as nn
from torchvision import models
from tensorflow.keras.applications.resnet50 import ResNet50



# Random Forest Classifier Functions
def train_model(X, y, param_grid):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

def predict_anomalies(model, scaler, X):
    X_scaled = scaler.fit_transform(X)
    probs = model.predict_proba(X_scaled)
    return probs[:, 1]



# Isolation Forest Functions

#Loading image and divide into patches
def get_patches_from_array(img, patch_size=(64, 64), step=32):
    """
    Breaking image up into smaller patches.
    The anomaly will be the patch that stands out the most in each image.
    """
    patches = []
    positions = []

    c,h,w = img.shape #Channel, Height, Width
    patch_h, patch_w = patch_size


    for y in range(0, h - patch_h + 1, step):
        for x in range(0, w - patch_w + 1, step):

            patch = img[:, y:y+patch_h, x:x+patch_w]  #shape (C, H, W)
            patch = np.transpose(patch, (1, 2, 0))    #convert to (H, W, C)

            patches.append(patch)
            positions.append((x, y))

    return np.array(patches), positions, (h, w)


#Isolation Forest

#Get anomaly scores
def detect_anomalies(features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    #Setting contamination low for accurate ruin simulation
    iso_forest = IsolationForest(contamination=0.025, random_state=42)
    scores = iso_forest.fit_predict(X_scaled)
    return scores



def build_feature_extractor(state_dict_path, input_channels=6, device='cpu'):
    """
    Sets up the resnet structure, removing final classification layer and resizing inputs
    """
    model = models.resnet50(pretrained=False)

    #Modifying first layer for new input channels size
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)


    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)

    #Loading weights
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)

    #Strip off classifier to make it a feature extractor
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    return feature_extractor



def build_anomalies(feat, geo_bounds, patch_size=(64, 64), step=32, contamination=0.0025, device='cpu'):
    """
    Detects and matches the anomalies from the score matrix to lat/lon position using the coordinate ranges
    and features given.

    """

    #Get Dimensions
    _, img_h, img_w = feat.shape

    #Get Patches
    patches, positions, _ = get_patches_from_array(feat, patch_size=patch_size, step=step)

    #To Tensor
    patch_tensor = torch.tensor(patches).permute(0, 3, 1, 2).float()  #[N, C, H, W]

    #Build extractor
    feature_extractor = build_feature_extractor(pretrained_weights_path, device=device)
    patch_tensor = patch_tensor.to(device)

    #Extract features
    with torch.no_grad():
        feats = feature_extractor(patch_tensor)         #[N, 2048, 1, 1]
        feats = feats.view(feats.size(0), -1).cpu().numpy()  #[N, 2048]

    scores = detect_anomalies(feats)

    #Matching scores to coordinates
    anomalies = []
    for i, pred in enumerate(scores):
        if pred == -1:
            x, y = positions[i]

            #Using pixel centers for lat/lon
            pixel_x_center = x + patch_size[0]//2
            pixel_y_center = y + patch_size[1]//2

            lon = geo_bounds['lon_min'] + (pixel_x_center/img_w)*(geo_bounds['lon_max'] - geo_bounds['lon_min'])
            lat = geo_bounds['lat_max'] - (pixel_y_center/img_h)*(geo_bounds['lat_max'] - geo_bounds['lat_min'])

            anomalies.append({
                'lat': lat,
                'lon': lon,
                'score': -1,
                'radius_m': step
            })

    return anomalies


def plot_array_with_anomalies(array, geo_bounds, anomalies, title="Data", cmap="terrain", label="Value"):
    """
    Generic plotter for single band arrays with anomalies

    """
    lat_min = geo_bounds['lat_min']
    lat_max = geo_bounds['lat_max']
    lon_min = geo_bounds['lon_min']
    lon_max = geo_bounds['lon_max']

    H, W = array.shape

    plt.figure(figsize=(6, 6))
    plt.imshow(array, cmap=cmap, origin="upper")

    for anomaly in anomalies:
        lat = anomaly['lat']
        lon = anomaly['lon']

        #Converting lat/lon to pixel coords
        x = (lon - lon_min)/(lon_max - lon_min) * W
        y = (lat_max - lat)/(lat_max - lat_min) * H

        plt.plot(x, y, 'ro', markersize=5)
        plt.gca().add_patch(
            plt.Circle(
                (x, y),
                anomaly['radius_m'] * H / 4440,
                color='red',
                fill=False,
                linewidth=1
            )
        )

    plt.colorbar(label=label)

    plt.title(f"{title} with Anomalies")
    plt.xlabel("Longitude (approx. px)")
    plt.ylabel("Latitude (approx. px)")
    plt.show()



def visualize_anomalies(image_pil, geo_bounds, anomalies, patch_size=(64, 64), save_path=None, show=True):
    """
    Visualizes anomaly patch with satellite imageing, showing the image + outlined patch
    geo_bounds: dictionary of the image bounding box coordinates
    image_pil: imput pil image

    """
    #Creating cv2 image object, getting dimensions
    img_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    img_h, img_w = img_cv2.shape[:2]

    #Drawing anomaly bounding boxes
    for anomaly in anomalies:

        #Converting lat/lon back to pixel
        lat, lon = anomaly['lat'], anomaly['lon']
        x = int(((lon - geo_bounds['lon_min'])/(geo_bounds['lon_max'] - geo_bounds['lon_min']))*img_w)
        y = int(((geo_bounds['lat_max'] - lat)/(geo_bounds['lat_max'] - geo_bounds['lat_min']))*img_h)
        top_left = (x - patch_size[0]//2, y - patch_size[1]//2)
        bottom_right = (x + patch_size[0]//2, y + patch_size[1]//2)
        cv2.rectangle(img_cv2, top_left, bottom_right, (0, 0, 255), 2)

    if show:
        plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

        plt.axis('off')
        plt.show()

    if save_path:
        cv2.imwrite(save_path, img_cv2)


#ChatGPT Query Functions

def image_to_data_url(image):
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (255 * (image - np.min(image)) / (np.ptp(image) + 1e-8)).astype(np.uint8)
        image = Image.fromarray(image)

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_base64}"


def query_chatgpt_with_images(prompt, images, model="gpt-4o"):

    image_messages = []
    for image in images:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_data_url = f"data:image/jpeg;base64,{img_base64}"
        image_messages.append({
            "type": "image_url",
            "image_url": {"url": img_data_url}
        })

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *image_messages
            ],
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1000,
    )

    return response.choices[0].message.content
