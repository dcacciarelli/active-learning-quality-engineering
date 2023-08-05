import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader
from PIL import Image

# Importing images (from a folder containing both good and faulty images)
DATASET = "metal_nut"
path = '/Users/dcac/Data/computer_vision/images_anomaly_detection/'+DATASET+'_mixed/'

# Showing one image
img = Image.open(path+"0_good/000.png")
plt.imshow(img)
plt.show()

transformations = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Transform example image and check dimensions
processed_image = transformations(img.convert('RGB'))
img.size
processed_image.size()
plt.imshow(processed_image.permute(1, 2, 0))
plt.show()

# Creating a data loader for torch model
dataset = datasets.ImageFolder(path, transform=transformations)
train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
images, labels = next(iter(train_loader))

# Feature extractor: removing last layer from pretrained resnet
model = torchvision.models.resnet18(pretrained=True)
new_model = torch.nn.Sequential(*(list(model.children())[:-1]))
print(new_model)

# Getting the extracted features
output = new_model(images)
output = output.reshape(-1, 512)

# Further reducing the dimensionality from 512 to 100 (retaining 90% of the total explained variance)
preprocessed_features = pd.DataFrame(output.cpu().detach().numpy())
pca = PCA(n_components=100, random_state=42)  # 100 PCs = 90% of explained variance
pca.fit(preprocessed_features)
preprocessed_features = pd.DataFrame(pca.transform(preprocessed_features))

# Exporting labeled csv to train a classification model
multiclass_y = labels.cpu().detach().numpy()
preprocessed_features["y"] = multiclass_y
binary_y = np.where(multiclass_y > 0, 1, multiclass_y)
preprocessed_features["y"] = binary_y
# preprocessed_features.to_csv("preprocessed_features_nocrop_"+DATASET+"s.csv", index=False)