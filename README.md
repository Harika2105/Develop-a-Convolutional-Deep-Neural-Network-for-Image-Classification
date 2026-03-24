# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET

Image classification is a fundamental task in computer vision in which an input image is analyzed and assigned to one of several predefined categories based on its visual content. The objective of this experiment is to design, build, and train a Convolutional Neural Network (CNN) using a labeled image dataset to automatically learn hierarchical features such as edges, textures, and complex patterns. The dataset is preprocessed through normalization and augmentation techniques to improve model generalization and performance. The trained model is then evaluated using performance metrics such as accuracy, confusion matrix, and classification report to assess its effectiveness in correctly predicting image classes. This experiment helps in understanding deep learning architectures and their practical application in real-world image recognition tasks

## Neural Network Model

<img width="974" height="428" alt="Screenshot 2026-03-24 212509" src="https://github.com/user-attachments/assets/29c0a4ae-4f55-4c3f-b1ff-0e32daea3a50" />


## DESIGN STEPS
### STEP 1: 

Import the required libraries (torch, torchvision, torch.nn, torch.optim) and load the image dataset with necessary preprocessing like normalization and transformation.

### STEP 2: 

Split the dataset into training and testing sets and create DataLoader objects to feed images in batches to the CNN model.

### STEP 3: 

Define the CNN architecture using convolutional layers, ReLU activation, max pooling layers, and fully connected layers as implemented in the CNNClassifier class.

### STEP 4: 

Initialize the model, define the loss function (CrossEntropyLoss), and choose the optimizer (Adam) for training the network.

### STEP 5: 

Train the model using the training dataset by performing forward pass, computing loss, backpropagation, and updating weights for multiple epochs.

### STEP 6: 

Evaluate the trained model on test images and verify the classification accuracy for new unseen images.

## PROGRAM

### Name:S Harika

### Register Number:212224240155

```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)
    def forward(self, x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x
# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Name:Sivamani Harika')
        print('Register Number:212224240155')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

### OUTPUT

## Training Loss per Epoch

<img width="430" height="215" alt="Screenshot 2026-02-12 214745" src="https://github.com/user-attachments/assets/2e2f94b9-e29f-401f-af49-015513aec2d9" />


## Confusion Matrix

<img width="942" height="751" alt="Screenshot 2026-02-12 214843" src="https://github.com/user-attachments/assets/be9e7764-9270-4d27-a83f-95f111a1987d" />

## Classification Report

<img width="685" height="444" alt="Screenshot 2026-02-12 215028" src="https://github.com/user-attachments/assets/a9a0f43c-6991-4235-bf10-dc232b3b0ee6" />

### New Sample Data Prediction

<img width="687" height="627" alt="Screenshot 2026-02-12 215143" src="https://github.com/user-attachments/assets/e4076915-bba5-4a0d-ae47-107a7e003e9f" />


## RESULT

The Convolutional Neural Network (CNN) model was successfully trained and achieved good classification performance on the given image dataset.
