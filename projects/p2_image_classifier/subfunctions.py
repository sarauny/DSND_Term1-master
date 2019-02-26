import argparse, json, torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn, optim
from PIL import Image


# load the data
def load_data(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                     ])

    valid_transforms = transforms.Compose([transforms.Resize(225),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                     ])

    test_transforms = transforms.Compose([transforms.Resize(225),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                     ])
    
    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    dataloaders = [trainloader, validloader, testloader]
    return train_data, trainloader, validloader, testloader


def create_model(arch='vgg16', hidden_units=120, learning_rate=0.001):
    # TODO: Build and train your network
    model = getattr(models, arch)(pretrained=True)
    in_features = model.classifier[0].in_features
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
        ('dropout', nn.Dropout(0.2)),
        ('inputs', nn.Linear(in_features, hidden_units)),
        ('relu1', nn.ReLU()),
        ('hidden_layer1', nn.Linear(hidden_units, 90)),
        ('relu2', nn.ReLU()),
        ('hidden_layer2', nn.Linear(90, 80)),
        ('relu3', nn.ReLU()),
        ('hidden_layer3', nn.Linear(80, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer


# train model
def train_model(device, trainloader, model, criterion, optimizer, epochs=5):
    # Train the classifier layers using backpropagation using the pre-trained network to get the features
    epochs = 5
    print_every = 5
    steps = 0
    running_loss = 0
    
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
        
            # Move input and label tensors to the GPU
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()    

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                valid_accuracy = 0
                model.eval()

                with torch.no_grad():
                    for ii, (inputs2, labels2) in enumerate(validloader):
                        inputs2, labels2 = inputs2.to(device), labels2.to(device)
                        logps = model.forward(inputs2)
                        batch_loss = criterion(logps, labels2)

                        valid_loss += batch_loss.item()

                        # Caluculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels2.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {e+1}/{epochs}..",
                      f"Train loss: {running_loss/print_every:.3f}..",
                      f"Validation loss: {valid_loss/len(validloader):.3f}..",
                      f"Validation accuracy: {valid_accuracy/len(validloader):.3f}")

                running_loss = 0
                model.train()
                
                return model
    
# save checkpoint
def save_checkpoint(model, optimizer, classifier):
    # TODO: Save the checkpoint 
    model.class_to_idx = train_data.class_to_idx
    model.epochs = epochs

    checkpoint = {'state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'classifier':model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'epochs': model.epochs,
                  'lr': learning_rate}

    torch.save(checkpoint, 'checkpoint.pth')

    
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, arch)(pretrained=True)

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.class_to_idx = checkpoint['class_to_idx']
    epochs = checkpoint['epochs']
   
    return model


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image_path)
    
    size = 256
    width, height = image.size
    shortest_side = min(width, height)
    image = image.resize((int((width/shortest_side)*size),
                          int((height/shortest_side)*size)))
    
    new_width, new_height = 224, 244
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    image = image.crop(box=(left, top, right, bottom))
    
    img_array = np.array(image)
    np_image = img_array/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    image = np_image.transpose((2, 0, 1))
    
    return image


# show image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# predict top classes
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # load and transform images
    image = process_image(image_path)
    
    # Convert 2D image to 1D vector
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image)
    image = image.float().to(device)
    
    model.eval()
    ps = torch.exp(model.forward(image))
    top_p, top_class = ps.topk(topk, dim=1)
    
    top_p = np.array(top_p.detach())[0]
    top_class = np.array(top_class.detach())[0]
    
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [index_to_class[each] for each in top_class]
    
    # change integer to flower name
    top_flowers = [cat_to_name[each] for each in top_classes]
    
    return top_p, top_classes, top_flowers