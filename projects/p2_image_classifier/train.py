
def main():
    from subfunctions import load_data, create_model,train_model,  save_checkpoint
    
    import argparse, json, torch
    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision import datasets, transforms, models
    from torch import nn, optim
    from PIL import Image
    
    arch = 'vgg16'
    hidden_units = 120
    learning_rate = 0.001
    epochs = 5
    device = 'cpu'
    
    # set up parameters for entry in command line
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, 
                        help='Location of directory with data for image classifier to train and test.')
    parser.add_argument('-a', '--arch', action='store', type=str,
                       help='Choose among 3 pretrained networks - vgg16, alexnet, and densenet121.')
    parser.add_argument('-u', '--hidden_units', action='store', type=int,
                       help='Select number of hidden units for 1st layer.')
    parser.add_argument('-l', '--learning_rate', action='store', type=float,
                       help='Choose a float number as the learning rate for the model.')
    parser.add_argument('-e', '--epochs', action='store', type=int,
                       help='Choose the number of epochs you want to perform gradient descnet.')
    parser.add_argument('-s', '--save_dir', action='store', type=str,
                       help='Select name of file to save the trained model')
    parser.add_argument('-g', '--gpu', action='store_true', 
                        help='Use GPU if available')
    
    args = parser.parse_args()
    
    # Select parameters entered in command line
    if args.arch:
        arch = args.arch
    if args.hidden_units:
        hidden_units = args.hidden_units
    if args.learning_rate:
        learning_rate = args.learning_rate
    if args.epochs:
        epochs = args.epochs
    if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        
    # load data
    data_dir = args.data_dir
    train_data, trainloader, validloader, testloader = load_data(data_dir)
    
    # Use GPU if it is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # create model
    model, criterion, optimizer = create_model(arch, hidden_units, learning_rate)
    model.to(device)
    
    # train mode
    trained_model = train_model(device, trainloader, model, criterion, optimizer, epochs)
     
    # save checkpoint
    save_checkpoint(trained_model, optimizer, classifier)
    
# implement train()
if __name__ == '__main__' : main()