
def main():
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # import require library
    from subfunctions import load_checkpoint, process_image, imshow, predict
    
    import argparse, json, torch
    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision import datasets, transforms, models
    from torch import nn, optim
    from PIL import Image
    
    checkpoint = 'checkpoint.pth'
    filepath = 'cat_to_name.json'
    image_path = 'image_06739.jpg'
    topk = 5
    learning_rate = 0.001
    
    # set up parameters for entry in command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', action='store', type=str,
                       help='Name of trained model to be loaded and used for predictions.')
    parser.add_argument('-i', '--image_path', action='store', type=str,
                       help='Location of image to predict.')
    parser.add_argument('-k', '--topk', action='store', type=int,
                       help='Select number of classes you wish to see in descending order.')
    parser.add_argument('-j', '--json', action='store', type=str,
                       help='Define name of json file holding class names.')
    parser.add_argument('-g', '--gpu', action='store_true', 
                       help='Use GPU if available.')
    
    args = parser.parse_args()
    
    if args.checkpoint:
        checkpoint = args.checkpoint
    if args.image_path:
        image_path = args.image_path
    if args.topk:
        topk = args.topk
    if args.json:
        filepath = args.json
    if args.gpu:
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 

    
    # read json file
    with open(filepath, 'r') as f:
        cat_to_name = json.load(f)

    # load trained model
    model = load_checkpoint(checkpoint)
    
    # TODO: Implement the code to predict the class from an image file
    # load and transform images
    top_p, top_classes, top_flowers = predict(image_path, model, topk)
    
if __name__ == '__main__': main()
    