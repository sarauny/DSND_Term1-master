
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # import require library
    from subfunctions import load_checkpoint, process_image

    # load trained model
    model = load_checkpoint('checkpoint.pth')
    
    # TODO: Implement the code to predict the class from an image file
    # load and transform images
    image = Image.open(image_path)
    image = process_image(image)
    print(image)
    # Convert 2D image to 1D vector
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image)
    
    model.eval()
    ps = torch.exp(model.forward(image))
    top_p, top_class = ps.topk(topk, dim=1)
    
    return top_p, top_class
    