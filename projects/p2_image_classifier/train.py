
def train(data_dir):
    from subfunctions import import_library, load_data, save_checkpoint 
    
    # load data
    train_data, trainloader, validloader, testloader = load_data(data_dir)
    
    # Use GPU if it is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # TODO: Build and train your network
    model = models.vgg16(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
        ('dropout', nn.Dropout(0.2)),
        ('inputs', nn.Linear(25088, 120)),
        ('relu1', nn.ReLU()),
        ('hidden_layer1', nn.Linear(120, 90)),
        ('relu2', nn.ReLU()),
        ('hidden_layer2', nn.Linear(90, 80)),
        ('relu3', nn.ReLU()),
        ('hidden_layer3', nn.Linear(80, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    model.to(device)
    
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
     
    # save checkpoint
    save_checkpoint(model, optimizer, classifier)
    
    # 
