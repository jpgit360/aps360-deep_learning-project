'''
HELPER FUNCTIONS
'''

def get_model_name(name, batch_size, learning_rate, epoch):
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path

def get_dir(name, batch_size, learning_rate, timestamp):
    dir = "model_{0}_bs{1}_lr{2}_{3}/".format(name,
                                        batch_size,
                                        learning_rate,
                                        timestamp
                                        )
    return dir

def evaluate(net, loader, criterion):
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0

    for i, data in enumerate(loader, 0):
        inputs, labels = data
        outputs = net(inputs)
        notonehot_labels = torch.argmax(labels, 1)
        loss = criterion(outputs, notonehot_labels)

        max_value, predicted_output = torch.max(outputs, 1)
        true_output = torch.argmax(labels, 1)
        corr = (predicted_output != true_output)
        total_err += int(corr.sum())

        total_loss += loss.item()
        total_epoch += len(labels)
    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)
    return err, loss

def evaluate_test(net, loader, criterion):
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0

    predicted_list = []
    true_list = []

    for i, data in enumerate(loader, 0):
        inputs, labels = data
        outputs = net(inputs)
        notonehot_labels = torch.argmax(labels, 1)
        loss = criterion(outputs, notonehot_labels)

        max_value, predicted_output = torch.max(outputs, 1)
        true_output = torch.argmax(labels, 1)
        #####
        #print("predicted output: ", predicted_output)
        #print("----------------")
        #print("true output: ", true_output)
        predicted_list += predicted_output.tolist()
        true_list += true_output.tolist()
        #####
        corr = (predicted_output != true_output)
        total_err += int(corr.sum())

        total_loss += loss.item()
        total_epoch += len(labels)
    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)
    return err, loss, predicted_list, true_list

def plot_training_curve(path):
    import matplotlib.pyplot as plt
    train_err = np.loadtxt("{}_train_err.csv".format(path))
    val_err = np.loadtxt("{}_val_err.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Error")
    n = len(train_err) # number of epochs
    plt.plot(range(1,n+1), train_err, label="Train")
    plt.plot(range(1,n+1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

def load_from_drive(path, signs_dict, transform, balancer=250):
    dataset = []
    num_classes = len(signs_dict)
    for sign_type, index in signs_dict.items():
        zip_path = path +  f"/{sign_type}.zip"
        print("currently loading: ", zip_path)
        i = 0
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for entry in zip_ref.infolist():
                if(i > balancer): # temporary way to balance dataset
                    break
                if(not entry.is_dir() and (
                entry.filename.lower().endswith('.jpg') or
                entry.filename.lower().endswith('.jpeg') or
                entry.filename.lower().endswith('.png'))):
                    with zip_ref.open(entry) as file:
                        img = PIL.Image.open(file).convert("RGB")
                        tensor_img = transform(img)
                        dataset.append((tensor_img, one_hot_encode(index, num_classes)))
                    i += 1
    return dataset

def one_hot_encode(label, num_classes):
    one_hot = np.zeros(num_classes)
    one_hot[label] = 1
    return torch.tensor(one_hot)


def get_data_loader(batch_size, train_percent=0.8, val_percent=0.1, balancer=250):
    # one hot encoding for all tyes of signs
    signs_dict = {
        'chevron': 0,
        'parking': 1,
        'pedestrian_cross': 2,
        'bicycle_only': 3,
        'keep_right': 4,
        'speed_limit': 5,
        'no_entry': 6,
        'no_heavy_vehicles': 7,
        'no_left_turn': 8,
        'no_overtaking': 9,
        'no_parking': 10,
        'no_right_turn': 11,
        'no_uturn': 12,
        'one_way_left': 13,
        'one_way_right': 14,
        'one_way_straight': 15,
        'stop': 16,
        'yield': 17,
        'children': 18,
        'curve_left': 19,
        'curve_right': 20,
        'road_bump': 21,
        'school_zone': 22,
        'stop_ahead': 23,
        'traffic_signal_ahead': 24,
        'merge_right': 25
    }

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    path = f"/content/gdrive/MyDrive/dataset_imgs/dataset_v2" #CHANGE LATER
    dataset = load_from_drive(path, signs_dict, transform, balancer)

    # Get the list of indices to sample from
    relevant_indices = [i for i in range(len(dataset))]

    np.random.seed(1000) # Fixed numpy random seed for reproducible shuffling
    np.random.shuffle(relevant_indices)
    train_split = int(len(relevant_indices) * train_percent) #split at 80% for training
    val_split = int(len(relevant_indices) * val_percent) #split at 10% for validation
    # 10% for testing

    # Split into training, validation and testing sets
    relevant_train_indices = relevant_indices[:train_split]
    relevant_val_indices = relevant_indices[train_split:train_split + val_split]
    relevant_test_indices = relevant_indices[train_split + val_split:]

    train_sampler = SubsetRandomSampler(relevant_train_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            num_workers=1, sampler=train_sampler)
    val_sampler = SubsetRandomSampler(relevant_val_indices)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            num_workers=1, sampler=val_sampler)

    test_sampler = SubsetRandomSampler(relevant_test_indices)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             num_workers=1, sampler=test_sampler)
    return train_loader, val_loader, test_loader

def get_conv_output(conv_params, maxpool_params, tensor_input_size):
    out_channels = conv_params[-1]['out_channels']
    out = tensor_input_size
    for conv in conv_params:
        # conv layer
        out = math.floor((out + 2 * conv['padding'] - conv['kernel_size']) /
                         conv['stride']) + 1
        # max pooling
        out = math.floor((out - maxpool_params['kernel_size']) /
                         maxpool_params['stride']) + 1

    return out_channels * out * out