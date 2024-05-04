'''
model architecture
'''
class CNN_v1(nn.Module):
    def __init__(self):
        super(CNN_v1, self).__init__()
        self.name = "cnnv1"
        # format assumes a conv -> maxpool -> conv -> maxpool -> ... structure
        conv_params = [
            {'in_channels': 3, 'out_channels': 8, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'in_channels': 8, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 0}
        ]
        maxpool_params = {'kernel_size': 2, 'stride': 2}
        self.conv0 = nn.Conv2d(conv_params[0]['in_channels'],
                               conv_params[0]['out_channels'],
                               conv_params[0]['kernel_size'],
                               conv_params[0]['stride'],
                               conv_params[0]['padding'])

        self.pool = nn.MaxPool2d(maxpool_params['kernel_size'],
                                 maxpool_params['stride'])

        self.conv1 = nn.Conv2d(conv_params[1]['in_channels'],
                               conv_params[1]['out_channels'],
                               conv_params[1]['kernel_size'],
                               conv_params[1]['stride'],
                               conv_params[1]['padding'])

        tensor_input_size = 150
        self.conv_output = get_conv_output(conv_params, maxpool_params, tensor_input_size)
        self.fc1 = nn.Linear(self.conv_output, 244) #change later
        self.fc2 = nn.Linear(244, 26)

    def forward(self, img):
        x = self.pool(F.relu(self.conv0(img)))
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, self.conv_output) # change later
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CNN_v2(nn.Module): # more output channels, no padding, no more stride
    def __init__(self):
        super(CNN_v2, self).__init__()
        self.name = "cnnv2"
        # format assumes a conv -> maxpool -> conv -> maxpool -> ... structure
        conv_params = [
            {'in_channels': 3, 'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'in_channels': 8, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0},
        ]
        maxpool_params = {'kernel_size': 2, 'stride': 2}
        self.conv0 = nn.Conv2d(conv_params[0]['in_channels'],
                               conv_params[0]['out_channels'],
                               conv_params[0]['kernel_size'],
                               conv_params[0]['stride'],
                               conv_params[0]['padding'])

        self.pool = nn.MaxPool2d(maxpool_params['kernel_size'],
                                 maxpool_params['stride'])

        self.conv1 = nn.Conv2d(conv_params[1]['in_channels'],
                               conv_params[1]['out_channels'],
                               conv_params[1]['kernel_size'],
                               conv_params[1]['stride'],
                               conv_params[1]['padding'])

        self.conv2 = nn.Conv2d(conv_params[2]['in_channels'],
                               conv_params[2]['out_channels'],
                               conv_params[2]['kernel_size'],
                               conv_params[2]['stride'],
                               conv_params[2]['padding'])

        tensor_input_size = 150
        self.conv_output = get_conv_output(conv_params, maxpool_params, tensor_input_size)
        self.fc1 = nn.Linear(self.conv_output, 244) #change later
        self.fc2 = nn.Linear(244, 26)

    def forward(self, img):
        x = self.pool(F.relu(self.conv0(img)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.conv_output) # change later
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CNN_v3(nn.Module): # no padding, slightly less output channels
    def __init__(self):
        super(CNN_v3, self).__init__()
        self.name = "cnnv3"
        # format assumes a conv -> maxpool -> conv -> maxpool -> ... structure
        conv_params = [
            {'in_channels': 3, 'out_channels': 3, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'in_channels': 3, 'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'in_channels': 8, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 0},
        ]
        maxpool_params = {'kernel_size': 2, 'stride': 2}
        self.conv0 = nn.Conv2d(conv_params[0]['in_channels'],
                               conv_params[0]['out_channels'],
                               conv_params[0]['kernel_size'],
                               conv_params[0]['stride'],
                               conv_params[0]['padding'])

        self.pool = nn.MaxPool2d(maxpool_params['kernel_size'],
                                 maxpool_params['stride'])

        self.conv1 = nn.Conv2d(conv_params[1]['in_channels'],
                               conv_params[1]['out_channels'],
                               conv_params[1]['kernel_size'],
                               conv_params[1]['stride'],
                               conv_params[1]['padding'])

        self.conv2 = nn.Conv2d(conv_params[2]['in_channels'],
                               conv_params[2]['out_channels'],
                               conv_params[2]['kernel_size'],
                               conv_params[2]['stride'],
                               conv_params[2]['padding'])

        tensor_input_size = 150
        self.conv_output = get_conv_output(conv_params, maxpool_params, tensor_input_size)
        self.fc1 = nn.Linear(self.conv_output, 244) #change later
        self.fc2 = nn.Linear(244, 26)

    def forward(self, img):
        x = self.pool(F.relu(self.conv0(img)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.conv_output) # change later
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CNN_v4(nn.Module): # more layers, consistent # of output channels
    def __init__(self):
        super(CNN_v4, self).__init__()
        self.name = "cnnv4"
        # format assumes a conv -> maxpool -> conv -> maxpool -> ... structure
        conv_params = [
            {'in_channels': 3, 'out_channels': 8, 'kernel_size': 2, 'stride': 1, 'padding': 0},
            {'in_channels': 8, 'out_channels': 8, 'kernel_size': 2, 'stride': 1, 'padding': 0},
            {'in_channels': 8, 'out_channels': 8, 'kernel_size': 2, 'stride': 1, 'padding': 0},
            {'in_channels': 8, 'out_channels': 8, 'kernel_size': 2, 'stride': 1, 'padding': 0},
            {'in_channels': 8, 'out_channels': 8, 'kernel_size': 2, 'stride': 1, 'padding': 0}
        ]
        maxpool_params = {'kernel_size': 2, 'stride': 2}
        self.conv0 = nn.Conv2d(conv_params[0]['in_channels'],
                               conv_params[0]['out_channels'],
                               conv_params[0]['kernel_size'],
                               conv_params[0]['stride'],
                               conv_params[0]['padding'])

        self.pool = nn.MaxPool2d(maxpool_params['kernel_size'],
                                 maxpool_params['stride'])

        self.conv1 = nn.Conv2d(conv_params[1]['in_channels'],
                               conv_params[1]['out_channels'],
                               conv_params[1]['kernel_size'],
                               conv_params[1]['stride'],
                               conv_params[1]['padding'])

        self.conv2 = nn.Conv2d(conv_params[2]['in_channels'],
                               conv_params[2]['out_channels'],
                               conv_params[2]['kernel_size'],
                               conv_params[2]['stride'],
                               conv_params[2]['padding'])

        self.conv3 = nn.Conv2d(conv_params[3]['in_channels'],
                               conv_params[3]['out_channels'],
                               conv_params[3]['kernel_size'],
                               conv_params[3]['stride'],
                               conv_params[3]['padding'])

        self.conv4 = nn.Conv2d(conv_params[4]['in_channels'],
                               conv_params[4]['out_channels'],
                               conv_params[4]['kernel_size'],
                               conv_params[4]['stride'],
                               conv_params[4]['padding'])

        tensor_input_size = 150
        self.conv_output = get_conv_output(conv_params, maxpool_params, tensor_input_size)
        self.fc1 = nn.Linear(self.conv_output, 244) #change later
        self.fc2 = nn.Linear(244, 26)

    def forward(self, img):
        x = self.pool(F.relu(self.conv0(img)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, self.conv_output) # change later
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CNN_v5(nn.Module): # like CNN_v1 but with more linear layers
    def __init__(self):
        super(CNN_v5, self).__init__()
        self.name = "cnnv5"
        # format assumes a conv -> maxpool -> conv -> maxpool -> ... structure
        conv_params = [
            {'in_channels': 3, 'out_channels': 8, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'in_channels': 8, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 0}
        ]
        maxpool_params = {'kernel_size': 2, 'stride': 2}
        self.conv0 = nn.Conv2d(conv_params[0]['in_channels'],
                               conv_params[0]['out_channels'],
                               conv_params[0]['kernel_size'],
                               conv_params[0]['stride'],
                               conv_params[0]['padding'])

        self.pool = nn.MaxPool2d(maxpool_params['kernel_size'],
                                 maxpool_params['stride'])

        self.conv1 = nn.Conv2d(conv_params[1]['in_channels'],
                               conv_params[1]['out_channels'],
                               conv_params[1]['kernel_size'],
                               conv_params[1]['stride'],
                               conv_params[1]['padding'])

        tensor_input_size = 150
        self.conv_output = get_conv_output(conv_params, maxpool_params, tensor_input_size)
        self.fc1 = nn.Linear(self.conv_output, 1000) #change later
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 26)

    def forward(self, img):
        x = self.pool(F.relu(self.conv0(img)))
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, self.conv_output) # change later
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class CNN_v6(nn.Module): # more padding
    def __init__(self):
        super(CNN_v6, self).__init__()
        self.name = "cnnv6"
        # format assumes a conv -> maxpool -> conv -> maxpool -> ... structure
        conv_params = [
            {'in_channels': 3, 'out_channels': 4, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'in_channels': 4, 'out_channels': 8, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'in_channels': 8, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'in_channels': 16, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 0}
        ]
        maxpool_params = {'kernel_size': 2, 'stride': 2}
        self.conv0 = nn.Conv2d(conv_params[0]['in_channels'],
                               conv_params[0]['out_channels'],
                               conv_params[0]['kernel_size'],
                               conv_params[0]['stride'],
                               conv_params[0]['padding'])

        self.pool = nn.MaxPool2d(maxpool_params['kernel_size'],
                                 maxpool_params['stride'])

        self.conv1 = nn.Conv2d(conv_params[1]['in_channels'],
                               conv_params[1]['out_channels'],
                               conv_params[1]['kernel_size'],
                               conv_params[1]['stride'],
                               conv_params[1]['padding'])

        self.conv2 = nn.Conv2d(conv_params[2]['in_channels'],
                               conv_params[2]['out_channels'],
                               conv_params[2]['kernel_size'],
                               conv_params[2]['stride'],
                               conv_params[2]['padding'])

        self.conv3 = nn.Conv2d(conv_params[3]['in_channels'],
                               conv_params[3]['out_channels'],
                               conv_params[3]['kernel_size'],
                               conv_params[3]['stride'],
                               conv_params[3]['padding'])

        tensor_input_size = 150
        self.conv_output = get_conv_output(conv_params, maxpool_params, tensor_input_size)
        self.fc1 = nn.Linear(self.conv_output, 244) #change later
        self.fc2 = nn.Linear(244, 26)

    def forward(self, img):
        x = self.pool(F.relu(self.conv0(img)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.conv_output) # change later
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN_v7(nn.Module): # more padding, and bigger classifer
    def __init__(self):
        super(CNN_v7, self).__init__()
        self.name = "cnnv7"
        # format assumes a conv -> maxpool -> conv -> maxpool -> ... structure
        conv_params = [
            {'in_channels': 3, 'out_channels': 4, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'in_channels': 4, 'out_channels': 8, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'in_channels': 8, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'in_channels': 16, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 0}
        ]
        maxpool_params = {'kernel_size': 2, 'stride': 2}
        self.conv0 = nn.Conv2d(conv_params[0]['in_channels'],
                               conv_params[0]['out_channels'],
                               conv_params[0]['kernel_size'],
                               conv_params[0]['stride'],
                               conv_params[0]['padding'])

        self.pool = nn.MaxPool2d(maxpool_params['kernel_size'],
                                 maxpool_params['stride'])

        self.conv1 = nn.Conv2d(conv_params[1]['in_channels'],
                               conv_params[1]['out_channels'],
                               conv_params[1]['kernel_size'],
                               conv_params[1]['stride'],
                               conv_params[1]['padding'])

        self.conv2 = nn.Conv2d(conv_params[2]['in_channels'],
                               conv_params[2]['out_channels'],
                               conv_params[2]['kernel_size'],
                               conv_params[2]['stride'],
                               conv_params[2]['padding'])

        self.conv3 = nn.Conv2d(conv_params[3]['in_channels'],
                               conv_params[3]['out_channels'],
                               conv_params[3]['kernel_size'],
                               conv_params[3]['stride'],
                               conv_params[3]['padding'])

        tensor_input_size = 150
        self.conv_output = get_conv_output(conv_params, maxpool_params, tensor_input_size)
        self.fc1 = nn.Linear(self.conv_output, 1000) #change later
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 26)

    def forward(self, img):
        x = self.pool(F.relu(self.conv0(img)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.conv_output) # change later
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x