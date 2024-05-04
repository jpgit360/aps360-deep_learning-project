#using a KNN Classifier
#https://pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/ doesnt use sklearn?

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Define the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
# one hot encoding for all tyes of signs
signs_dict = {
    'stop': (1.0, 0.0, 0.0, 0.0),
    'signal_ahead': (0.0, 1.0, 0.0, 0.0),
    'ped_cross': (0.0, 0.0, 1.0, 0.0),
    'keep_right': (0.0, 0.0, 0.0, 1.0),
}

path = f'/content/gdrive/MyDrive/dataset_imgs'

# Reload data ?
dataset = load_from_drive(path, signs_dict, transform)


# split data into features and labels
X = np.array([data[0] for data in dataset])  # Features (flattened pixel values)
y = np.array([data[1] for data in dataset])  # Labels (one-hot encoded)

# Shuffle data
indices = np.arange(len(dataset))
np.random.seed(1000)  # Fixed numpy random seed for reproducible shuffling
np.random.shuffle(indices)
X_shuffled = X[indices]
y_shuffled = y[indices]

print(len(dataset))
# Define training and testing splits
train_split = int(0.8 * len(dataset))  # 80% for training
X_train, X_test = X_shuffled[:train_split], X_shuffled[train_split:]
y_train, y_test = y_shuffled[:train_split], y_shuffled[train_split:]
print(X_train.shape)
print(y_train.shape)

# Train the classifier

#resize to two dimensions
num_samples = X_train.shape[0]
num_features = np.prod(X_train.shape[1:])
X_train = X_train.reshape((num_samples, num_features))

#resize to two dimensions
num_samples = X_test.shape[0]
num_features = np.prod(X_test.shape[1:])
X_test = X_test.reshape((num_samples, num_features))

knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)

print(y_pred)
print(y_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of KNN baseline model:", accuracy)

transform = transforms.Compose([
        transforms.ToTensor()
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

# one hot encoding for all tyes of signs
signs_dict = {
    'stop': (1.0, 0.0, 0.0, 0.0),
    'signal_ahead': (0.0, 1.0, 0.0, 0.0),
    'ped_cross': (0.0, 0.0, 1.0, 0.0),
    'keep_right': (0.0, 0.0, 0.0, 1.0),
}

path = f'/content/gdrive/MyDrive/dataset_imgs'

# Reload data ?
dataset = load_from_drive(path, signs_dict, transform)

# split data into features and labels
X = np.array([data[0] for data in dataset])  # Features (flattened pixel values)
y = np.array([data[1] for data in dataset])  # Labels (one-hot encoded)

# Shuffle data
indices = np.arange(len(dataset))
np.random.seed(1000)  # Fixed numpy random seed for reproducible shuffling
np.random.shuffle(indices)
X_shuffled = X[indices]
y_shuffled = y[indices]

# Define training and testing splits
train_split = int(0.8 * len(dataset))  # 80% for training
X_train, X_test = X_shuffled[:train_split], X_shuffled[train_split:]
y_train, y_test = y_shuffled[:train_split], y_shuffled[train_split:]


#Calculate the sum of RGB values along each column (color)
def avg_rgb_value(image_tensor):
  image_tensor = image_tensor.reshape(3, -1).T
  total_pixels = image_tensor.shape[0]
  r = 0
  g = 0
  b = 0
  for row in image_tensor:
    r += float(row[0])
    g += float(row[1])
    b += float(row[2])

  percent_red = r / total_pixels
  percent_green = g / total_pixels
  percent_blue = b / total_pixels
  avg_list = [percent_red, percent_green, percent_blue]
  return np.array(avg_list)

def train_and_test(train_data, train_label, test_data, test_label):
  # stop = np.array([0,0,0])
  # signal_ahead = np.array([0,0,0])
  # ped_cross = np.array([0,0,0])
  # keep_right = np.array([0,0,0])
  class_array = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
  num_stop = 0.0
  num_signal_ahead = 0.0
  num_ped_cross= 0.0
  num_keep_right = 0.0

  for k, img in enumerate(train_data):
    rgb_avg = avg_rgb_value(img)

    sign_type = np.argmax(train_label[k])
    #print(sign_type)
    if sign_type == 0:
      class_array[0] = np.add(class_array[0], rgb_avg)
      num_stop+=1.0

    elif sign_type == 1:
      class_array[1] = np.add(class_array[1], rgb_avg)
      num_signal_ahead+=1.0

    elif sign_type == 2:
      class_array[2] = np.add(class_array[2], rgb_avg)
      num_ped_cross+=1.0

    elif sign_type == 3:
      class_array[3] = np.add(class_array[3], rgb_avg)
      num_keep_right+=1.0

  print(f"Num stops: {num_stop}")
  print(f"Num signal_ahead: {num_signal_ahead}")
  print(f"Num ped_cross: {num_ped_cross}")
  print(f"Num keep_right: {num_keep_right}")

  #mean rgb values for every class
  class_array[0] = class_array[0] / float(num_stop)
  class_array[1] = class_array[1] / float(num_signal_ahead)
  class_array[2] = class_array[2] / float(num_ped_cross)
  class_array[3] = class_array[3] / float(num_keep_right)
  print(class_array[0])
  print(class_array[1])
  print(class_array[2])
  print(class_array[3])

  correct = 0
  for j, img in enumerate(test_data):
    actual_sign_type = np.argmax(test_label[j])
    test_rgb_avg = avg_rgb_value(img)

    lowest_mse = 999999
    lowest_index = -1
    #find the most evenly matched mse
    for i, sign_class in enumerate(class_array):
      squared_diff = (test_rgb_avg - sign_class) ** 2
      mse = np.mean(squared_diff)
      if mse < lowest_mse:
        lowest_mse = mse
        lowest_index = i

    if lowest_index == actual_sign_type:
      correct += 1

  accuracy = correct / len(test_data)
  print(f"test accuracy is {accuracy}")



train_and_test(X_train, y_train, X_test, y_test)

