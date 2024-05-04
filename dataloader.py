# would only run this once per batch size to get consistent data loaders
batch_size = 32
train_loader, val_loader, test_loader = get_data_loader(batch_size, balancer=2000)

def save_data_loader(train_loader, val_loader, test_loader):
    data_loader_path = "/content/gdrive/MyDrive/APS360-Notebooks/Project/data_loader/"
    torch.save(train_loader, data_loader_path + "train_loader.pt")
    torch.save(val_loader, data_loader_path + "val_loader.pt")
    torch.save(test_loader, data_loader_path + "test_loader.pt")

def load_data_loader():
    torch.load("/content/gdrive/MyDrive/APS360-Notebooks/Project/data_loader/train_loader.pt")
    torch.load("/content/gdrive/MyDrive/APS360-Notebooks/Project/data_loader/val_loader.pt")
    torch.load("/content/gdrive/MyDrive/APS360-Notebooks/Project/data_loader/test_loader.pt")
    return train_loader, val_loader, test_loader

save_data_loader(train_loader, val_loader, test_loader)
train_loader, val_loader, test_loader = load_data_loader()