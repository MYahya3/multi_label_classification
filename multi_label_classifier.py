import torch,torchvision
from torch.utils.data import DataLoader
from multlabel_dataloader import MultiLabelDataset
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np


                                                #### Example ####
num_classes = 10
csv_dir = "D:/GitHub/pytorch/multi-label_classification/data/Labels_file.csv"
data_dir = "D:/GitHub/pytorch/multi-label_classification/data/Images"

# Transformer
transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

# Load Data from dir, transform and return batches
dataset = MultiLabelDataset(csv_dir=csv_dir,data_dir=data_dir, transform=transform)
# Split Dataset into train and test
train_data, test_data = torch.utils.data.random_split(dataset, [270,80])

train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
test_dl = DataLoader(test_data,batch_size=32, shuffle=False)

def train(model, optimizer, loss_fn, train_dl, val_dl,nb_classes, epochs=1):

    for epoch in range(1, epochs+1):

                                     # --- TRAIN AND EVALUATE ON TRAINING SET --- #
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0

        for imgs, labels in train_dl:

            pred = model(imgs)
            total += (labels.size(0)) * nb_classes
            loss = loss_fn(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # For Loss
            train_loss += loss.item() * imgs.size(0)
            # For accuracy
            pred = (torch.sigmoid(pred).round().detach().numpy())
            labels = labels.numpy()
            train_correct += (pred == labels).sum()
        # Final accuracy and loss
        train_acc = train_correct / total
        train_loss  = train_loss / total

                                          ### ---- EVALUATE ON VALIDATION SET ---- ###
        model.eval()
        val_loss       = 0.0
        val_correct    = 0
        total          = 0

        for imgs, labels in val_dl:
            pred = model(imgs)
            total += (labels.size(0)) * nb_classes
            loss = loss_fn(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # For loss
            val_loss += loss.item() * imgs.size(0)
            # For accuracy
            pred = (torch.sigmoid(pred).round().detach().numpy())
            labels = labels.numpy()
            val_correct += (pred == labels).sum()
        # Final accuracy and loss
        val_acc = val_correct / total
        val_loss = val_loss / total

        print('Epoch %3d/%3d, train loss: %5.2f | train acc: %5.2f | test loss: %5.2f | test acc: %5.2f' % \
                (epoch, epochs, train_loss,train_acc,val_loss, val_acc))


# Pre-trained Resnet50 Model
model = torchvision.models.resnet50()
for param in model.parameters():
    param.requires_grad = False  # Freeze grads
# Fully Connected Layer
model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)

# Loss_funciton and optimizer
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and Validation Loop
train(model, optimizer, loss_fn,train_dl,test_dl,nb_classes=10,epochs=30)
# torch.save(model.state_dict("E:/GitHub/pytorch_projects/multi-label_classification/data/multilabel_model.pth"))
