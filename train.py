# import the necessary packages
import copy
import csv
import os
import time
import cv2

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import gest_model

from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

CATEGORIES = ['Direita', 'Esquerda', 'Frente', 'Parado', 'Tras']

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, writer, last_epoch, num_epochs, checkpoint_path, num_outputs):

    since = time.time()

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    history = []
    history_cols = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_lfw_auc']

    loss_log = []
    acc_log = []

    val_loss_log = []
    val_acc_log = []

    if last_epoch > 0:
        with open(f'{checkpoint_path}/history.csv', 'r+') as history_file:
            csv_reader = csv.DictReader(history_file, history_cols)
            history = [dict(row) for row in csv_reader][1:]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    min_loss = float('inf')
    stop_cont = 0
    delta=0.001
    patience = 10
    final_epoch = 0


    for epoch in range(last_epoch+1, num_epochs):

        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        history.append({'epoch': epoch})

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            i = 0
            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                # print('input: ', inputs)
                # inputs = inputs/255.0
                # print('norm: ', inputs)
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model.forward(inputs)
                    if num_outputs == 1:
                        preds = torch.gt(outputs.squeeze(1), 0.5).type_as(labels.data)
                        loss = criterion(outputs.squeeze(1), labels.type_as(outputs))
                    else:
                        _, preds = torch.max(outputs, 1)

                        loss = criterion(outputs, labels)
                        # print(labels.type_as(outputs))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]


            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                writer.add_scalars('metric/acc', {'train': epoch_acc}, epoch)
                writer.add_scalars('loss/loss', {'train': epoch_loss}, epoch)

                history[epoch]['train_loss'] = epoch_loss
                history[epoch]['train_acc'] = epoch_acc

                loss_log.append(epoch_loss)
                acc_log.append(epoch_acc)

            else:

                writer.add_scalars('metric/acc', {'val': epoch_acc}, epoch)
                writer.add_scalars('loss/loss', {'val': epoch_loss}, epoch)
                # writer.add_scalars('val/lfw_auc', {'auc': auc}, epoch)

                history[epoch]['val_loss'] = epoch_loss
                history[epoch]['val_acc'] = epoch_acc
                # history[epoch]['val_lfw_auc'] = auc

                val_loss_log.append(epoch_loss)
                val_acc_log.append(epoch_acc)

            if phase == 'val':
                best_acc = epoch_acc
                if(running_loss >= min_loss + delta):
                    stop_cont +=1
                    print(f'EarlyStopping counter: {stop_cont} \
                                                out of {patience}')
                    if stop_cont >= patience:
                        print("Early Stopping! \t Training Stopped")
                        # final_epoch = epoch




                else:
                    stop_cont = 0
                    min_loss = running_loss
                    print('Saving best model')
                    best_model_wts = copy.deepcopy(model.state_dict())

        # scheduler.step()

        with open(f'{checkpoint_path}/history.csv', 'w') as history_file:
            csv_writer = csv.DictWriter(history_file, history_cols)
            csv_writer.writeheader()
            csv_writer.writerows(history)

        # if epoch % 2 == 0:
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            # 'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }

        torch.save(checkpoint, f'{checkpoint_path}/epoch{epoch}.pth')
        final_epoch = epoch

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # print('Best val Acc epoch: {:4f}'.format(epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_log, acc_log, val_loss_log, val_acc_log, final_epoch


def main():

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.Grayscale(num_output_channels=1),
            # transforms.RandomRotation(10),
            # transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=((1, 1.15)), shear=20),
            transforms.ToTensor(),
            # transforms.Normalize((0.5), (0.5))
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0.5, 0.5, 0.5],
            #                      std=[0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            # transforms.Normalize((0.5), (0.5))
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0.5, 0.5, 0.5],
            #                      std=[0.5, 0.5, 0.5])
        ])
    }

    # data_dir = 'dataset/Train_Gest_Dataset_Resized'
    # data_dir = 'dataset/Train_Gest_Dataset_Resized'
    data_dir = '/home/gabriel_carvalho/teste/dataset/Train_Gest_Dataset_Resized'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                        for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
                        for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(image_datasets['train'].class_to_idx)
    print(device)


    # base_model = torchvision.models.mobilenet_v2(pretrained=True)
    # base_model = torchvision.models.mnasnet1_0(pretrained=True)
    # base_model = torchvision.models.mnasnet0_5(pretrained=True)
    base_model = gest_model.GestModel()

    for param in base_model.parameters():
        param.requires_grad = True

    '''
    m1 - 50 epocas
    m2 - 100 epocas
    m3 - 100 epocas
    m4 - 100 epocas
    m5 - 1000 epocas
    m6 - 100 epocas
    '''

    num_outputs = 5

    # base_model.classifier = torch.nn.Sequential(
    # # torch.nn.AvgPool2d((7, 7)),
    # # torch.nn.Flatten(),
    # torch.nn.Linear(1280, 128),
    # torch.nn.Sigmoid(),
    # torch.nn.Dropout(),
    # torch.nn.Linear(128, num_outputs),
    # torch.nn.Softmax(1)
    # )

    # for m in base_model.classifier.modules():
    #     if isinstance(m, torch.nn.Linear):
    #         torch.nn.init.kaiming_normal_(m.weight, mode="fan_out",
    #                                     nonlinearity="sigmoid")
    #         torch.nn.init.zeros_(m.bias)




    base_model.to(device)

    if num_outputs == 1:
        criterion = torch.nn.BCELoss() #torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(base_model.parameters(), lr=0.0010, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    # optimizer = torch.optim.Adam(base_model.parameters(), lr=0.0008, weight_decay=0.001/30)
    scheduler = None

    writer = SummaryWriter('model/f36/logs')

    model, loss_log, acc_log, val_loss_log, val_acc_log, final_epoch = train_model(
        model=base_model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        device=device,
        writer=writer,
        last_epoch=-1,
        num_epochs=50,
        checkpoint_path='model/f36',
        num_outputs=num_outputs)

    print('Ultima epoca: ', final_epoch)

    N = final_epoch+1
    plt.style.use("ggplot")
    fig = plt.figure()
    ax = fig.add_subplot(yticks=[])
    plt.plot(np.arange(0, N), loss_log, label="train_loss")
    plt.plot(np.arange(0, N), val_loss_log, label="val_loss")
    plt.plot(np.arange(0, N), acc_log, label="train_acc")
    plt.plot(np.arange(0, N), val_acc_log, label="val_acc")
    ax.set_yticks(np.arange(0, 1.1, step=0.1))
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('Model_f36.jpg')
    plt.show()



if __name__ == '__main__':
    main()
