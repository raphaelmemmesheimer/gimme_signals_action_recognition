import os
import pytorch_lightning as pl
import torch
from torch import nn

#import torchvision
from torch.nn import functional as F
from torch.nn import Conv2d, Linear, MaxPool2d

import torchvision
from torchvision import models
from torchvision import transforms
from torch_lr_finder import LRFinder
from torchvision import datasets
import matplotlib.pyplot as plt

import seaborn as sn
import pandas as pd
import numpy as np
from efficientnet_pytorch import EfficientNet

#class SimpleNet(nn.Module):
#    def __init__(self):
#        super(SimpleNet, self).__init__()
#        self.num_classes = 6
#        self.fc1 = nn.Linear(3, 1024)
#        self.fc2 = nn.Linear(1024, 512)
#        self.fc3 = nn.Linear(512, 64)
#        self.fc4 = nn.Linear(64, self.num_classes)
#
#    def forward(self, x):
#        x = x.view(-1)
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = F.relu(self.fc3(x))
#        x = self.fc4(x)
#        return x
#
#class TutNet(nn.Module):
#    def __init__(self):
#        super(TutNet, self).__init__()
#        self.num_classes = 6
#        self.conv1 = nn.Conv2d(3, 6, 5)
#        self.pool = nn.MaxPool2d(2, 2)
#        self.conv2 = nn.Conv2d(6, 16, 5)
#        self.fc1 = nn.Linear(16 * 53 * 53, 120)
#        self.fc2 = nn.Linear(120, 84)
#        self.fc3 = nn.Linear(84, self.num_classes)
#
#    def forward(self, x):
#        try:
#            x = self.pool(F.relu(self.conv1(x)))
#            x = self.pool(F.relu(self.conv2(x)))
#            x = x.view(-1, 16 * 53 * 53)
#            x = F.relu(self.fc1(x))
#            x = F.relu(self.fc2(x))
#            x = self.fc3(x)
#        except Exception as e:
#            print(e, x.shape)
#        return x
#
#class Net(nn.Module):
#    """
#    This architecture seems to work quite fine with the ARIL dataset
#    """
#    def __init__(self):
#        super(Net, self).__init__()
#        self.num_classes = 6
#        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5)
#        self.pool = nn.MaxPool2d(2, 2)
#        self.conv2 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=5)
#        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5)
#
#        self.fc1 = nn.Linear(in_features=256 * 24 * 24, out_features=1024)
#        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
#        self.fc3 = nn.Linear(in_features=1024, out_features=224)
#        self.fc4 = nn.Linear(in_features=224, out_features=self.num_classes)
#
#    def forward(self, x):
#        try:
#            x = self.pool(F.relu(self.conv1(x)))
#            x = self.pool(F.relu(self.conv2(x)))
#            x = self.pool(F.relu(self.conv3(x)))
#            x = x.view(-1, 256 * 24 * 24)
#            x = F.relu(self.fc1(x))
#            x = F.relu(self.fc2(x))
#            x = F.relu(self.fc3(x))
#            #x = F.relu(self.fc3(x))
#            x = self.fc4(x)
#            #x = F.softmax(x)
#
#        except Exception as e:
#            print(e, x.shape)
#        return x
#
#class LiuNet(nn.Module):
#    """
#    inspired by Enhanced skeleton visualization for view invariant human action recognition
#    """
#    def __init__(self):
#        super(LiuNet, self).__init__()
#        self.num_classes = 6
#        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11)
#        self.pool = nn.MaxPool2d(2, 2)
#        self.pool4 = nn.MaxPool2d(4, 4)
#        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5)
#        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
#        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
#        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
#
#        self.fc1 = nn.Linear(in_features=256 * 9 * 9, out_features=4096)
#        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
#        self.fc3 = nn.Linear(in_features=4096, out_features=self.num_classes)
#
#    def forward(self, x):
#        x = self.pool4(F.relu(self.conv1(x)))
#        x = self.pool(F.relu(self.conv2(x)))
#        x = F.relu(self.conv3(x))
#        x = F.relu(self.conv4(x))
#        x = self.pool(F.relu(self.conv5(x)))
#        print(x.shape)
#        x = x.view(-1, 256 * 9 * 9)
#        #x = F.dropout2d(F.relu(self.fc1(x)), p=0.5)
#        #x = F.dropout2d(F.relu(self.fc2(x)), p=0.5)
#
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        #x = F.relu(self.fc3(x))
#        x = self.fc3(x)
#        #x = F.softmax(x)
#        return x


class ClassificationLightningModule(pl.LightningModule):

    def __init__(self, hparams):
        super(ClassificationLightningModule, self).__init__()
        
        print("Hyperparameters: "+ str(hparams))

        self.hparams = hparams
        self.learning_rate = hparams.learning_rate
        self.momentum = hparams.momentum
        self.weight_decay = hparams.weight_decay
        self.pretrained = hparams.pretrained
        self.data_dir = hparams.data_dir
        self.batch_size = hparams.batch_size
        self.model_name = hparams.model_name
        self.optimizer_name = hparams.optimizer_name
        self.one_cycle_policy = hparams.once_cycle_policy
        self.lr_scheduler = hparams.lr_scheduler

        
        
#       scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, hparams.max_epochs)
#       scheduler_warmup = GradualWarmupScheduler(self.optimizer, multiplier=8, total_epoch=10, 
#                                                 after_scheduler=scheduler_cosine)

        self.trainloader, self.valloader, self.testloader = self.init_dataloaders()
        self.classes = self.testloader.dataset.classes
        print(f"Classes: {self.classes},\n Number of classes: {len(self.classes)}")
        self.val_confusion_matrix = torch.zeros([len(self.classes), 
                                                 len(self.classes)], dtype=torch.float)
        
        self.test_confusion_matrix = torch.zeros([len(self.classes), 
                                                  len(self.classes)], dtype=torch.float)
        if self.model_name == "simple":
            self.model = SimpleNet()
        elif self.model_name == "efficientnet":
            if self.hparams.pretrained:
                self.model = EfficientNet.from_pretrained('efficientnet-'+hparams.model_type)
            else:
                self.model = EfficientNet.from_name('efficientnet-'+hparams.model_type)
        elif self.model_name == "liu":
            self.model = LiuNet()
        elif self.model_name == "tut":
            self.model = TutNet()
        elif self.model_name == "custom":
            self.model = Net()
        else:
            self.model = models.__dict__[self.hparams.model_name](pretrained=self.hparams.pretrained) #, num_classes=len(self.classes))

        
#         self.trainloader = DataLoader(CIFAR10(".", train=True, download=True, 
#                                       transform=transforms.ToTensor()), batch_size=32)
#         self.testloader = DataLoader(CIFAR10(".", train=False, download=True, 
#                                       transform=transforms.ToTensor()), batch_size=32)


    def forward(self, x):
        x = self.model.forward(x)
        return x
    
    def init_dataloaders(self):
        common_transforms = []
        train_transforms = []
        test_transforms = []
        #if self.hparams.transform_resize_match:
        common_transforms.append(transforms.Resize((self.hparams.transform_resize,self.hparams.transform_resize)))
        #else:
            #common_transforms.append(transforms.Resize(self.hparams.transform_resize))



        if self.hparams.transform_normalize:
            common_transforms.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

        if self.hparams.transform_random_resized_crop:
            train_transforms.append(transforms.RandomResizedCrop(self.hparams.transform_resize))
        if self.hparams.transform_random_horizontal_flip:
            train_transforms.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))
        if self.hparams.transform_random_rotation:
            train_transforms.append(transforms.RandomRotation(self.hparams.transform_random_rotation_degrees))#, fill=255))
        if self.hparams.transform_random_shear:
            train_transforms.append(torchvision.transforms.RandomAffine(0,
                                                                        shear=(
                                                                            self.hparams.transform_random_shear_x1,
                                                                            self.hparams.transform_random_shear_x2,
                                                                            self.hparams.transform_random_shear_y1,
                                                                            self.hparams.transform_random_shear_y2
                                                                            ),
                                                                        fillcolor=255)) 
        if self.hparams.transform_random_perspective:
            train_transforms.append(transforms.RandomPerspective(distortion_scale=self.hparams.transform_perspective_scale, 
                                         p=0.5, 
                                         interpolation=3)
                                    )
        if self.hparams.transform_random_affine:
            train_transforms.append(transforms.RandomAffine(degrees=(self.hparams.transform_degrees_min,
                                                                     self.hparams.transform_degrees_max),
                                                            translate=(self.hparams.transform_translate_a,
                                                                       self.hparams.transform_translate_b),
                                                            fillcolor=255))


        #common_transforms.append()

        data_transforms = {
        'train': transforms.Compose(common_transforms+train_transforms+[transforms.ToTensor()]),
        'test': transforms.Compose(common_transforms+[transforms.ToTensor()]),
        }

        trainset = datasets.ImageFolder(os.path.join(self.data_dir, "train"),
                                          data_transforms["train"])
        
        testset = datasets.ImageFolder(os.path.join(self.data_dir, "test"),
                                                  data_transforms["test"])
        
        print("Trainset: ",len(trainset), "Testset: ",len(testset))
        
        self.classes = testset.classes
        #train_percentage = 0.8
        #train_items = int(train_percentage*len(trainset))
        #trainset, valset = torch.utils.data.random_split(trainset, [train_items, len(trainset)-train_items])
        
        trainloader = torch.utils.data.DataLoader(trainset, 
                                                  shuffle=True, 
                                                  batch_size=self.batch_size,
                                                  num_workers=4)

        valloader = torch.utils.data.DataLoader(testset, 
                                                shuffle=False, 
                                                batch_size=self.batch_size,
                                                num_workers=4)

        testloader = torch.utils.data.DataLoader(testset, 
                                                 batch_size=self.batch_size, 
                                                 num_workers=4)
        
        return trainloader, valloader, testloader

    def lr_find(self, device="cuda"):
        """
        This method is a pretraining method that plots the result of the learning rate finder
        to find an optimal learning rate. See also 
        * https://github.com/davidtvs/pytorch-lr-finder
        * 
        """
#         with torch.no_grad():
        lr_finder = LRFinder(self.model, self.optimizer, 
                             self.criterion, device=device)
        lr_finder.range_test(self.train_dataloader(), start_lr=0.0000001, end_lr=10, num_iter=100)
        lr_finder.plot() # to inspect the loss-learning rate graph
        lr_finder.reset() # to reset the model and optimizer to their initial state

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        #assert not np.any(np.isnan(x)) TODO
        y_hat = self.forward(x)
        
        loss, acc1, acc2, acc5, tensorboard_logs = self.get_step_metrics(y, y_hat, "train")
        tensorboard_logs["learning_rate"] = self.optimizer.param_groups[0]['lr']
        #if self.lr_scheduler == "one_cycle":
            #self.scheduler.step()
        return {'loss': loss, 'log': tensorboard_logs, 
                'progress_bar': {"lr": self.optimizer.param_groups[0]['lr']}}

    #def training_end(self, outputs):
        # just used for steppina atm
        #if not self.lr_scheduler == "one_cycle":
        #    self.scheduler.step()
        #return
    
    # define what happens for validation here
    def validation_step(self, batch, batch_idx):    
        x, y = batch
        y_hat = self.forward(x)
        loss, acc1, acc2, acc5, tensorboard_logs = self.get_step_metrics(y, y_hat, "val")
        if batch_idx == 0:
            self.log_images(x, "examples/validation")
        #self.val_confusion_matrix = self.update_confusion_matrix(y, y_hat, self.val_confusion_matrix)
    
        return {'val_loss': loss, "val_acc": torch.tensor(acc1),  
                'log': tensorboard_logs} 
    

    def validation_end(self, outputs):
        # OPTIONAL
#         print(outputs)

#         self.scheduler_warmup.step()
    
        self.log_confusion_matrix(self.val_confusion_matrix, name="val-confusion-matrix")
        avg_loss, avg_acc1, avg_acc2, avg_acc5, tensorboard_logs = self.get_mean_metrics(outputs, "val") 
        print(tensorboard_logs)
        return {'val_loss': avg_loss, 'val_acc': avg_acc1, 
                'log': tensorboard_logs}
                #, 'progress_bar': tensorboard_logs}
        
    def test_step(self, batch, batch_idx):
        # OPTIONAL
        with torch.no_grad():
            x, y = batch
            if batch_idx == 0:
                self.log_images(x, "examples/test")
            y_hat = self.forward(x)
            loss, acc1, acc2, acc5, tensorboard_logs = self.get_step_metrics(y, y_hat, "test")

            self.test_confusion_matrix = self.update_confusion_matrix(y, y_hat, self.test_confusion_matrix)
        return {'test_loss': loss,
                'test_acc': torch.tensor(acc1),
                'log': tensorboard_logs}

    def test_end(self, outputs):
        # OPTIONAL        
        avg_loss, avg_acc1, avg_acc2, avg_acc5, tensorboard_logs = self.get_mean_metrics(outputs, "test") 
        print(avg_acc1)

        
        self.log_confusion_matrix(self.test_confusion_matrix, name="test-confusion-matrix")
        return {'test_loss': avg_loss, 'test_acc': avg_acc1, 
                'log': tensorboard_logs}

    def update_confusion_matrix(self, y, y_hat, confusion_matrix):
        """
        Updates a confusion matrix with the current given estimates and gt for use in a step 
        method. 

        y: contains the labels
        y_hat: contains the estimates from a forward path
        confusion matrix: the intermediate confusion matrix.

        returns: updated confusion matrix
        """
        #print(y.shape,y_hat.shape)
        _, pred = torch.max(y_hat, 1)
        #print(y,pred,y_hat)
        #print(len(y),len(y_hat), len(pred))
        for t, p in zip(y.view(-1), pred.view(-1)):
            # This is for pretrained models where in the beginning classes outsite of the class number can be predicted
            if (p >= 0) & (p < len(self.classes)): 
                confusion_matrix[t.long(), p.long()] += 1
        return confusion_matrix

    def log_confusion_matrix(self, confusion_matrix, name="confusion_matrix", normalize=True, show=False):
        try:
            cm = confusion_matrix.numpy()
            if normalize:
                #if cm.sum(1) > 0:
                cm = cm / cm.sum(1)
                #else:
                    #print("cm seems to be empty or malformed currently, this is common in the first epochs")
            df_cm = pd.DataFrame(cm, 
                                 self.classes, 
                                 self.classes) # to get the class replace len(cm) by a list of class_names
            fig = plt.figure(figsize=(30,30))
            sn.set(font_scale=1.4) # for label size
            sn.heatmap(df_cm, annot=True, fmt='.2f', annot_kws={"size": 16}) # font size
            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            torch_img = torch.from_numpy(image_from_plot)

            self.logger.experiment.add_image(name, torch_img, 0, dataformats='HWC')
            #print("Saving confusion matrix "+self.hparams.save_path+"/confusion_matrix"+"as svg, png, pkl")
            #plt.savefig(self.hparams.save_path+"/confusion_matrix"+str(self.current_epoch).zfill(5)+".svg")
            #plt.savefig(self.hparams.save_path+"/confusion_matrix"+str(self.current_epoch).zfill(5)+".png")
            np.save(self.hparams.save_path+"/"+name+"confusion_matrix"+str(self.current_epoch).zfill(5)+".pkl", confusion_matrix.numpy())
            plt.close()
        except Exception as e:
            print("Error", e)
        #kreturn confusion_matrix.numpy()
        #plt.show()

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        #print(self.learning_rate)
        if self.optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                             lr=self.learning_rate, 
                                             momentum=self.momentum,
                                             weight_decay=self.weight_decay)
        if self.optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                              lr=self.learning_rate, 
                                              weight_decay=self.weight_decay)
        if self.optimizer_name == "RMSProp":
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                              lr=self.learning_rate, 
                                              weight_decay=self.weight_decay,
                                              momentum=self.momentum)
        self.criterion = torch.nn.CrossEntropyLoss()

        if self.lr_scheduler == "one_cycle":
            # Note this is not yet tuned to work in general
            #self.scheduler = OneCycleLR(self.optimizer, num_steps=27*20, lr_range=(0.001, 0.1))
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.hparams.max_lr, steps_per_epoch=len(data_loader), epochs=self.hparams.min_expochs)
        elif self.lr_scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hparams.lr_scheduler_step_size, gamma=self.hparams.lr_scheduler_gamma)
        return [self.optimizer], [self.scheduler]
    
    def get_step_metrics(self, y, y_hat, prefix="train"):
        """
        This function returns the Loss, Top-1-, Top-2-, Top-5-Accuracy
        from labels and estimations of a forward pass.
        
        :y: Labels
        :y_hat: Estimates
        :prefix: Prefix name for the logs(default: train)
        
        Returns:
        
        :loss: Loss based on the class criterion
        :acc1: Accuracy
        :acc2: Top-2 Accuracy
        :acc5: Top-5 Accuracy
        :tensorboard_logs: All metrics in a loggable dictionary format
        """
        loss = self.criterion(y_hat, y)
        acc1, acc2, acc5 = self.__accuracy(y_hat, y, topk=(1, 2, 5))
        tensorboard_logs = {f'loss/{prefix}_loss': loss, 
                            f'acc/{prefix}_acc1': acc1, 
                            f'acc/{prefix}_acc2': acc2, 
                            f'acc/{prefix}_acc5': acc5}
        
        return loss, acc1, acc2, acc5, tensorboard_logs
    
    def get_mean_metrics(self, outputs, prefix="train"):
        avg_loss = torch.stack([x["log"][f'loss/{prefix}_loss'] for x in outputs]).mean()
        avg_acc1 = torch.stack([x["log"][f'acc/{prefix}_acc1'] for x in outputs]).mean()
        avg_acc2 = torch.stack([x["log"][f'acc/{prefix}_acc2'] for x in outputs]).mean()
        avg_acc5 = torch.stack([x["log"][f'acc/{prefix}_acc5'] for x in outputs]).mean()
        
        tensorboard_logs = {f'loss/{prefix}_loss': avg_loss, 
                            f'acc/{prefix}_acc1': avg_acc1, 
                            f'acc/{prefix}_acc2': avg_acc2, 
                            f'acc/{prefix}_acc5': avg_acc5}

        return avg_loss, avg_acc1, avg_acc2, avg_acc5, tensorboard_logs
    
    
    @classmethod
    def __get_correct(self, output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
    #             print(correct)
        return correct

    
    @classmethod
    def __accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            correct = self.__get_correct(output, target, topk)
            batch_size = target.size(0)
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def show_sample_images(self):
        """
        This method shows a batch from the train data
        """
        imgs, labels = next(iter(self.train_dataloader()))
        grid = torchvision.utils.make_grid(imgs,padding=10, nrow = 5)
        plt.imshow(grid.permute(1, 2, 0))
        print(labels)
    
    def log_images(self, x, name="examples", num=6):
        sample_imgs = x[:num]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(name, grid, 0)

        
    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return self.trainloader

    @pl.data_loader
    def val_dataloader(self):
        return self.valloader
    
    @pl.data_loader
    def test_dataloader(self):
        return self.testloader
