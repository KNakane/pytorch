import os
import time
import datetime
import torch
import torchvision
import torch.nn as nn
from torchsummary import summary
from collections import OrderedDict
from Utility.logger import get_logger
from Utility.optimizer import get_optimizer
from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(self,
                 model,
                 dataset,
                 epoch=1000,
                 optimizer="SGD",
                 learning_rate=0.001,
                 init_model=None,
                 save_checkpoint_steps=100,
                 checkpoints_to_keep=5,
                 debug=False,
                 ):
        self.dataset = dataset
        self.n_epoch = epoch
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
        self.lr = learning_rate
        self.optimizer = get_optimizer(name=optimizer, model=self.model, lr=self.lr)
        self._load_save_model(init_model)   # load init model
        self.result_dir, self.model_dir = self._get_result_dir()
        self.logger = get_logger(outdir=self.result_dir, debug=debug)
        self.summary_writer = SummaryWriter(log_dir=self.result_dir + "/tfboard",)
        self.save_checkpoint_steps = save_checkpoint_steps
        self.checkpoints_to_keep = checkpoints_to_keep
        self.log_interval = 1000

    def _get_result_dir(self):
        path = os.path.join("results/" + datetime.datetime.now().strftime("%y%m%d_%H%M%S"))
        if not os.path.exists(path):
            os.makedirs(path)
        save_dir = os.path.join(path, "model")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return path, save_dir

    def _load_save_model(self, model_dir):
        """

        :param model_dir:
        :return:
        """
        if model_dir is not None:
            checkpoint = torch.load(model_dir)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return

    def _save_model(self, epoch):
        filenum = sum(
            os.path.isfile(os.path.join(self.model_dir, name)) for name in os.listdir(self.model_dir)
        )

        if filenum > self.checkpoints_to_keep - 1:
            os.listdir(self.model_dir)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.model_dir + "/model_{}.pth".format(epoch))

    def __call__(self, *args, **kwargs):
        #
        summary(self.model, (self.dataset.channel, self.dataset.height, self.dataset.width))
        dummy_input = torch.zeros(1, self.dataset.channel, self.dataset.height, self.dataset.width)
        self.summary_writer.add_graph(self.model, input_to_model=dummy_input, verbose=False)

        # get dataset
        train_loader, test_loader = self.dataset()

        for epoch in range(1, self.n_epoch + 1):
            start_time = time.time()
            train_images, train_loss, train_accuracy = self.train(epoch, train_loader)
            time_per_epoch = time.time() - start_time

            test_images, test_loss, test_accuracy = self.eval(test_loader)

            # Training results
            metrics = OrderedDict({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "time/epoch": time_per_epoch
            })

            other_metrics = OrderedDict({
                "train_image": train_images,
                "test_image": test_images
            })

            self.epoch_end(metrics=metrics, other=other_metrics)
        return

    def train(self, epoch, train_loader):
        self.model.train()
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_func(output, target)
            loss.backward()
            self.optimizer.step()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        loss = loss.item() / len(data)
        accuracy = 100. * correct / len(train_loader.dataset)
        return data, loss, accuracy

    def eval(self, test_loader):
        self.model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += self.loss_func(output, target)  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        return data, loss, accuracy

    def epoch_end(self, metrics, other=None):
        epoch = metrics['epoch']
        self.summary_writer.add_scalar('detail/time_per_step', metrics['time/epoch'], epoch)
        #self.summary_writer.add_scalar('detail/learning_rate', learning_rate)
        self.summary_writer.add_scalar('train/loss', metrics['train_loss'], epoch)
        self.summary_writer.add_scalar('train/accuracy', metrics['train_accuracy'], epoch)
        self.summary_writer.add_scalar('test/loss', metrics['test_loss'], epoch)
        self.summary_writer.add_scalar('test/accuracy', metrics['test_accuracy'], epoch)
        if other is not None and ('train_image' in other and len(other['train_image'].shape) == 4):
            train_image = torchvision.utils.make_grid(other['train_image'])
            test_image = torchvision.utils.make_grid(other['test_image'])
            self.summary_writer.add_image('train/image', train_image, epoch)
            self.summary_writer.add_image('test/image', test_image, epoch)

        self.logger.info(
            "Epoch:{} train_loss:{:.4f} train_acc:{:.2f}% test_loss:{:.4f} test_acc:{:.2f}% time/epoch:{:.2f}sec".format(
                epoch, metrics['train_loss'], metrics['train_accuracy'],
                metrics['test_loss'], metrics['test_accuracy'], metrics['time/epoch']
            )
        )

        if epoch % self.save_checkpoint_steps == 0:
            self._save_model(epoch=epoch)

        del metrics
        return
