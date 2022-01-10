from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Dataset():
    def __init__(self,
                 name=None,
                 batch_size=32,
                 intensity=1.0):
        assert name is not None
        self.name = name
        self.batch_size = batch_size
        self.intensity = intensity
        self.channel, self.height, self.width = 1, 28, 28
        self.class_num = 10

    @property
    def input_shape(self):
        return (self.channel, self.height, self.width)

    def __call__(self, *args, **kwargs):
        """
        データ取得（Pytorchはchannel first, tensorflowはchannel last)
        :param args:
        :param kwargs:
        :return:
        """
        train_loader = DataLoader(
                datasets.MNIST('./data',
                               train=True,
                               download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Lambda(lambda x: x * self.intensity)
                               ])),
                batch_size=self.batch_size,
                shuffle=True)

        test_loader = DataLoader(
            datasets.MNIST('./data',
                           train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: x * self.intensity)
                           ])),
            batch_size=self.batch_size*3,
            shuffle=True)

        return train_loader, test_loader