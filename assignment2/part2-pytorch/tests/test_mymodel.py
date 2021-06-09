import torch
import torchvision
import torchvision.transforms as transforms
import unittest

from models import *

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]

    _, pred = torch.max(output, dim=-1)

    correct = pred.eq(target).sum() * 1.0

    acc = correct / batch_size

    return acc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TestMyModel(unittest.TestCase):
    """ The class containing all test cases for this assignment"""

    def setUp(self):
        """Define the functions to be tested here."""
        model = model = MyModel()
        model.load_state_dict(torch.load('./checkpoints/mymodel.pth'))

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=100, shuffle=False, num_workers=2)

        self.acc = AverageMeter()

        for data, target in test_loader:
            out = model(data)
            batch_acc = accuracy(out, target)
            self.acc.update(batch_acc, out.shape[0])

    def test_accuracy_easy(self):
        self.assertGreater(self.acc.avg, 0.5)

    def test_accuracy_medium(self):
        self.assertGreater(self.acc.avg, 0.6)

    def test_accuracy_hard(self):
        self.assertGreater(self.acc.avg, 0.7)

