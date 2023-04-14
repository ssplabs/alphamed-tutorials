from time import time
from typing import Dict

import torch
import torch.nn.functional as F
from alphafed import get_dataset_dir, logger
from torch.nn import Conv2d, Dropout2d, Linear, Module
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .scheduler import SimpleFedAvgScheduler


class ConvNet(Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = Dropout2d()
        self.fc1 = Linear(in_features=320, out_features=50)
        self.fc2 = Linear(in_features=50, out_features=10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class SimpleTaskScheduler(SimpleFedAvgScheduler):

    def __init__(self,
                 clients: int,
                 rounds: int,
                 batch_size: int,
                 learning_rate: float,
                 momentum: float) -> None:
        super().__init__(clients=clients, rounds=rounds)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = 42
        torch.manual_seed(self.seed)

    def build_model(self) -> Module:
        model = ConvNet()
        return model.to(self.device)

    def build_optimizer(self, model: Module) -> Optimizer:
        assert self.model, 'must initialize model first'
        return SGD(self.model.parameters(),
                   lr=self.learning_rate,
                   momentum=self.momentum)

    def build_train_dataloader(self) -> DataLoader:
        return DataLoader(
            datasets.MNIST(
                get_dataset_dir(self.task_id),
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            ),
            batch_size=self.batch_size,
            shuffle=True
        )

    def build_test_dataloader(self) -> DataLoader:
        return DataLoader(
            datasets.MNIST(
                get_dataset_dir(self.task_id),
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            ),
            batch_size=self.batch_size,
            shuffle=False
        )

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.model.load_state_dict(state_dict)

    def train_an_epoch(self) -> None:
        self.model.train()
        for data, labels in self.train_loader:
            data: torch.Tensor
            labels: torch.Tensor
            data, labels = data.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, labels)
            loss.backward()
            self.optimizer.step()

    def test(self):
        start = time()
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                output: torch.Tensor = self.model(data)
                test_loss += F.nll_loss(output, labels, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = correct / len(self.test_loader.dataset)
        correct_rate = 100. * accuracy
        logger.info(f'Test set: Average loss: {test_loss:.4f}')
        logger.info(
            f'Test set: Accuracy: {accuracy} ({correct_rate:.2f}%)'
        )

        end = time()

        self.tb_writer.add_scalar('timer/run_time', end - start, self.round)
        self.tb_writer.add_scalar('test_results/average_loss', test_loss, self.round)
        self.tb_writer.add_scalar('test_results/accuracy', accuracy, self.round)
        self.tb_writer.add_scalar('test_results/correct_rate', correct_rate, self.round)


scheduler = SimpleTaskScheduler(clients=2,
                                rounds=5,
                                batch_size=128,
                                learning_rate=0.01,
                                momentum=0.9)
scheduler.submit(task_id='task_id')
