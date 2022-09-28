import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchaudio
import sys

import matplotlib.pyplot as plt
import IPython.display as ipd
from torchaudio.datasets import SPEECHCOMMANDS
from tqdm import tqdm
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from alphafed import logger
from alphafed.fed_avg import FedSGDScheduler
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 运行之前看这里的path怎么改动


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=False)
# FileNotFoundError: [Errno 2] No such file or directory: './123\\SpeechCommands\\speech_commands_v0.02\\validation_list.txt'

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            print(self._path)   # 下划线不会影响使用，通常作为类内提示：用于类内名称
            print(filepath)
            with open(filepath) as fileobj:   # strip 去掉空格
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]
            # nrompath:规范化路径
#         if subset == "validation":
#             self._walker = load_list("validation_list.txt")
#             print(self._walker)
        if subset == "testing":
            self._walker = load_list("testing_list.txt")
#             print(self._walker)
        elif subset == "training":
            excludes = load_list("train_list.txt")
            excludes = set(excludes)  # 设置为集合
            # 这里排除test和val中的文件。但这里self._walker怎么来的？
            self._walker = [w for w in self._walker if w not in excludes]


train_set = SubsetSC("training")
# test_set = SubsetSC("testing")
waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

labels = sorted(list(set(datapoint[2] for datapoint in train_set)))  # 所有的类


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


# transfromed.shape[0] = 1
model = M5(n_input=1, n_output=len(labels))
model.to(device)


def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))    # 这里就一个数


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    # 用0填充，使一个batch有相同的长度
    batch = [item.t() for item in batch]  # 转置
    batch = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]   # 每个波形（tensor类型）加入到列表tensor中
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)   # 得到tensor的batch
    targets = torch.stack(targets)  # 连接，在新的维度进行堆叠   得到targets

    return tensors, targets


class DemoFedSGD(FedSGDScheduler):

    def __init__(self,
                 min_clients: int,
                 name: str = None,
                 max_rounds: int = 0,
                 calculation_timeout: int = 300,
                 log_rounds: int = 0,
                 is_centralized: bool = True,
                 batch_size: int = 64,
                 learning_rate: float = 0.01,
                 momentum: float = 0.5) -> None:
        super().__init__(min_clients=min_clients,
                         name=name,
                         max_rounds=max_rounds,
                         calculation_timeout=calculation_timeout,
                         log_rounds=log_rounds,
                         is_centralized=is_centralized)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.loss = []
        self._time_metrics = None

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = 42
        self.optimizer = optim.Adam(
            model.parameters(), lr=0.01, weight_decay=0.0001)
        # reduce the learning after 20 epochs by a factor of 10
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.1)

        self.sample_rate = 16000
        self.new_sample_rate = 8000
        self.transform = torchaudio.transforms.Resample(
            orig_freq=self.sample_rate, new_freq=self.new_sample_rate)
        transformed = self.transform(waveform)
        if self.device == "cuda":
            self.num_workers = 1
            self.pin_memory = True
        else:
            self.num_workers = 0
            self.pin_memory = False
        torch.manual_seed(self.seed)

    def make_model(self) -> nn.Module:
        model = M5()
        return model

    def make_optimizer(self) -> optim.Optimizer:
        assert self.model, 'must initialize model first'
        return optim.SGD(self.model.parameters(),
                         lr=self.learning_rate,
                         momentum=self.momentum)

    def make_train_dataloader(self) -> DataLoader:
        dataset = SubsetSC("training")
        return DataLoader(dataset=dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=collate_fn,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory
                          )

    def make_test_dataloader(self) -> DataLoader:
        dataset = SubsetSC("testing")
        return DataLoader(dataset=dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=collate_fn,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory
                          )

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.model.load_state_dict(state_dict)

    def number_of_correct(self, pred, target):
        # count number of correct predictions
        return pred.squeeze().eq(target).sum().item()

    def get_likely_index(self, tensor):   # 匹配最接近的类别
        # find most likely label index for each element in the batch
        return tensor.argmax(dim=-1)

    def validate_context(self):
        super().validate_context()
        train_loader = self.make_train_dataloader()
        assert train_loader and len(
            train_loader) > 0, 'failed to load train data'
        logger.info(
            f'There are {len(train_loader.dataset)} samples for training.')
        test_loader = self.make_test_dataloader()
        assert test_loader and len(
            test_loader) > 0, 'failed to load test data'
        logger.info(
            f'There are {len(test_loader.dataset)} samples for testing.')

    def train(self):
        self.model.train()
        train_loader = self.make_train_dataloader()
        for batch_idx, (data, target) in enumerate(train_loader):   # 这里迭代返回的是data和target

            data = data.to(device)
            target = target.to(device)

            # apply transform and model on whole batch directly on device
            data = self.transform(data)
            output = model(data)

            # negative log-likelihood for a tensor of size (batch x 1 x n_output)
            loss = F.nll_loss(output.squeeze(), target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # print training stats
            # if batch_idx % log_interval == 0:
            #     print(
            #         f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

            # record loss
            self.losses.append(loss.item())

            # log_interval = 20
            # n_epoch = 5

            # pbar_update = 1 / (len(train_loader) + len(test_loader))
            # losses = []

            # # The transform needs to live on the same device as the model and the data.
            # transform = transform.to(device)
            # with tqdm(total=n_epoch) as pbar:
            #     for epoch in range(1, n_epoch + 1):
            #         train(model, epoch, log_interval)
            #         test(model, epoch)
            #         scheduler.step()

            # # Let's plot the training loss versus the number of iteration.
            # plt.plot(losses);
            # plt.title("training loss");
    @register_metrics(name='timer', keys=['run_time'])
    @register_metrics(name='test_results', keys=['average_loss', 'accuracy', 'correct_rate'])
    def test(self):
        start = time()
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            test_loader = self.make_test_dataloader()
            for data, target in test_loader:

                data = data.to(device)
                target = target.to(device)

                # apply transform and model on whole batch directly on device
                data = self.transform(data)
                output = model(data)
                test_loss += F.nll_loss(output, labels, reduction='sum').item()
                pred = self.get_likely_index(output)
                correct += self.number_of_correct(pred, target)

                # update progress bar
                # pbar.update(pbar_update)

            # print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")

        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
        correct_rate = 100. * accuracy
        logger.info(f'Test set: Average loss: {test_loss:.4f}')
        logger.info(
            f'Test set: Accuracy: {accuracy} ({correct_rate:.2f}%)'
        )

        end = time()
        self.get_metrics('timer').append_metrics_item(
            {'run_time': end - start})
        self.get_metrics('test_results').append_metrics_item({
            'average_loss': test_loss,
            'accuracy': accuracy,
            'correct_rate': correct_rate
        })


scheduler = DemoFedSGD(min_clients=2,
                       name='SPEECH_DEMO_ONE',
                       max_rounds=5,
                       log_rounds=1,
                       calculation_timeout=120)
scheduler.launch_task(task_id='YOUR_TASK_ID')
