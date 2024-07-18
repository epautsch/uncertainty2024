import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import time


parser = argparse.ArgumentParser(description='Fusion Ensembles NN')
parser.add_argument('--gpus', type=int, default=1, help='num gpus to use')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int, default=2, help='epochs')
parser.add_argument('--num_ensembles', type=int, default=3, help='number of ensembles')
args = parser.parse_args()

gpus = args.gpus
batch_size = args.batch_size
epochs = args.epochs
num_ensembles = args.num_ensembles

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.25)
        self.flatten = nn.Flatten()

        self._initialize_weights()
        dummy_input = torch.zeros(1, 1, 28, 28)
        with torch.no_grad():
            self.flattened_dim = self._forward_features(dummy_input).shape[1]

        self.fc = nn.Linear(self.flattened_dim, num_classes)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _forward_features(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.flatten(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.fc(x)
        return x


class FusionNet(nn.Module):
    def __init__(self, num_ensembles, num_classes=10):
        super(FusionNet, self).__init__()
        self.ensembles = nn.ModuleList([ConvNet(num_classes) for _ in range(num_ensembles)])
        self.fc1 = nn.Linear(num_ensembles * num_classes, num_classes)
        self.fc2 = nn.Linear(num_ensembles * num_classes, num_classes)

    def forward(self, x):
        ensemble_outputs = [ensemble(x) for ensemble in self.ensembles]
        concatenated = torch.cat(ensemble_outputs, dim=1)
        out = self.fc1(concatenated)
        avg = torch.mean(torch.stack(ensemble_outputs, dim=0), dim=0)
        return out, avg


def get_data_loaders(batch_size, rank=None, world_size=1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler)

    return train_loader, test_loader

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        return test_loss, accuracy

def main(rank, world_size, use_ddp):
    if use_ddp:
        setup(rank, world_size)
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_data_loaders(batch_size, rank, world_size)

    model = FusionNet(num_ensembles).to(device)
    if use_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train(model, train_loader, criterion, optimizer, device)
    end_time = time.time()
    training_time = end_time - start_time
    if not use_ddp or rank == 0:
        print(f'Training time: {training_time} seconds')

    start_time = time.time()
    test_loss, accuracy = evaluate(model, test_loader, criterion, device)
    end_time = time.time()
    eval_time = end_time - start_time
    if not use_ddp or rank == 0:
        print(f'Evaluation time: {eval_time} seconds')
        print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

    if use_ddp:
        cleanup()


if __name__ == '__main__':
    use_ddp = gpus > 1
    if use_ddp:
        world_size = gpus
        mp.spawn(main,
                 args=(world_size, use_ddp),
                 nprocs=world_size,
                 join=True)
    else:
        main(0, 1, use_ddp)

