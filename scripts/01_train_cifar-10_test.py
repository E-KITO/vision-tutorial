import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from tqdm import tqdm
import sys
sys.path.append('./src/my_project')
from model import Resblock, get_ResNet50


def main():
    wandb.login()
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="kitoueita1130-the-university-of-tokyo",
        # Set the wandb project where this run will be logged.
        project="resnet-test",
        # Track hyperparameters and run metadata.
        name="test"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 1. データセットの準備
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # CIFAR-10 の RGB の mean, std
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    # 2. モデル、損失関数、最適化関数の定義
    net = get_ResNet50(Resblock, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)

    # 3. トレーニングループ
    net.train()
    for epoch in range(1, 6): 
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for i, (inputs, labels) in tqdm(enumerate(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # backward
            loss.backward()

            # パラメータの更新
            optimizer.step()

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += labels.size(0)
            if i % 100 == 99:
                current_loss = running_loss / 100
                current_acc = (running_corrects / total_samples) * 100

                print(f'[Epoch {epoch}, Batch {i+1}] loss: {current_loss:.3f} | acc: {current_acc:.2f}%')
                run.log({'acc': current_acc, 'loss': current_loss})
            
                running_loss = 0.0

    print('Finished Training')

if __name__ == '__main__':
    main()