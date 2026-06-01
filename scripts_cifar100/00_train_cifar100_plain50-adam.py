import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from tqdm import tqdm
import sys
sys.path.append('./src/my_project')
from model_plain import Plainblock, get_Plain50


def main():
    wandb.login()
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="kitoueita1130-the-university-of-tokyo",
        # Set the wandb project where this run will be logged.
        project="resnet-cifar100",
        # Track hyperparameters and run metadata.
        name="plain-50 (AdamW)"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 設定パラメータ
    batch_size = 128
    max_epochs = 300
    initial_lr = 1e-3

    # 1. データセットの準備
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # ランダムにずらして切り抜き
        transforms.RandomHorizontalFlip(), # ランダムに左右反転
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 2. モデル、損失関数、最適化関数の定義
    net = get_Plain50(Plainblock, num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(net.parameters(), lr=initial_lr, weight_decay=1e-2)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # 3. トレーニングループ
    print('--- 本番トレーニング開始 ---')
    best_acc = 0.0

    for epoch in range(1, max_epochs + 1):
        # --- training phase ---
        net.train()
        running_loss = 0.0
        train_corrects = 0
        train_total = 0

        for i, (inputs, labels) in tqdm(enumerate(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # train acc
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            train_corrects += torch.sum(preds == labels.data).item()
            train_total += labels.size(0)
        
        # エポック終了時点での train acc
        epoch_loss = running_loss / len(trainloader)
        epoch_train_acc = (train_corrects / train_total) * 100

        # --- test phase ---
        net.eval()
        test_corrects = 0
        test_total = 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, preds = torch.max(outputs, 1)
                test_corrects += torch.sum(preds == labels.data).item()
                test_total += labels.size(0)
        
        epoch_test_acc = (test_corrects / test_total) * 100

        # scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # results by epoch
        print(f'Epoch [{epoch}/{max_epochs}] '
              f'Loss: {epoch_loss:.4f} | '
              f'Train Acc: {epoch_train_acc:.2f}% | '
              f'Test Acc: {epoch_test_acc:.2f}% | '
              f'Learning Rate: {current_lr:.6f}')
        run.log({'Loss': epoch_loss, 'Train Acc': epoch_train_acc, 'Test Acc': epoch_test_acc, 'Learning Rate': current_lr})
        
        # record best acc
        if epoch_test_acc > best_acc:
            best_acc = epoch_test_acc
            torch.save(net.state_dict(), './output/best_plain50_cifar100.pth')

    print(f'Finished Training. Best Test Accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main()