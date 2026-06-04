import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timm
import wandb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

def main():
    wandb.login()
    run = wandb.init(
        # Set the wandb entity where you project wil be logged (generally your team name).
        entity="kitoueita1130-the-university-of-tokyo",
        project="resnet-cifar100",
        name="vit-timm (lr=5e-7)"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_classes = 100
    batch_size = 32
    learning_rate = 5e-7
    epochs = 30

    # 前処理
    img_size = 224

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # データセットとデータローダーの読み込み
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train images: {len(trainset)}, Test images: {len(testset)}")

    # timm から事前学習済み ViT モデルをロード
    print("Loading pretrained ViT model from timm...")

    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    model = model.to(device)

    # 損失関数と最適化関数の定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)

    # 学習・評価ループ
    print("Starting training on CIFAR-100...")
    best_acc = 0.0

    for epoch in range(epochs):
        # --- training phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_acc = 100.0 * correct / total

        # --- evaluating phase ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        val_loss = test_loss / len(testloader.dataset)
        val_acc = 100.0 * test_correct / test_total
        
        print(f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Val acc: {val_acc:.2f}%")
        run.log({'Loss': epoch_loss, 'Train Acc': epoch_acc, 'Test Acc': val_acc, 'Learning Rate': learning_rate})

        # record best acc
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), './output/best_vit-timm_cifar100_5e-7.pth')


print("Training Complete!")

if __name__ == '__main__':
    main()