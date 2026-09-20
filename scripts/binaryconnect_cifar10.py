import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

class BinaryConnectCifar10(nn.Module):
    def __init__(self):
        super().__init__()

        # 実数の重み
        # RGBで3チャネルあるから3（Mnistは白黒だから1）
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1,  bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1,  bias=False)
        self.fc = nn.Linear(64 * 4 * 4, 10)
        self.layers = nn.ModuleList([self.conv1, self.conv2, self.conv3, self.fc])

        # 二値化の重み
        self.b_conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.b_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1,  bias=False)
        self.b_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1,  bias=False)
        self.b_fc = nn.Linear(64 * 4 * 4, 10)
        self.b_layers = nn.ModuleList([self.b_conv1, self.b_conv2, self.b_conv3, self.b_fc])

        # 神の一手
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()


    def binarize(self):
        """
        実数の重みを二値化

        """

        for layers, b_layers in zip(self.layers, self.b_layers):
            # torch.signでもいいけど、0 のときに 0 を返してしまい，重みが +1 でも -1 でもなくなってしまう_
            b_layers.weight.data = torch.where(layers.weight.data >= 0, 1.0, -1.0)

            # 実数層 (layers) のバイアスを 二値化層 (b_layer) にそのまま複製
            if layers.bias is not None and b_layers.bias is not None:
                b_layers.bias.data = layers.bias.data.clone()


    def forward(self, x):
        """
        順伝播：二値化された重みで計算
        予測値をだす？誤差から逆算する前に一旦今の結果を見る感じ

        """

        self.binarize()

        x = self.b_conv1(x)
        x = self.relu(self.bn1(x))
        x = self.pool(x)

        x = self.b_conv2(x)
        x = self.relu(self.bn2(x))
        x = self.pool(x)

        x = self.b_conv3(x)
        x = self.relu(self.bn3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.b_fc(x)

        return x

    def set_grad(self):
        """
        勾配を実数層にコピー

        """
        for layers, b_layers in zip(self.layers, self.b_layers):
            if b_layers.weight.grad is not None:
                layers.weight.grad = b_layers.weight.grad.clone()
            if b_layers.bias is not None and b_layers.bias.grad is not None:
                layers.bias.grad = b_layers.bias.grad.clone()

    def clipping(self):
        """
        実数の重みが大きくなりすぎて，二値化重み（+1,-1のみなので）が変化しにくくなるのを防ぐ
        為に実数重みを-1.0 ~ 1.0に制限

        """
        for layers in self.layers:
            layers.weight.data.clamp_(-1.0, 1.0)


    def update(self, optimizer, loss):
        """
        一回分の学習更新を一括処理する関数

        """
        optimizer.zero_grad()
        for b_layer in self.b_layers:
            if b_layer.weight.grad is not None:
                b_layer.weight.grad.zero_()
            if b_layer.bias is not None and b_layer.bias.grad is not None:
                b_layer.bias.grad.zero_()

        loss.backward()
        self.set_grad()
        optimizer.step()
        self.clipping()


def test_evaluate(model, x, y, device):
    """テストデータに対するlossとaccuracyの計算"""
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        output = model(x)
        loss = nn.functional.cross_entropy(output, y).item()
        acc = (output.argmax(dim=1) == y).float().mean().item() * 100.0

    return acc, loss


def train_evaluate(model, x, y):
    """
    """
    output = model(x)
    loss = nn.functional.cross_entropy(output, y)
    with torch.no_grad():
        acc = (output.argmax(dim=1) == y).float().mean().item() * 100.0

    return acc, loss


def plot(train_losses, train_accuracies, test_losses, test_accuracies):
    """
    Train/TestのLossとAccuracyをグラフ化する。
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(
        epochs,
        train_losses,
        color='tab:red',
        label='Train Loss'
    )
    plt.plot(
        epochs,
        test_losses,
        color='tab:orange',
        label='Test Loss'
    )
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(
        epochs,
        train_accuracies,
        color='tab:green',
        label='Train Accuracy'
    )
    plt.plot(
        epochs,
        test_accuracies,
        color='tab:blue',
        label='Test Accuracy'
    )
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    os.makedirs('./output', exist_ok=True)

    graph_path = ('./output/binaryconnect_cifar_3conv_seed43_result_500.png')
    plt.savefig(graph_path, dpi=300)
    plt.close()

    print(f"\nグラフを '{graph_path}' に保存しました。")


def save_csv(train_losses, train_accuracies, test_losses, test_accuracies):
    """
    エポックごとの評価結果をCSVに保存する。
    """
    os.makedirs('./output', exist_ok=True)

    csv_path = (
        './output/'
        'binaryconnect_cifar_3conv_seed_43_result_500.csv'
    )

    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        writer.writerow([
            'epoch',
            'train_loss',
            'train_accuracy',
            'test_loss',
            'test_accuracy'
        ])

        for epoch, values in enumerate(
            zip(
                train_losses,
                train_accuracies,
                test_losses,
                test_accuracies
            ),
            start=1
        ):
            train_loss, train_acc, test_loss, test_acc = values

            writer.writerow([
                epoch,
                train_loss,
                train_acc,
                test_loss,
                test_acc
            ])


def main():

    random_seed = 43
    epochs = 500
    batch_size = 64
    learning_rate = 0.001
    torch.manual_seed(random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # デバイス判定
    print(f"使用デバイス: {device}")
    model = BinaryConnectCifar10().to(device) # モデルを GPU へ転送

    # データセットの準備（正規化も）
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
    ])
        # テスト用：ランダムな切り抜き・反転はしない
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # テストデータをTensorにまとめる
    test_x = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).to(device)
    test_y = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))]).to(device)

    optimizer = optim.Adam(
        list(model.layers.parameters()) +
        list(model.bn1.parameters()) +
        list(model.bn2.parameters()) +
        list(model.bn3.parameters()),
        lr=learning_rate
    )

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    print("学習開始")

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_acc = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device) # データを GPU へ転送
            # モデルを学習モードに
            model.train()
            acc, loss = train_evaluate(model, data, target)
            model.update(optimizer, loss)
            running_loss += loss.item() * data.size(0)
            running_acc += acc * data.size(0)

        # 学習データの評価
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_acc / len(train_loader.dataset)

        # テストデータの評価
        model.eval()
        test_acc, test_loss = test_evaluate(model, test_x, test_y, device)
       

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)

        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    plot(train_losses, train_accuracies, test_losses, test_accuracies)
    save_csv(train_losses, train_accuracies, test_losses, test_accuracies)

   # 学習済みパラメータとデータ分割を保存
   # main()内
    save_path = './output/binaryconnect_cifar_3conv_seed43_500.pt'

    torch.save({
        'model_state_dict': model.state_dict(),
        'epochs': epochs,
        'seed': 42,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
    }, save_path)

    print(f"学習済みパラメータを '{save_path}' に保存しました。")

if __name__ == '__main__':
    main()






