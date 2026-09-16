import os
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
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2, bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2,  bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2,  bias=False)
        self.layers = nn.ModuleList([self.conv1, self.conv2, self.conv3])

        # 二値化の重み
        self.b_conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2, bias=False)
        self.b_conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2,  bias=False)
        self.b_conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2,  bias=False)
        self.b_layers = nn.ModuleList([self.b_conv1, self.b_conv2, self.b_conv3])

        # 神の一手
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        # padding=2だから32*32→16*16→8*8
        self.fc = nn.Linear(64 * 8 * 8, 10)

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

        x = x.view(x.size(0), -1)
        x = self.fc(x)

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


def evaluate(model, x, y, device):
    """
    正答率の計算
    
    """
    model.eval()

    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=1)

        correct = (pred == y).sum().item()
        total = y.size(0)

    accuracy = 100.0 * correct / total

    return accuracy


def cross_entropy_loss(model, x, y):
    output = model(x)
    loss =  nn.functional.cross_entropy(output, y)

    return loss


def plot(train_losses, test_accuracies):
    """
    グラフ描画

    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Lossのグラフ
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'o-', color='tab:red', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Accuracyのグラフ
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, 'o-', color='tab:blue', label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    os.makedirs('./output', exist_ok=True) # フォルダがなければ作成
    plt.savefig('./output/binaryconnect_continue50_seed42_result.png')
    print("\nグラフを 'binaryconnect_continue50_seed42_result.png' に保存しました。")


def main():

    random_seed = 43
    epochs = 50
    batch_size = 64
    learning_rate = 0.0001
    torch.manual_seed(random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # デバイス判定
    print(f"使用デバイス: {device}")

    # 事前学習の重みを読み込む
    load_path = './output/binaryconnect_cifar_aug_3conv_bp_seed42_1500ep.pt'
    checkpoint = torch.load(load_path, map_location='cpu', weights_only=True)
    model = BinaryConnectCifar10().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

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

    # --- 訓練用とGA検証用に分割 ---
    # 訓練: 45000枚、GA検証用: 5000枚(学習には一切使わない)
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # 事前学習で分けておいた画像設定をそのまま呼び出す
    train_dataset = Subset(full_train_dataset, checkpoint['train_indices'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # テストデータをTensorにまとめる
    test_x = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    test_y = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    optimizer = optim.Adam(
        list(model.layers.parameters()) +
        list(model.bn1.parameters()) +
        list(model.bn2.parameters()) +
        list(model.bn3.parameters()) +
        list(model.fc.parameters()),
        lr=learning_rate
    )

    # 学習率をcos関数の波形に沿って徐々に小さくしていく
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    test_accuracies = []

    # 学習前のスコア
    before_acc = evaluate(model, test_x, test_y, device)
    print(f"追加BP前のTest Accuracy: {before_acc:.2f}%")

    print("学習開始")

    for epoch in range(1, epochs + 1):
        # モデルを学習モードに
        model.train()
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device) # データを GPU へ転送
            loss = cross_entropy_loss(model, data, target)
            model.update(optimizer, loss)
            # そのエポック全体のloss
            running_loss += loss.item() * data.size(0)
        # 平均loss
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = evaluate(model, test_x, test_y, device)

        scheduler.step()

        train_losses.append(epoch_loss)
        test_accuracies.append(epoch_acc)

        print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.4f} | Test Acc: {epoch_acc:.2f}%")

    after_acc = test_accuracies[-1]
    print(f"Test Accuracy: {before_acc:.2f}% → {after_acc:.2f}%")
    print(f"変化: {after_acc - before_acc:+.2f}ポイント")

    # グラフ描画実行
    plot(train_losses, test_accuracies)

if __name__ == '__main__':
    main()






