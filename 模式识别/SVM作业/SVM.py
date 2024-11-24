import paddle
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# 生成分类数据集，调整特征参数
X, y = make_classification(
    n_samples=500,     # 样本数量
    n_features=5,      # 总特征数量
    n_informative=2,   # 有效特征数量
    n_redundant=1,     # 冗余特征数量
    n_classes=2,       # 分类数
    random_state=42
)
print(X.shape, y.shape)  # 确认数据形状


# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#加载数据集

import paddle
from paddle.vision.transforms import Compose, Normalize

import numpy as np
from paddle.vision.datasets import MNIST
from paddle.vision.transforms import Compose, Normalize

# 数据预处理：将像素值归一化到 [-1, 1]
transform = Compose([Normalize(mean=[127.5], std=[127.5], data_format='CHW')])

# 加载 MNIST 数据集
train_dataset = MNIST(mode='train', transform=transform, download=True)
test_dataset = MNIST(mode='test', transform=transform, download=True)

# 提取训练集样本和标签
X_train = np.array([np.array(img).flatten() for img, label in train_dataset]).astype('float32') / 255.0
y_train = np.array([label for img, label in train_dataset])

# 提取测试集样本和标签
X_test = np.array([np.array(img).flatten() for img, label in test_dataset]).astype('float32') / 255.0
y_test = np.array([label for img, label in test_dataset])

# 打印形状
print(f"训练集样本形状: {X_train.shape}, 标签形状: {y_train.shape}")
print(f"测试集样本形状: {X_test.shape}, 标签形状: {y_test.shape}")



import numpy as np
import paddle.vision as vision
from paddle.vision.transforms import Compose, Normalize

# 数据预处理：归一化 CIFAR-10 数据集
transform = Compose([Normalize(mean=[0.5], std=[0.5], data_format='CHW')])

# 加载 CIFAR-10 数据集
train_dataset = vision.datasets.Cifar10(mode='train', transform=transform, download=True)
test_dataset = vision.datasets.Cifar10(mode='test', transform=transform, download=True)

# 提取训练集样本和标签
X_train = np.array([np.array(img).flatten() for img, label in train_dataset]).astype('float32') / 255.0
y_train = np.array([label for img, label in train_dataset])

# 提取测试集样本和标签
X_test = np.array([np.array(img).flatten() for img, label in test_dataset]).astype('float32') / 255.0
y_test = np.array([label for img, label in test_dataset])

# 打印形状
print(f"训练集样本形状: {X_train.shape}, 标签形状: {y_train.shape}")
print(f"测试集样本形状: {X_test.shape}, 标签形状: {y_test.shape}")



from sklearn.decomposition import PCA

# 降维到 50 维
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"降维后特征维数: {X_train_pca.shape[1]}")


# 定义自定义 SVM 模型
class SVM(paddle.nn.Layer):
    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.linear = paddle.nn.Linear(input_dim, 10)  # CIFAR-10/MNIST 为多分类任务

    def forward(self, x):
        return self.linear(x)

# 初始化模型
model = SVM(input_dim=X_train_pca.shape[1])
criterion = paddle.nn.CrossEntropyLoss()
optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())


# 模型训练与测试

# 转为飞桨张量
X_train_tensor = paddle.to_tensor(X_train_pca, dtype='float32')
y_train_tensor = paddle.to_tensor(y_train, dtype='int64')
X_test_tensor = paddle.to_tensor(X_test_pca, dtype='float32')
y_test_tensor = paddle.to_tensor(y_test, dtype='int64')

# 训练
for epoch in range(10):
    model.train()
    pred = model(X_train_tensor)
    loss = criterion(pred, y_train_tensor)
    
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()

    print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")


# 测试
model.eval()
pred_test = model(X_test_tensor)
pred_labels = pred_test.argmax(axis=1).numpy()

# 计算准确率
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, pred_labels)
print(f"Test Accuracy: {accuracy:.4f}")



