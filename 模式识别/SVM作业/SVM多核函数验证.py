import numpy as np
import paddle
from paddle.vision.datasets import MNIST
from paddle.vision.transforms import Compose, Normalize
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

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

# 打印形状检查
print(f"训练集样本形状: {X_train.shape}, 标签形状: {y_train.shape}")
print(f"测试集样本形状: {X_test.shape}, 标签形状: {y_test.shape}")

# 选择子集大小
subset_size = 10000  # 用于训练的子集大小
test_subset_size = 2000  # 用于测试的子集大小

# 子集化数据
X_train_subset = X_train[:subset_size]
y_train_subset = y_train[:subset_size]
X_test_subset = X_test[:test_subset_size]
y_test_subset = y_test[:test_subset_size]

# 应用 PCA 降维
pca = PCA(n_components=100)  # 将特征降到 100 维
X_train_pca = pca.fit_transform(X_train_subset)
X_test_pca = pca.transform(X_test_subset)

# 打印降维后数据形状
print(f"降维后训练集样本形状: {X_train_pca.shape}, 标签形状: {y_train_subset.shape}")
print(f"降维后测试集样本形状: {X_test_pca.shape}, 标签形状: {y_test_subset.shape}")

# 定义 SVM 模型
svm_linear = SVC(kernel='linear', C=1.0)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_poly = SVC(kernel='poly', degree=3, C=1.0)

# 训练并测试线性核
print("Training Linear SVM...")
svm_linear.fit(X_train_pca, y_train_subset)
accuracy_linear = svm_linear.score(X_test_pca, y_test_subset)

# 训练并测试 RBF 核
print("Training RBF SVM...")
svm_rbf.fit(X_train_pca, y_train_subset)
accuracy_rbf = svm_rbf.score(X_test_pca, y_test_subset)

# 训练并测试多项式核
print("Training Polynomial SVM...")
svm_poly.fit(X_train_pca, y_train_subset)
accuracy_poly = svm_poly.score(X_test_pca, y_test_subset)

# 输出准确率
print(f"Linear Kernel Accuracy: {accuracy_linear:.4f}")
print(f"RBF Kernel Accuracy: {accuracy_rbf:.4f}")
print(f"Polynomial Kernel Accuracy: {accuracy_poly:.4f}")

# 分类报告
print("Linear Kernel Classification Report:")
print(classification_report(y_test_subset, svm_linear.predict(X_test_pca)))

print("RBF Kernel Classification Report:")
print(classification_report(y_test_subset, svm_rbf.predict(X_test_pca)))

print("Polynomial Kernel Classification Report:")
print(classification_report(y_test_subset, svm_poly.predict(X_test_pca)))

# 准确率对比
kernels = ['Linear', 'RBF', 'Polynomial']
accuracies = [accuracy_linear, accuracy_rbf, accuracy_poly]

plt.bar(kernels, accuracies, color=['blue', 'green', 'red'])
plt.xlabel("Kernel Function")
plt.ylabel("Accuracy")
plt.title("SVM Kernel Comparison on MNIST Dataset")
plt.show()
