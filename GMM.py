import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 尝试读取数据文件
try:
    data = pd.read_csv(r'D:\适宜性评价\csv_300\label.csv', index_col='FID')
except FileNotFoundError:
    print("无法找到文件，请检查文件路径是否正确。")
    exit()

print(data.info())

# 对数据进行预处理
# 使用均值填充缺失值
imp = SimpleImputer(strategy='mean')
# 数据标准化
scaler = StandardScaler()
preprocessor = Pipeline([('imputer', imp), ('scaler', scaler)])

# 将预处理器应用于所有数据
data.loc[:, 'A2':'A12'] = preprocessor.fit_transform(data.loc[:, 'A2':'A12'])

# 对未标记的数据进行预测
Ytest = data.loc[:, 'label'][data.loc[:, 'label'].isnull()]
Xtest = data.loc[Ytest.index, 'A2':'A12']
Xtest = preprocessor.transform(Xtest)  # 使用之前相同的预处理

# 使用GMM进行聚类
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(Xtest)

# 获取每个类别的预测概率
Ytest_proba = gmm.predict_proba(Xtest)

# 找出概率最大的类别
Ytest_pred = Ytest_proba.argmax(axis=1)

# 将预测的类别填回到原数据中
data.loc[data.loc[:, 'label'].isnull(), 'label'] = Ytest_pred

print(data.info())

data.to_csv(r'D:\适宜性评价\csv_300\label_predict_GMM.csv')