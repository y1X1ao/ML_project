import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 尝试读取数据文件
try:
    data = pd.read_csv(r'D:\适宜性评价\csv_300\row_label.csv', index_col='FID')
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

# 使用KMeans进行无监督分类
kmeans = KMeans(n_clusters=3, random_state=0)
data['label'] = kmeans.fit_predict(data.loc[:, 'A2':'A12'])

print(data.info())
data.to_csv(r'D:\适宜性评价\csv_300\row_label_predict_Kmeans.csv')