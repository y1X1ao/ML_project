import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score
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

# 划分数据
Y = data.loc[:, 'label'][data.loc[:, 'label'].notnull()]
X = data.loc[Y.index, 'A2':'A12']
print(X.head())

# 划分训练集和验证集
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, random_state=10)
from imblearn.over_sampling import SMOTE
# 平衡训练集数据
sm = SMOTE(random_state=42)
X_train,Y_train = sm.fit_resample(X_train,Y_train)


# 建立模型并设置超参数网格
rfc = RandomForestClassifier(random_state=42
                             # ,class_weight='balanced'
                             )
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
}

# 使用网格搜索进行超参数调整
grid_search = GridSearchCV(rfc, param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X_train, Y_train)
# 打印最佳参数组合
print("Best parameters: {}".format(grid_search.best_params_))

# 验证模型并打印准确度
Y_val_pred = grid_search.predict(X_val)
accuracy = accuracy_score(Y_val, Y_val_pred)
f1 = f1_score(Y_val, Y_val_pred, average='macro')
auc = roc_auc_score(Y_val, Y_val_pred)
print("Validation accuracy: {}, F1 score: {}, AUC: {}".format(accuracy, f1, auc))

# 对未标记的数据进行预测
Ytest = data.loc[:, 'label'][data.loc[:, 'label'].isnull()]
Xtest = data.loc[Ytest.index, 'A2':'A12']
Xtest = preprocessor.transform(Xtest)  # 使用之前相同的预处理

# 获取每个类别的预测概率
Ytest_proba = grid_search.predict_proba(Xtest)

# 找出概率为1的概率值，假设1是正类别
# 注意这里要检查一下grid_search.classes_，确认正类别是否为1
pos_class_index = list(grid_search.classes_).index(1)
Ytest_pos_proba = Ytest_proba[:, pos_class_index]

# 将预测的概率填回到原数据中
data.loc[data.loc[:, 'label'].isnull(), 'label'] = Ytest_pos_proba
print(data.info())
data.to_csv(r'D:\适宜性评价\csv_300\label_predict_rf_new.csv')
