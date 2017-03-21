#coding: utf-8
import pandas as pd #数据分析
import numpy as np #科学计算

from sklearn.ensemble import RandomForestRegressor
###使用 RandomForesetClassifier 填补年龄属性

def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges
    return df

# 船票有号码的填YES, 空的话写 NO
def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df

def set_data_dummies(df):
    # 因为逻辑回归建模时，需要输入的特征都是数值型特征
    # 我们先对类目型的特征离散/因子化
    # 以Cabin为例，原本一个属性维度，因为其取值可以是['yes','no']，而将其平展开为'Cabin_yes','Cabin_no'两个属性
    # 原本Cabin取值为yes的，在此处的'Cabin_yes'下取值为1，在'Cabin_no'下取值为0
    # 原本Cabin取值为no的，在此处的'Cabin_yes'下取值为0，在'Cabin_no'下取值为1
    # 我们使用pandas的get_dummies来完成这个工作，并拼接在原来的data_train之上，如下所示
    dummies_Cabin = pd.get_dummies(df['Cabin'], prefix='Cabin')

    dummies_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')

    dummies_Sex = pd.get_dummies(df['Sex'], prefix='Sex')

    dummies_Pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')

    df = pd.concat([df, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

    return df
# 接下来我们要接着做一些数据预处理的工作，比如scaling，将一些变化幅度较大的特征化到[-1,1]之内
# 这样可以加速logistic regression的收敛
def set_age_scaling(df):
    import sklearn.preprocessing as preprocessing
    scaler = preprocessing.StandardScaler()
    fare_scale_param = scaler.fit(df['Age'])
    df['Age_scaled'] = scaler.fit_transform(df['Age'], fare_scale_param)
    return df

def set_fare_sacling(df):
    import sklearn.preprocessing as preprocessing
    scaler = preprocessing.StandardScaler()
    fare_scale_param = scaler.fit(df['Fare'])
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
    return df

def get_data_feature(df):
    # 我们把需要的feature字段取出来，转成numpy格式，使用scikit-learn中的LogisticRegression建模
    from sklearn import linear_model

    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.as_matrix()

    # y即Survival结果
    y = train_np[:, 0]

    # X即特征属性值
    X = train_np[:, 1:]

    # fit到RandomForestRegressor之中
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)
    return clf

def handle_testdata():
    data_test = pd.read_csv("test.csv")
    data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
    data_test = set_missing_ages(data_test)
    data_test = set_Cabin_type(data_test)
    data_test = set_data_dummies(data_test)
    data_test = set_age_scaling(data_test)
    data_test = set_fare_sacling(data_test)
    return data_test

def cli():
    data_train = pd.read_csv("train.csv")
    data_train= set_missing_ages(data_train)
    data_train = set_Cabin_type(data_train)
    data_train = set_data_dummies(data_train)
    data_train = set_age_scaling(data_train)
    data_train = set_fare_sacling(data_train)
    clf = get_data_feature(data_train)

    data_test = handle_testdata()
    filtter_test = data_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = clf.predict(filtter_test)
    from pandas import DataFrame
    result = pd.DataFrame(
        {'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
    # 写到csv文件中
    result.to_csv("logistic_regression_predictions.csv", index=False)
if __name__ == '__main__':
    print 'start……'
    cli()
    print 'Do.'