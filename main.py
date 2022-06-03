import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")

features_with_enumerable = ["MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities",
                            "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2",
                            "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st",
                            "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation",
                            "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
                            "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual",
                            "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
                            "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "SaleType",
                            "SaleCondition"]

for feature in features_with_enumerable:
    le = LabelEncoder()
    le.fit(df_train[feature])
    print(le.classes_)
    df_train[feature] = le.transform(df_train[feature])

dataset = df_train.values.tolist()
X_train = dataset[:][0:-1]

