from builtins import int

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")
df_test_tags = pd.read_csv("./data/sample_submission.csv")

df_test["SalePrice"] = df_test_tags["SalePrice"].values.tolist()

df = pd.concat([df_train,df_test], ignore_index=True)

df.fillna("NA", inplace=True)
print(df["MSZoning"])

categorical_nominal_features = ["MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig",
                                "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle",
                                "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
                                "MasVnrType", "Foundation", "Heating", "CentralAir", "Electrical",
                                "Functional", "GarageType", "PavedDrive", "MiscFeature", "SaleType",
                                "SaleCondition"]

categorical_ordinal_features = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual",
    "GarageCond", "PoolQC", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "GarageFinish", "Fence"
]

""" forse Utilities è ordinato """
quality_scale_dict = {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
BstmExposure_dict = {"NA": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
BsmtFinType_dict = {"NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
GarageFinish_dict = {"NA": 0, "Unf": 1, "RFn": 2, "Fin": 3}
Fence_dict = {"NA": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}

# contiene coppie del tipo feature:scala usata
feature_scale_dict = {
    "ExterQual": quality_scale_dict,
    "ExterCond": quality_scale_dict,
    "BsmtQual": quality_scale_dict,
    "BsmtCond": quality_scale_dict,
    "HeatingQC": quality_scale_dict,
    "KitchenQual": quality_scale_dict,
    "FireplaceQu": quality_scale_dict,
    "GarageQual": quality_scale_dict,
    "GarageCond": quality_scale_dict,
    "PoolQC": quality_scale_dict,
    "BsmtExposure": BstmExposure_dict,
    "BsmtFinType1": BsmtFinType_dict,
    "BsmtFinType2": BsmtFinType_dict,
    "GarageFinish": GarageFinish_dict,
    "Fence": Fence_dict
}
print(df[categorical_ordinal_features].loc[0].values.tolist())
for feature in categorical_nominal_features:
    # change enum values to integer
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])

for feature in categorical_ordinal_features:
    dict = feature_scale_dict[feature]

    df[feature] = df[feature].apply(lambda x: dict[x])

print(df[categorical_ordinal_features].loc[0].values.tolist())

df.replace("NA", 0, inplace=True)

# splitting data from tags
X_train = df.drop(["SalePrice"], axis=1)
Y_train = df["SalePrice"].values

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.10)


linear_regression = LinearRegression()
linear_regression.fit(X_train, Y_train)

print(linear_regression.score(X_test, Y_test))

# tree_regression = DecisionTreeRegressor(max_depth=5)
# tree_regression.fit(X_train, Y_train)
# pred = tree_regression.predict(X_test, Y_test)
# print(tree_regression.score(X_test, Y_test))
# print(mean_squared_error(Y_test, pred))

# regr = RandomForestRegressor(n_estimators=2, random_state=0)
# regr.fit(X_train, Y_train)
# print(regr.score(X_train, Y_train))
# print(regr.score(X_test, Y_test))

