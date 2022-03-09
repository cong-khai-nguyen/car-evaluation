import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

#Read data in
data = pd.read_csv("car.data")
print(data.head())

# Convert data to numerical value to be able to do computational calculating.
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
door = le.fit_transform(list(data["door"]))
maint = le.fit_transform(list(data["maint"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = cls

x = list(zip(buying, door, maint, persons, lug_boot, safety))
y = list(predict)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors = 9)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)


predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]


for x in range(len(predicted)):
    print("Predicted:", names[predicted[x]], "Data:", x_test[x], "Actual:", names[y_test[x]])

# Accuracy : 95.375723%
print("Accuracy :", format(acc, "%"))

