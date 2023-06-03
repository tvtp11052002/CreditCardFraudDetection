import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn import svm

import itertools

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import seaborn

data = pd.read_csv('creditcard.csv')  # Đọc file csv
df = pd.DataFrame(data)

df.describe()  # Mô tả các chỉ số thống kê (Sum, Average, Variance, minimum, 1st quartile, 2nd quartile, 3rd Quartile, Maximum)

df_fraud = df[df['Class'] == 1]  # fraud data trong dataset: các giao dịch bị gian lận
plt.figure(figsize=(15, 10))
plt.scatter(df_fraud['Time'], df_fraud['Amount'])  # Hiển thị số tiền gian lận theo thời gian
plt.title('Scratter plot amount fraud')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.xlim([0, 175000])
plt.ylim([0, 2500])
plt.show()

nb_big_fraud = df_fraud[df_fraud['Amount'] > 1000].shape[0]  # Recovery of frauds over 1000
print('There are only ' + str(nb_big_fraud) + ' frauds where the amount was bigger than 1000 over ' + str(
    df_fraud.shape[0]) + ' frauds')

number_fraud = len(data[data.Class == 1])
number_no_fraud = len(data[data.Class == 0])
print('There are only ' + str(number_fraud) + ' frauds in the original dataset, even though there are ' + str(
    number_no_fraud) + ' no frauds in the dataset.')

print("The accuracy of the classifier then would be : " + str(
    (284315 - 492) / 284315) + " which is the number of good classification over the number of tuple to classify")

df_corr = df.corr()  # Calculation of the correlation coefficients in pairs, with the default method:
# Pearson, Standard Correlation Coefficient

plt.figure(figsize=(15, 10))
seaborn.heatmap(df_corr, cmap="YlGnBu")
seaborn.set(font_scale=2, style='white')

plt.title('Heatmap correlation')
plt.show()

rank = df_corr['Class']  # Truy xuất các hệ số tương quan trên mỗi tính năng liên quan đến lớp tính năng
df_rank = pd.DataFrame(rank)
df_rank = np.abs(df_rank).sort_values(by='Class',
                                      ascending=False)  # Xếp giá trị tuyệt đối của các hệ số theo thứ tự giảm dần
df_rank.dropna(inplace=True)  # Loại bỏ các dữ liệu bị thiếu (các dữ liệu không phải số)

# Tách dữ liệu thành hai nhóm: tập train và tập test

# Tạo tập train
df_train_all = df[0:150000]  # Cắt thành 2 dữ liệu gốc
# Tách dữ liệu fraud và no fraud
df_train_1 = df_train_all[df_train_all['Class'] == 1]
df_train_0 = df_train_all[df_train_all['Class'] == 0]
print('In this dataset, we have ' + str(len(df_train_1)) + " frauds so we need to take a similar number of non-fraud")

df_sample = df_train_0.sample(300)
df_train = df_train_1.append(df_sample)
df_train = df_train.sample(frac=1)

X_train = df_train.drop(['Time', 'Class'], axis=1)  # Loại bỏ features Time (useless), Class (label)
y_train = df_train['Class']  # Tạo label mới
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

# xét tất cả test dataset xem mô hình có đang học đúng không
df_test_all = df[150000:]

X_test_all = df_test_all.drop(['Time', 'Class'],axis=1)
y_test_all = df_test_all['Class']
X_test_all = np.asarray(X_test_all)
y_test_all = np.asarray(y_test_all)

X_train_rank = df_train[df_rank.index[1:11]] # Lấy 10 features ranked đầu tiên
X_train_rank = np.asarray(X_train_rank)

X_test_all_rank = df_test_all[df_rank.index[1:11]]
X_test_all_rank = np.asarray(X_test_all_rank)
y_test_all = np.asarray(y_test_all)

class_names=np.array(['0','1']) # Binary label, Class = 1 (fraud) and Class = 0 (no fraud)


# Hàm để vẽ confusion matrix
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Khởi tạo model SVM với kernel tuyến tính
classifier = svm.SVC(kernel='linear')

# Huấn luyện mô hình
classifier.fit(X_train, y_train)

prediction_SVM_all = classifier.predict(X_test_all) #Dự đoán tập test

cm = confusion_matrix(y_test_all, prediction_SVM_all)
plot_confusion_matrix(cm,class_names)

print('Our criterion give a result of '
      + str( ( (cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1]/(cm[1][0]+cm[1][1])) / 5))
print('We have detected ' + str(cm[1][1]) + ' frauds / ' + str(cm[1][1]+cm[1][0]) + ' total frauds.')
print('\nSo, the probability to detect a fraud is ' + str(cm[1][1]/(cm[1][1]+cm[1][0])))
print("the accuracy is : "+str((cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))

classifier.fit(X_train_rank, y_train)
prediction_SVM = classifier.predict(X_test_all_rank)

cm = confusion_matrix(y_test_all, prediction_SVM)
plot_confusion_matrix(cm,class_names)

print('Our criterion give a result of '
      + str( ( (cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1]/(cm[1][0]+cm[1][1])) / 5))
print('We have detected ' + str(cm[1][1]) + ' frauds / ' + str(cm[1][1]+cm[1][0]) + ' total frauds.')
print('\nSo, the probability to detect a fraud is ' + str(cm[1][1]/(cm[1][1]+cm[1][0])))
print("the accuracy is : "+str((cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))

classifier_b = svm.SVC(kernel='linear',class_weight={0:0.60, 1:0.40})
classifier_b.fit(X_train, y_train)

prediction_SVM_b_all = classifier_b.predict(X_test_all)
cm = confusion_matrix(y_test_all, prediction_SVM_b_all)
plot_confusion_matrix(cm,class_names)

print('Our criterion give a result of '
      + str( ( (cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1]/(cm[1][0]+cm[1][1])) / 5))
print('We have detected ' + str(cm[1][1]) + ' frauds / ' + str(cm[1][1]+cm[1][0]) + ' total frauds.')
print('\nSo, the probability to detect a fraud is ' + str(cm[1][1]/(cm[1][1]+cm[1][0])))
print("the accuracy is : "+str((cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))
