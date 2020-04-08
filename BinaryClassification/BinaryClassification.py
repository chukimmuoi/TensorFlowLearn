# @title Load the imports

# from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# Các dòng sau đây điều chỉnh mức độ chi tiết của báo cáo.
# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
tf.keras.backend.set_floatx('float32')

print("Ran the import statements.")

train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
train_df = train_df.reindex(np.random.permutation(train_df.index))  # Xáo trộn tập huấn luyện.

# Tính Z-score của từng cột trong tập huấn luyện
# và ghi các z-score đó vào một DataFrame mới có tên train_df_norm.
# Calculate the Z-scores of each column in the training set and
# write those Z-scores into a new pandas DataFrame named train_df_norm.
train_df_mean = train_df.mean()
train_df_std = train_df.std()
train_df_norm = (train_df - train_df_mean) / train_df_std

# Kiểm tra một số giá trị của tập huấn luyện chuẩn hóa.
# Lưu ý rằng hầu hết các Z-score nằm trong khoảng từ -2 đến +2.
# Examine some of the values of the normalized training set. Notice that most
# Z-scores fall between -2 and +2.
train_df_norm.head()

# Tính Z-score của từng cột trong bộ kiểm tra
# và ghi các Z-score đó vào một DataFrame mới có tên test_df_norm.
# Calculate the Z-scores of each column in the test set and
# write those Z-scores into a new pandas DataFrame named test_df_norm.
test_df_mean = test_df.mean()
test_df_std = test_df.std()
test_df_norm = (test_df - test_df_mean) / test_df_std

# @title Double-click for possible solutions.

# > 265,000 --> 1
# <= 265,000 --> 0
# We arbitrarily set the threshold to 265,000, which is
# the 75th percentile for median house values.  Every neighborhood
# with a median house price above 265,000 will be labeled 1,
# and all other neighborhoods will be labeled 0.
threshold = 265000
train_df_norm["median_house_value_is_high"] = (train_df["median_house_value"] > threshold).astype(float)
test_df_norm["median_house_value_is_high"] = (test_df["median_house_value"] > threshold).astype(float)
train_df_norm["median_house_value_is_high"].head(8000)

# Alternatively, instead of picking the threshold
# based on raw house values, you can work with Z-scores.
# For example, the following possible solution uses a Z-score
# of +1.0 as the threshold, meaning that no more
# than 16% of the values in median_house_value_is_high
# will be labeled 1.

# threshold_in_Z = 1.0
# train_df_norm["median_house_value_is_high"] = (train_df_norm["median_house_value"] > threshold_in_Z).astype(float)
# test_df_norm["median_house_value_is_high"] = (test_df_norm["median_house_value"] > threshold_in_Z).astype(float)


# Tạo một danh sách trống mà cuối cùng sẽ giữ tất cả các cột tính năng được tạo.
# Create an empty list that will eventually hold all created feature columns.
feature_columns = []

# Tạo một cột tính năng số để biểu thị median_income.
# Create a numerical feature column to represent median_income.
median_income = tf.feature_column.numeric_column("median_income")
feature_columns.append(median_income)

# Tạo một cột tính năng số để thể hiện total_rooms.
# Create a numerical feature column to represent total_rooms.
tr = tf.feature_column.numeric_column("total_rooms")
feature_columns.append(tr)

# Chuyển đổi danh sách các cột tính năng thành một lớp mà sau này sẽ được đưa vào mô hình.
# Convert the list of feature columns into a layer that will later be fed into the model.
feature_layer = layers.DenseFeatures(feature_columns)

# In 3 hàng đầu tiên và 3 hàng cuối cùng của đầu ra feature_layer khi được áp dụng cho train_df_norm:
# Print the first 3 and last 3 rows of the feature_layer's output when applied to train_df_norm:
feature_layer(dict(train_df_norm))


# @title Xác định các chức năng tạo và huấn luyện một mô hình.
# Các bài tập trước sử dụng ReLU làm chức năng kích hoạt.
# Ngược lại, bài tập này sử dụng sigmoid làm chức năng kích hoạt.
# @title Define the functions that create and train a model.
def create_model(my_learning_rate, feature_layer, my_metrics):
    """Tạo và biên dịch một mô hình phân loại đơn giản."""
    """Create and compile a simple classification model."""
    # Hầu hết các mô hình tf.keras đơn giản là tuần tự.
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Thêm lớp tính năng (danh sách các tính năng và cách chúng được thể hiện) vào mô hình.
    # Add the feature layer (the list of features and how they are represented) to the model.
    model.add(feature_layer)

    # Kênh hồi quy giá trị thông qua một hàm sigmoid.
    # Funnel the regression value through a sigmoid function.
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,),
                                    activation=tf.sigmoid), )

    # Gọi phương thức biên dịch để xây dựng các layers thành một mô hình mà TensorFlow có thể thực thi.
    # Lưu ý rằng chúng ta đang sử dụng một loss function khác để phân loại hơn là hồi quy.
    # Call the compile method to construct the layers into a model that
    # TensorFlow can execute.  Notice that we're using a different loss
    # function for classification than for regression.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=my_metrics)

    return model


def train_model(model, dataset, epochs, label_name, batch_size=None, shuffle=True):
    """Đưa dữ liệu vào mô hình để huấn luyện nó."""
    """Feed a dataset into the model in order to train it."""

    # Tham số x của tf.keras.Model.fit có thể là danh sách các mảng,
    # trong đó mỗi mảng chứa dữ liệu cho một tính năng.
    # Ở đây, chúng tôi vượt qua mọi cột trong bộ dữ liệu.
    # Lưu ý rằng feature_layer sẽ lọc hầu hết các cột đó,
    # chỉ để lại các cột mong muốn và các biểu diễn của chúng làm các tính năng.
    # The x parameter of tf.keras.Model.fit can be a list of arrays, where
    # each array contains the data for one feature.  Here, we're passing
    # every column in the dataset. Note that the feature_layer will filter
    # away most of those columns, leaving only the desired columns and their
    # representations as features.
    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=shuffle)

    # Danh sách các epoch được lưu trữ riêng biệt với phần còn lại của history.
    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Cô lập số liệu phân loại cho mỗi epoch.
    # Isolate the classification metric for each epoch.
    hist = pd.DataFrame(history.history)

    return epochs, hist


print("Xác định các hàm created_model và train_model.")
print("Defined the create_model and train_model functions.")


# @title Xác định chức năng vẽ đồ thị.
# @title Define the plotting function.
def plot_curve(epochs, hist, list_of_metrics):
    """Vẽ đường cong của một hoặc nhiều số liệu phân loại so với epoch."""
    """Plot a curve of one or more classification metrics vs. epoch."""
    # list_of_metrics phải là một trong những tên được hiển thị trong:
    # list_of_metrics should be one of the names shown in:
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)

    plt.legend()
    plt.show()


print("Xác định hàm plot_curve.")
print("Defined the plot_curve function.")

# Các biến sau đây là siêu đường kính.
# The following variables are the hyperparameters.
learning_rate = 0.001
epochs = 20
batch_size = 100
label_name = "median_house_value_is_high"
classification_threshold = 0.35

# Dưới đây là định nghĩa cập nhật của METRICS:
# Here is the updated definition of METRICS:
METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy',
                                    threshold=classification_threshold),
    tf.keras.metrics.Precision(thresholds=classification_threshold,
                               name='precision'),
    tf.keras.metrics.Recall(thresholds=classification_threshold,
                            name="recall"),
]

# Thiết lập địa hình của mô hình.
# Establish the model's topography.
my_model = create_model(learning_rate, feature_layer, METRICS)

# Huấn luyện mô hình trên tập huấn luyện.
# Train the model on the training set.
epochs, hist = train_model(my_model, train_df_norm, epochs,
                           label_name, batch_size)

# Vẽ đồ thị của số liệu so với epoch.
# Plot a graph of the metric(s) vs. epochs.
list_of_metrics_to_plot = ['accuracy', 'precision', 'recall']

plot_curve(epochs, hist, list_of_metrics_to_plot)

features = {name: np.array(value) for name, value in test_df_norm.items()}
label = np.array(features.pop(label_name))

my_model.evaluate(x=features, y=label, batch_size=batch_size)
