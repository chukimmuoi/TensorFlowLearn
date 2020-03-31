# @title Load the imports

# from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers

from matplotlib import pyplot as plt

# Các dòng sau điều chỉnh mức độ chi tiết của báo cáo.
# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

tf.keras.backend.set_floatx('float32')

print("Imported the modules.")

# Tải tập dữ liệu
# Load the dataset
train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

# Thu nhỏ nhãn
# Scale the labels
scale_factor = 1000.0
# Thu nhỏ nhãn của tập huấn luyện.
# Scale the training set's label.
train_df["median_house_value"] /= scale_factor

# Thu nhỏ nhãn của bộ kiểm tra
# Scale the test set's label
test_df["median_house_value"] /= scale_factor

# Xáo trộn dữ liệu
# Shuffle the examples
train_df = train_df.reindex(np.random.permutation(train_df.index))

resolution_in_degrees = 1.0

# Tạo một danh sách trống mà cuối cùng sẽ giữ tất cả các cột tính năng.
# Create an empty list that will eventually hold all feature columns.
feature_columns = []

# Tạo một cột bucket feature để biểu thị vĩ độ.
# Create a bucket feature column for latitude.
latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
latitude_boundaries = list(np.arange(int(min(train_df['latitude'])),
                                     int(max(train_df['latitude'])),
                                     resolution_in_degrees))
latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column,
                                               latitude_boundaries)

# Tạo một cột bucket feature để biểu thị kinh độ.
# Create a bucket feature column for longitude.
longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
longitude_boundaries = list(np.arange(int(min(train_df['longitude'])),
                                      int(max(train_df['longitude'])),
                                      resolution_in_degrees))
longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column,
                                                longitude_boundaries)

# Tạo một feature cross của vĩ độ và kinh độ.
# Create a feature cross of latitude and longitude.
latitude_x_longitude = tf.feature_column.crossed_column([latitude, longitude], hash_bucket_size=100)
crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
feature_columns.append(crossed_feature)

# Chuyển đổi danh sách các cột tính năng thành một Layer mà cuối cùng sẽ trở thành một phần của mô hình.
# Convert the list of feature columns into a layer that will ultimately become part of the model.
# Understanding layers is not important right now.
feature_cross_feature_layer = layers.DenseFeatures(feature_columns)


# @title Define functions to create and train a model, and a plotting function
def create_model(my_learning_rate, feature_layer):
    """Tạo và biên dịch một mô hình hồi quy tuyến tính đơn giản."""
    """Create and compile a simple linear regression model."""
    # Hầu hết các mô hình tf.keras đơn giản là tuần tự.
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Thêm Layer chứa các cột tính năng cho mô hình.
    # Add the layer containing the feature columns to the model.
    model.add(feature_layer)

    # Thêm một Layer tuyến tính vào mô hình để mang lại hồi quy tuyến tính đơn giản.
    # Add one linear layer to the model to yield a simple linear regressor.
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    # Xây dựng các lớp thành một mô hình mà TensorFlow có thể thực thi.
    # Construct the layers into a model that TensorFlow can execute.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(model, dataset, epochs, batch_size, label_name):
    """Đưa dữ liệu vào mô hình để huấn luyện nó."""
    """Feed a dataset into the model in order to train it."""

    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=True)

    # Danh sách các Epochs được lưu trữ riêng biệt với phần còn lại của history.
    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Cô lập sai số trung bình tuyệt đối cho mỗi Epochs.
    # Isolate the mean absolute error for each epoch.
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    return epochs, rmse


def plot_the_loss_curve(epochs, rmse):
    """Vẽ một đường cong Loss so với Epochs."""
    """Plot a curve of loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min() * 0.94, rmse.max() * 1.05])
    plt.show()


print("Đã xác định các hàm created_model, train_model và plot_the_loss_curve.")
print("Defined the create_model, train_model, and plot_the_loss_curve functions.")

# Các biến sau đây là hyperparameters.
# The following variables are the hyperparameters.
learning_rate = 0.04
epochs = 35
batch_size = 100
label_name = 'median_house_value'

# Xây dựng mô hình, lần này đi qua trong buckets_feature_layer.
# Build the model, this time passing in the buckets_feature_layer.
my_model = create_model(learning_rate, feature_cross_feature_layer)

# Huấn luyện mô hình trên tập huấn luyện.
# Train the model on the training set.
epochs, rmse = train_model(my_model, train_df, epochs, batch_size, label_name)

plot_the_loss_curve(epochs, rmse)

print("\n: Đánh giá mô hình mới so với bộ thử nghiệm:")
print(": Evaluate the new model against the test set:")
test_features = {name: np.array(value) for name, value in test_df.items()}
test_label = np.array(test_features.pop(label_name))
my_model.evaluate(x=test_features, y=test_label, batch_size=batch_size)
