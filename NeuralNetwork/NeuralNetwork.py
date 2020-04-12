import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import seaborn as sns

# Các dòng sau đây điều chỉnh mức độ chi tiết của báo cáo.
# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

print("Imported modules.")

train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
train_df = train_df.reindex(np.random.permutation(train_df.index))  # shuffle the examples
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

print("Download data.")

# Tính Z-scores của từng cột trong tập huấn luyện:
# Calculate the Z-scores of each column in the training set:
train_df_mean = train_df.mean()
train_df_std = train_df.std()
train_df_norm = (train_df - train_df_mean) / train_df_std

# Tính Z-scores của từng cột trong bộ kiểm tra.
# Calculate the Z-scores of each column in the test set.
test_df_mean = test_df.mean()
test_df_std = test_df.std()
test_df_norm = (test_df - test_df_mean) / test_df_std

print("Normalized the values.")

# Tạo một danh sách trống mà cuối cùng sẽ giữ tất cả các cột tính năng được tạo.
# Create an empty list that will eventually hold all created feature columns.
feature_columns = []

# Chúng tôi đã thu nhỏ tất cả các cột, bao gồm cả vĩ độ và kinh độ, vào Z scores của chúng.
# Vì vậy, thay vì chọn độ phân giải theo độ, chúng tôi sẽ sử dụng resolution_in_Zs.
# resolution_in_Zs của 1 tương ứng với độ lệch chuẩn đầy đủ.
# We scaled all the columns, including latitude and longitude, into their
# Z scores. So, instead of picking a resolution in degrees, we're going
# to use resolution_in_Zs.  A resolution_in_Zs of 1 corresponds to
# a full standard deviation.
resolution_in_Zs = 0.3  # = 3/10 of a standard deviation.

# Tạo một cột tính năng bucket cho vĩ độ.
# Create a bucket feature column for latitude.
latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
latitude_boundaries = list(np.arange(int(min(train_df_norm['latitude'])),
                                     int(max(train_df_norm['latitude'])),
                                     resolution_in_Zs))
latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column,
                                               latitude_boundaries)

# Tạo một cột tính năng bucket cho kinh độ.
# Create a bucket feature column for longitude.
longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
longitude_boundaries = list(np.arange(int(min(train_df_norm['longitude'])),
                                      int(max(train_df_norm['longitude'])),
                                      resolution_in_Zs))
longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column,
                                                longitude_boundaries)

# Tạo một tính năng chéo của vĩ độ và kinh độ.
# Create a feature cross of latitude and longitude.
latitude_x_longitude = tf.feature_column.crossed_column([latitude, longitude], hash_bucket_size=100)
crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
feature_columns.append(crossed_feature)

# Đại diện cho median_income là một giá trị dấu phẩy động.
# Represent median_income as a floating-point value.
median_income = tf.feature_column.numeric_column("median_income")
feature_columns.append(median_income)

# Đại diện cho dân số như một giá trị dấu phẩy động.
# Represent population as a floating-point value.
population = tf.feature_column.numeric_column("population")
feature_columns.append(population)

# Chuyển đổi danh sách các cột tính năng thành một lớp mà sau này sẽ được đưa vào mô hình.
# Convert the list of feature columns into a layer that will later be fed into
# the model.
my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


def plot_the_loss_curve(epochs, mse):
    """Vẽ một đường cong loss so với epoch."""
    """Plot a curve of loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")

    plt.plot(epochs, mse, label="Loss")
    plt.legend()
    plt.ylim([mse.min() * 0.95, mse.max() * 1.03])
    plt.show()


print("Defined the plot_the_loss_curve function.")


def create_model(my_learning_rate, feature_layer):
    """Tạo và biên dịch mô hình hồi quy tuyến tính đơn giản"""
    """Create and compile a simple linear regression model."""
    # Hầu hết các mô hình tf.keras đơn giản là tuần tự.
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Thêm lớp chứa các cột tính năng cho mô hình.
    # Add the layer containing the feature columns to the model.
    model.add(feature_layer)

    # Thêm một lớp tuyến tính vào mô hình để mang lại hồi quy tuyến tính đơn giản.
    # Add one linear layer to the model to yield a simple linear regressor.
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    # Xây dựng các lớp thành một mô hình mà TensorFlow có thể thực thi.
    # Construct the layers into a model that TensorFlow can execute.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    return model


def train_model(model, dataset, epochs, batch_size, label_name):
    """Đưa dữ liệu vào mô hình để huấn luyện nó."""
    """Feed a dataset into the model in order to train it."""

    # Chia dữ liệu thành các tính năng và nhãn.
    # Split the dataset into features and label.
    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=True)

    # Nhận thông tin chi tiết sẽ hữu ích cho việc vẽ đường cong mất mát.
    # Get details that will be useful for plotting the loss curve.
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    rmse = hist["mean_squared_error"]

    return epochs, rmse


print("Defined the create_model and train_model functions.")

# Các biến sau đây là hyperparameters.
# The following variables are the hyperparameters.
learning_rate = 0.01
epochs = 15
batch_size = 1000
label_name = "median_house_value"

# Thiết lập địa hình của mô hình.
# Establish the model's topography.
my_model = create_model(learning_rate, my_feature_layer)

# Huấn luyện mô hình trên tập huấn luyện chuẩn hóa.
# Train the model on the normalized training set.
epochs, mse = train_model(my_model, train_df_norm, epochs, batch_size, label_name)
plot_the_loss_curve(epochs, mse)

test_features = {name: np.array(value) for name, value in test_df_norm.items()}
test_label = np.array(test_features.pop(label_name))  # isolate the label
print("\n Evaluate the linear regression model against the test set:")
my_model.evaluate(x=test_features, y=test_label, batch_size=batch_size)
