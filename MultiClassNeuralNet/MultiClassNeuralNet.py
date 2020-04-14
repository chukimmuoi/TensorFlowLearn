import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# Các dòng sau đây điều chỉnh mức độ chi tiết của báo cáo.
# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# Dòng sau cải thiện định dạng khi nhập mảng NumPy.
# The following line improves formatting when ouputting NumPy arrays.
np.set_printoptions(linewidth=200)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Ví dụ đầu ra #2917 của tập huấn luyện.
# Output example #2917 of the training set.
x_train[2917]

# Sử dụng màu để trực quan hóa mảng.
# Use false colors to visualize the array.
plt.imshow(x_train[2917])

# Hàng đầu ra #10 của ví dụ #2917.
# Output row #10 of example #2917.
x_train[2917][10]

# Pixel đầu ra #16 của hàng #10 của ví dụ #2900.
# Output pixel #16 of row #10 of example #2900.
x_train[2917][10][16]

# Task 01
x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0
print(x_train_normalized[2900][12])  # Output a normalized row


# Xác định hàm vẽ
# Define a plotting function
def plot_curve(epochs, hist, list_of_metrics):
    """Vẽ đường cong của một hoặc nhiều số liệu phân loại so với epoch."""
    """Plot a curve of one or more classification metrics vs. epoch."""
    # list_of_metrics phải là một trong những tên được hiển thị trong
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


print("Loaded the plot_curve function.")


def create_model(my_learning_rate):
    """Tạo và biên dịch một mạng lưới thần kinh."""
    """Create and compile a deep neural net."""

    # Tất cả các mô hình trong khóa học này là tuần tự.
    # All models in this course are sequential.
    model = tf.keras.models.Sequential()

    # Các tính năng được lưu trữ trong một mảng 28X28 hai chiều.
    # Làm phẳng mảng hai chiều đó thành mảng 784 một phần tử.
    # The features are stored in a two-dimensional 28X28 array.
    # Flatten that two-dimensional array into a a one-dimensional
    # 784-element array.
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    # Xác định lớp ẩn đầu tiên.
    # Define the first hidden layer.
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))

    # Xác định một lớp dropout regularization.
    # Define a dropout regularization layer.
    model.add(tf.keras.layers.Dropout(rate=0.4))

    # Xác định lớp đầu ra.
    # Tham số đơn vị được đặt thành 10 vì mô hình phải chọn trong số 10 giá trị đầu ra có thể
    # (đại diện cho các chữ số từ 0 đến 9, đã bao gồm).
    # Define the output layer. The units parameter is set to 10 because
    # the model must choose among 10 possible output values (representing
    # the digits from 0 to 9, inclusive).
    #
    # Đừng thay đổi lớp này.
    # Don't change this layer.
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    # Xây dựng các lớp thành một mô hình mà TensorFlow có thể thực thi.
    # Lưu ý rằng hàm Loss cho phân loại nhiều lớp khác với hàm Loss cho phân loại nhị phân.
    # Construct the layers into a model that TensorFlow can execute.
    # Notice that the loss function for multi-class classification
    # is different than the loss function for binary classification.
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    return model


def train_model(model, train_features, train_label, epochs,
                batch_size=None, validation_split=0.1):
    """Huấn luyện mô hình bằng cách cho nó ăn dữ liệu."""
    """Train the model by feeding it data."""

    history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                        epochs=epochs, shuffle=True,
                        validation_split=validation_split)

    # Để theo dõi tiến trình đào tạo, thu thập ảnh chụp nhanh các số liệu của mô hình ở mỗi kỷ nguyên.
    # To track the progression of training, gather a snapshot
    # of the model's metrics at each epoch.
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist


# Các biến sau đây là siêu đường kính.
# The following variables are the hyperparameters.
learning_rate = 0.003
epochs = 50
batch_size = 4000
validation_split = 0.2

# Thiết lập địa hình của mô hình.
# Establish the model's topography.
my_model = create_model(learning_rate)

# Huấn luyện mô hình trên tập huấn luyện chuẩn hóa.
# Train the model on the normalized training set.
epochs, hist = train_model(my_model, x_train_normalized, y_train,
                           epochs, batch_size, validation_split)

# Vẽ đồ thị của số liệu so với epochs.
# Plot a graph of the metric vs. epochs.
list_of_metrics_to_plot = ['accuracy']
plot_curve(epochs, hist, list_of_metrics_to_plot)

# Đánh giá theo bộ thử nghiệm.
# Evaluate against the test set.
print("\n Evaluate the new model against the test set:")
my_model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)
