import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt


# hyperparameter ở đây đại diện cho:
#   + learning rate: Một số, được sử dụng để đào tạo mô hình.
#                    Learning rate sẽ được nhân với độ dốc sau mỗi lần lặp (gradient step).
#   + epochs       : Một lần đào tạo trên toàn bộ tập dữ liệu. Sao cho mỗi dữ liệu được thấy một lần.
#                    Do đó 1 epochs đại diện cho số lần lặp (iterations) để đào tạo dữ liệu kích thước là N / batch_size
#                    Cho đến khi mỗi dữ liệu được thấy một lần. Tối ưu nhấ là N / batch_size là một số nguyên.
#   + batch size   : Số dữ liệu sử dụng trong một batch.
#                    Vì quá trình đào tạo rất nhiều dữ liệu gây tốn bộ nhớ. Nên chia nhỏ để dễ đào tạo.

# @title Define the functions that build and train a model
def build_model(my_learning_rate):
    """Tạo và biên dịch mô hình hồi quy tuyến tính đơn giản."""
    """Create and compile a simple linear regression model."""
    # Hầu hết các mô hình tf.keras đơn giản là tuần tự.
    # Most simple tf.keras models are sequential.
    # Một mô hình tuần tự chứa một hoặc nhiều layers (lớp).
    # A sequential model contains one or more layers.
    model = tf.keras.models.Sequential()

    # Mô tả địa hình của mô hình.
    # Describe the topography of the model.
    # Địa hình của mô hình hồi quy tuyến tính đơn giản là một nút trong một lớp.
    # The topography of a simple linear regression model is a single node in a single layer.
    model.add(tf.keras.layers.Dense(units=1,
                                    input_shape=(1,)))

    # Biên dịch địa hình mô hình thành mã
    # Compile the model topography into code that
    # TensorFlow có thể thực thi hiệu quả. Cấu hình đào tạo để giảm mean squared error của mô hình.
    # TensorFlow can efficiently execute. Configure training to minimize the model's mean squared error.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(model, feature, label, epochs, batch_size):
    """Huấn luyện mô hình bằng dữ liệu."""
    """Train the model by feeding it data."""

    # Cung cấp các giá trị feature và các giá trị label cho mô hình.
    # Feed the feature values and the label values to the model.
    # Mô hình sẽ đào tạo theo số epochs được chỉ định,
    # The model will train for the specified number of epochs,
    # dần dần tìm hiểu làm thế nào các giá trị feature liên quan đến các giá trị label
    # gradually learning how the feature values relate to the label values.
    history = model.fit(x=feature,
                        y=label,
                        batch_size=batch_size,
                        epochs=epochs)

    # Tập hợp weight và bias của mô hình được đào tạo.
    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # Danh sách epoch được lưu trữ riêng biệt với phần còn lại của history.
    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Tập hợp history (ảnh chụp nhanh) của mỗi epoch.
    # Gather the history (a snapshot) of each epoch.
    hist = pd.DataFrame(history.history)

    # Thu thập root mean squared error của mô hình tại mỗi epoch.
    # Specifically gather the model's root mean squared error at each epoch.
    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse


print("Đã tạo function created_model và train_model")
print("Defined create_model and train_model")


# @title Define the plotting functions
def plot_the_model(trained_weight, trained_bias, feature, label):
    """Vẽ mô hình được đào tạo với feature và label."""
    """Plot the trained model against the training feature and label."""

    # Dán nhãn các trục.
    # Label the axes.
    plt.xlabel("feature")
    plt.ylabel("label")

    # Vẽ các giá trị tính năng so với các giá trị nhãn.
    # Plot the feature values vs. label values.
    plt.scatter(feature, label)

    # Tạo một đường màu đỏ đại diện cho mô hình.
    # Create a red line representing the model.
    # Đường màu đỏ bắt đầu tại tọa độ (x0, y0) và kết thúc tại tọa độ (x1, y1).
    # The red line starts at coordinates (x0, y0) and ends at coordinates (x1, y1).
    x0 = 0
    y0 = trained_bias
    x1 = my_feature[-1]
    y1 = trained_bias + (trained_weight * x1)
    plt.plot([x0, x1], [y0, y1], c='r')

    # Kết xuất biểu đồ phân tán và đường màu đỏ.
    # Render the scatter plot and the red line.
    # plt.show()


def plot_the_loss_curve(epochs, rmse):
    """Vẽ đường cong loss, cho thấy loss so với epoch."""
    """Plot the loss curve, which shows loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min() * 0.97, rmse.max()])
    plt.show()


print("Đã tạo function plot_the_model và plot_the_loss_curve.")
print("Defined the plot_the_model and plot_the_loss_curve functions.")

my_feature = ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
my_label = ([5.0, 8.8, 9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])

learning_rate = 0.1
epochs = 50
my_batch_size = 12

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature,
                                                         my_label, epochs,
                                                         my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)
