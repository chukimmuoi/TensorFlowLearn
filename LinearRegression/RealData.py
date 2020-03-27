# @title Import relevant modules
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

# The following lines adjust the granularity of reporting.
pd.options.display.max_columns = None
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# Nhập dữ liệu
# Import the dataset.
training_df = pd.read_csv(
    filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")

# Thu nhỏ Lable bằng cách chia giá trị cho 1000
# Việc này làm giá trị của Loss và Learning rate nhỏ, tối ưu trong tính toán.
# Việc này là cần thiết khi sử dụng dữ liệu có nhiều Feature.
# Scale the label.
training_df["median_house_value"] /= 1000.0

# In các hàng đầu tiên từ dữ liệu.
# Print the first rows of the pandas DataFrame.
print(training_df.head())

# Cung cấp thông tin, số liệu thống kê về toàn bộ dữ liệu.
# Get statistics on the dataset.
print(training_df.describe())

# Cung cấp độ tương quan giữa các cột.
print(training_df.corr())


# @title Double-click to view a possible answer.

# The maximum value (max) of several columns seems very
# high compared to the other quantiles. For example,
# example the total_rooms column. Given the quantile
# values (25%, 50%, and 75%), you might expect the
# max value of total_rooms to be approximately
# 5,000 or possibly 10,000. However, the max value
# is actually 37,937.

# When you see anomalies in a column, become more careful
# about using that column as a feature. That said,
# anomalies in potential features sometimes mirror
# anomalies in the label, which could make the column
# be (or seem to be) a powerful feature.
# Also, as you will see later in the course, you
# might be able to represent (pre-process) raw data
# in order to make columns into useful features.


# @title Define the functions that build and train a model
def build_model(my_learning_rate):
    """Tạo và biên dịch mô hình hồi quy tuyến tính đơn giản."""
    """Create and compile a simple linear regression model."""
    # Hầu hết các mô hình tf.keras đơn giản là tuần tự.
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Mô tả địa hình của mô hình.
    # Describe the topography of the model.
    # Địa hình của mô hình hồi quy tuyến tính đơn giản là một nút trong một lớp.
    # The topography of a simple linear regression model is a single node in a single layer.
    model.add(tf.keras.layers.Dense(units=1,
                                    input_shape=(1,)))

    # Compile the model topography into code that TensorFlow can efficiently execute.
    # Biên dịch địa hình mô hình thành mã mà TensorFlow có thể thực thi một cách hiệu quả.
    # Configure training to minimize the model's mean squared error.
    # Cấu hình đào tạo để giảm thiểu lỗi bình phương trung bình của mô hình.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(model, df, feature, label, epochs, batch_size):
    """Huấn luyện mô hình bằng dữ liệu."""
    """Train the model by feeding it data."""

    # Cung cấp các giá trị Feature và các giá trị Label cho mô hình.
    # Feed the model the feature and the label.
    # Mô hình sẽ đào tạo theo số Epochs được chỉ định,
    # The model will train for the specified number of epochs.
    history = model.fit(x=df[feature],
                        y=df[label],
                        batch_size=batch_size,
                        epochs=epochs)

    # Tập hợp Weight và Bias của mô hình được đào tạo.
    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # Danh sách Epoch được lưu trữ riêng biệt với phần còn lại của History.
    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Tập hợp History (ảnh chụp nhanh) của mỗi Epoch.
    # Isolate the error for each epoch.
    hist = pd.DataFrame(history.history)

    # Để theo dõi tiến trình đào tạo, chúng tôi sẽ chụp nhanh
    # MSE của mô hình tại mỗi epoch.
    # To track the progression of training, we're going to take a snapshot
    # of the model's root mean squared error at each epoch.
    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse


print("Defined the create_model and traing_model functions.")


# @title Define the plotting functions
def plot_the_model(trained_weight, trained_bias, feature, label):
    """Vẽ mô hình đào tạo với 200 ví dụ đào tạo ngẫu nhiên."""
    """Plot the trained model against 200 random training examples."""

    # Dán nhãn các trục.
    # Label the axes.
    plt.xlabel(feature)
    plt.ylabel(label)

    # Tạo một biểu đồ phân tán từ 200 điểm ngẫu nhiên của bộ dữ liệu.
    # Create a scatter plot from 200 random points of the dataset.
    random_examples = training_df.sample(n=200)
    plt.scatter(random_examples[feature], random_examples[label])

    # Create a red line representing the model.
    # Tạo một đường màu đỏ đại diện cho mô hình.
    # The red line starts at coordinates (x0, y0) and ends at coordinates (x1, y1).
    # Đường màu đỏ bắt đầu tại tọa độ (x0, y0) và kết thúc tại tọa độ (x1, y1).
    x0 = 0
    y0 = trained_bias
    x1 = 10000
    y1 = trained_bias + (trained_weight * x1)
    plt.plot([x0, x1], [y0, y1], c='r')

    # Render the scatter plot and the red line.
    # plt.show()


def plot_the_loss_curve(epochs, rmse):
    """Vẽ đường cong Loss, cho thấy Loss so với Epoch."""
    """Plot a curve of loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min() * 0.97, rmse.max()])
    plt.show()


print("Defined the plot_the_model and plot_the_loss_curve functions.")


def predict_house_values(n, feature, label):
    """Dự đoán giá trị ngôi nhà dựa trên Feature."""
    """Predict house values based on a feature."""

    batch = training_df[feature][10000:10000 + n]
    predicted_values = my_model.predict_on_batch(x=batch)

    print("feature   label          predicted")
    print("  value   value          value")
    print("          in thousand$   in thousand$")
    print("--------------------------------------")
    for i in range(n):
        print("%5.0f %6.0f %15.0f" % (training_df[feature][i],
                                      training_df[label][i],
                                      predicted_values[i][0]))


learning_rate = 0.001
epochs = 30
batch_size = 1000

my_feature = "median_income"
my_label = "median_house_value"

my_model = None

my_model = build_model(learning_rate)
weight, bias, epochs, rmse = train_model(my_model, training_df,
                                         my_feature, my_label,
                                         epochs, batch_size)

print("\nThe learned weight for your model is %.4f" % weight)
print("The learned bias for your model is %.4f\n" % bias)

plot_the_model(weight, bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)

predict_house_values(10, my_feature, my_label)
