from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFileDialog, QTextEdit, QComboBox
)
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from tensorflow.keras.losses import MeanSquaredError


class LSTMGui(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("LSTM NOX Predictor")
        self.setGeometry(100, 100, 600, 800)

        # Available models and corresponding target sensors
        self.models = {
            "lstm_model_nox_100_2": "Z01970",
            "lstm_model_sox_100": "Z00518",
            "lstm_model_nox_50_onehot": "Z01970",
            "lstm_model_sox_50_onehot": "Z00518",
            "kan_model_nox_v1": "Z01970",
            "kan_model_sox_v1": "Z00518"
        }

        # Load the default model
        self.model = None
        self.scaler = None
        self.target_sensor = None
        self.load_model_and_scaler("lstm_model_nox_100_2")  # Default model

        self.window_size = 10  

        # Central widget and scroll area
        central_widget = QWidget()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(central_widget)
        self.setCentralWidget(self.scroll_area)

        # Layout
        layout = QVBoxLayout(central_widget)

        # Model selection dropdown
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(self.models.keys())
        self.model_dropdown.currentTextChanged.connect(self.on_model_change)
        layout.addWidget(QLabel("Select Model:"))
        layout.addWidget(self.model_dropdown)

        # Time display
        self.time_label = QLabel("Time: ")
        layout.addWidget(self.time_label)

        # Upload CSV Button
        self.upload_button = QPushButton("Upload Test Data CSV")
        self.upload_button.clicked.connect(self.load_csv)
        layout.addWidget(self.upload_button)

        # Predict Button
        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict_from_csv)
        layout.addWidget(self.predict_button)

        # Prediction result
        self.result_label = QLabel("Prediction: ")
        layout.addWidget(self.result_label)

        # Text box to display status or details
        self.status_textbox = QTextEdit()
        self.status_textbox.setReadOnly(True)
        self.status_textbox.setPlaceholderText("Status updates will appear here...")
        layout.addWidget(self.status_textbox)

        # Plotting area
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Storage
        self.predictions = []
        self.denormalized_predictions = []
        self.test_data = None
        self.sequences = None
        self.y_values = None

    def load_model_and_scaler(self, model_name):
        """Load the selected model and scaler."""
        try:
            # Check if the status_textbox is already initialized
            if hasattr(self, 'status_textbox'):
                # self.status_textbox.append(f"Loading model: {model_name}...")
                if model_name == "lstm_model_nox_100_2":
                    self.status_textbox.append("LSTM NOX model with Dropping Initial Parameters")
                elif model_name == "lstm_model_sox_100":
                    self.status_textbox.append("LSTM SOX model with Dropping Initial Parameters")
                elif model_name == "lstm_model_nox_50_onehot":
                    self.status_textbox.append("LSTM NOX model with One-Hot Encoding")
                elif model_name == "lstm_model_sox_50_onehot":
                    self.status_textbox.append("LSTM SOX model with One-Hot Encoding")
                elif model_name == "kan_model_nox_v1":
                    self.status_textbox.append("KAN NOX model")
                elif model_name == "kan_model_sox_v1":
                    self.status_textbox.append("KAN SOX model")

            # self.model = load_model(f"{model_name}.h5", custom_objects={'mse': MeanSquaredError()})
            # self.model = load_model(f"{model_name}.keras", custom_objects={'mse': MeanSquaredError()})

            if model_name == "kan_model_nox_v1" or model_name == "kan_model_sox_v1":
                self.model = load_model(f"{model_name}.keras", custom_objects={'mse': MeanSquaredError()})
            else:
                self.model = load_model(f"{model_name}.h5", custom_objects={'mse': MeanSquaredError()})

            # with open("scaler_noinit.pkl", "rb") as f:
            #     self.scaler = pickle.load(f)

            if model_name == "lstm_model_nox_50_onehot" or model_name == "lstm_model_sox_50_onehot":
                print("LOADING ONEHOT SCALER")
                self.status_textbox.append("Loading one-hot scaler...")
                with open("scaler_onehot.pkl", "rb") as f:
                    self.scaler = pickle.load(f)
            else:
                with open("scaler_noinit.pkl", "rb") as f:
                    self.scaler = pickle.load(f)

            self.target_sensor = self.models[model_name]

            if hasattr(self, 'status_textbox'):
                self.status_textbox.append(f"Model loaded successfully! Target sensor: {self.target_sensor}")

        except Exception as e:
            if hasattr(self, 'status_textbox'):
                self.status_textbox.append(f"Error loading model: {e}")

    def on_model_change(self, model_name):
        """Handle model selection from the dropdown."""
        self.load_model_and_scaler(model_name)


    def load_csv(self):
        """Load test data from a CSV file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "",
            "CSV Files (*.csv);;All Files (*)", 
            options=options
        )
        if file_path:
            try:
                self.test_data = pd.read_csv(file_path)
                self.result_label.setText(f"Test data loaded: {file_path}")
                self.prepare_data()
            except Exception as e:
                self.result_label.setText(f"Error loading CSV: {e}")

    def prepare_data(self):
        """Prepare the test data by converting time, scaling, and creating sequences."""
        try:
            self.status_textbox.append("Preparing data...")

            # Load the original training feature order from a saved file
            with open("expected_features.pkl", "rb") as f:
                feature_columns = pickle.load(f)

            if self.model_dropdown.currentText() in ["lstm_model_nox_50_onehot", "lstm_model_sox_50_onehot"]:
                # One-hot encode categorical columns
                self.test_data = pd.get_dummies(self.test_data, columns=['Class', 'Load', 'SeaTemp', 'WindSpeed'])

                # Convert bool to integer
                for col in self.test_data.columns:
                    if self.test_data[col].dtype == 'bool':
                        self.test_data[col] = self.test_data[col].astype(int)

                # Reindex to match training data columns
                self.test_data = self.test_data.reindex(columns=feature_columns, fill_value=0)

            else:
                # Drop categorical columns for non-onehot models
                categorical_columns = ['Class', 'Load', 'SeaTemp', 'WindSpeed']
                self.test_data.drop(columns=categorical_columns, inplace=True, errors='ignore')

            # Identify sensor columns
            # sensor_columns = self.test_data.columns.difference(['RunId', 'Time'])

            sensor_columns = self.test_data.columns.tolist()
            sensor_columns.remove('RunId')
            sensor_columns.remove('Time')

            # Scale sensor data
            self.test_data[sensor_columns] = self.scaler.transform(self.test_data[sensor_columns])

            # # Create sequences
            # self.sequences, self.y_values = self.create_sequences(
            #     self.test_data, self.window_size, self.target_sensor
            # )

            if self.model_dropdown.currentText() == "kan_model_nox_v1" or self.model_dropdown.currentText() == "kan_model_sox_v1":
                self.sequences, self.y_values = self.create_sequences(self.test_data, 1, self.target_sensor)

                # Reshape test data
                self.sequences = self.sequences.reshape((self.sequences.shape[0], 1 * self.sequences.shape[2]))
                self.sequences = self.sequences.astype(np.float32)
            
            else:
                self.sequences, self.y_values = self.create_sequences(self.test_data, self.window_size, self.target_sensor)

            self.result_label.setText("Test data prepared: Time converted, scaled, and sequenced.")
            
        except Exception as e:
            self.result_label.setText(f"Error preparing data: {e}")


    def create_sequences(self, data, window_size, target_sensor, time_col='Time'):
        """Create sequences from the test data."""
        X, y = [], []
        for run_id in data['RunId'].unique():
            filtered_data = data[data['RunId'] == run_id]
            sorted_data = filtered_data.sort_values(by=time_col)
            run_data = sorted_data.drop(columns=['RunId', time_col])
            target_data = run_data[target_sensor].values
            input_data = run_data.drop(columns=[target_sensor]).values

            for i in range(len(run_data) - window_size + 1):
                seq_X = input_data[i:i + window_size]
                seq_y = target_data[i + window_size - 1]
                X.append(seq_X)
                y.append(seq_y)

        return np.array(X), np.array(y)

    def predict_from_csv(self):
        if self.test_data is None:
            self.result_label.setText("Please upload a test data CSV first.")
            return

        try:
            self.status_textbox.append("Running predictions...")
            predictions = self.model.predict(self.sequences)
            self.predictions = predictions.flatten()

            sensor_columns = self.test_data.columns.difference(['RunId', 'Time']).tolist()
            predictions_2d = predictions.reshape(-1, 1)
            y_values_2d = self.y_values.reshape(-1, 1)
            num_features = self.scaler.n_features_in_

            dummy_array_pred = np.zeros((predictions_2d.shape[0], num_features))
            dummy_array_true = np.zeros((y_values_2d.shape[0], num_features))
            target_index = sensor_columns.index(self.target_sensor)
            dummy_array_pred[:, target_index] = predictions_2d[:, 0]
            dummy_array_true[:, target_index] = y_values_2d[:, 0]
            denormalized_predictions = self.scaler.inverse_transform(dummy_array_pred)[:, target_index]
            denormalized_y_true = self.scaler.inverse_transform(dummy_array_true)[:, target_index]
            self.denormalized_predictions = denormalized_predictions
            denormalized_mse = np.mean((denormalized_predictions - denormalized_y_true) ** 2)

            self.status_textbox.append(f"MSE: {denormalized_mse:.5f}")
            self.update_plot()

            # Display predictions vs true values in the status textbox
            self.status_textbox.append("Predictions")
            for idx, (prediction, true_value) in enumerate(zip(denormalized_predictions, denormalized_y_true)):
                self.status_textbox.append(f"{idx + 1}: Predicted = {prediction:.4f}, True = {true_value:.4f}")


            self.result_label.setText("Predictions completed. Check the plot.")
        except Exception as e:
            self.result_label.setText(f"Error during prediction: {e}")

    def update_plot(self):
        if self.model_dropdown.currentText() == "lstm_model_nox_50_onehot" or self.model_dropdown.currentText() == "lstm_model_nox_100_2" or self.model_dropdown.currentText() == "kan_model_nox_v1":
            self.ax.clear()
            self.ax.plot(self.denormalized_predictions, marker='o', label='Predicted')
            self.ax.set_title('Predictions NOX')
            self.ax.set_xlabel('Time Index')
            self.ax.set_xticks(range(len(self.denormalized_predictions)))
            self.ax.set_ylabel('Value')
            self.ax.get_yaxis().get_major_formatter().set_useOffset(False)
            self.ax.ticklabel_format(style='plain', axis='y')
            self.ax.grid(True)
            self.ax.legend()
            self.canvas.draw()

        else:
            self.ax.clear()
            self.ax.plot(self.denormalized_predictions, marker='o', label='Predicted')
            self.ax.set_title('Predictions SOX')
            self.ax.set_xlabel('Time Index')
            self.ax.set_xticks(range(len(self.denormalized_predictions)))
            self.ax.set_ylabel('Value')
            self.ax.get_yaxis().get_major_formatter().set_useOffset(False)
            self.ax.ticklabel_format(style='plain', axis='y')
            self.ax.grid(True)
            self.ax.legend()
            self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = LSTMGui()
    gui.show()
    sys.exit(app.exec_())


