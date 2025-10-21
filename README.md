# AI Stock Price Predictor (Multivariate LSTM) 📈

This is a full-stack, decoupled web application that predicts the next **7 days** of closing stock prices using a **Multivariate Long Short-Term Memory (LSTM)** neural network.

The model uses the last 60 days of **Open, High, Low, Close, and Volume** data to forecast the future closing prices for 48 major US stocks.

This project showcases skills in:
* **Backend Development:** Python, Flask API
* **Deep Learning:** TensorFlow, Keras (LSTM Model Building & Training)
* **Data Science:** Pandas, NumPy, Scikit-learn (Data Processing, Scaling)
* **Frontend Development:** HTML, CSS, JavaScript (Chart.js for visualization)
* **API Integration:** Fetching live data via `yfinance`.
* **Version Control:** Using **Git LFS** to manage large model files.

![Project Screenshot](image.png)


---

## 🚀 Features

* **Full-Stack Architecture:** Decoupled JS frontend consumes a Python/Flask backend API.
* **Advanced AI Model:** Uses a Multivariate LSTM trained on 5 features (O, H, L, C, V) per stock.
* **7-Day Forecasting:** Predicts a sequence of the next 7 trading days' closing prices.
* **Dynamic Model Loading:** Backend dynamically loads the specific `.keras` model and `MinMaxScaler` for the requested stock.
* **Data Visualization:** Displays a 1-year historical price chart using Chart.js.
* **Includes Pre-trained Models:** 48 `.keras` models and scalers are included via Git LFS, so you don't need to run the lengthy training script.

---

## 🛠️ Tech Stack

* **Backend:** Python 3.x, Flask, TensorFlow, Keras, Scikit-learn, Pandas, NumPy, yfinance, joblib
* **Frontend:** HTML5, CSS3, JavaScript (ES6+), Chart.js
* **Core Concepts:** REST API, Deep Learning, Time Series Forecasting, Multivariate LSTM, Data Normalization, Git LFS

---

## 🏁 How to Run This Project

You need Python 3.10+ and Git LFS installed.

### 1. Clone the Repository (with LFS)

Make sure you have Git LFS installed ([git-lfs.com](https://git-lfs.com/)). Then clone:

```bash
git clone [https://github.com/YOUR_USERNAME/AI-Stock-Predictor.git](https://github.com/YOUR_USERNAME/AI-Stock-Predictor.git)
cd AI-Stock-Predictor
```
Git LFS will automatically download the large model files during the clone process. This might take a few minutes.

### 2. Run the Backend (API)

1.  Navigate to the `backend` directory:
    ```bash
    cd backend
    ```

2.  Create and activate a Python virtual environment:
    ```bash
    # Create venv
    python -m venv venv

    # Activate (Windows)
    .\venv\Scripts\activate
    # Activate (macOS/Linux)
    source venv/bin/activate
    ```

3.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
    *(This will install TensorFlow and may take a while).*

4.  Run the Flask server:
    ```bash
    python app.py
    ```
    The server will start on `http://127.0.0.1:5000`. It will load models as they are requested.

### 3. Run the Frontend (Client)

1.  Open the `frontend` directory in your file explorer.
2.  Double-click the `index.html` file to open it in your web browser.

You can now use the application! Enter any of the 48 supported stock tickers (e.g., AAPL, TSLA, JPM, MSFT) to get a 7-day forecast.

*(Note: The `train_model.py` script is included but takes over an hour to run. The necessary pre-trained models are already provided in the `backend/models` directory via Git LFS.)*

