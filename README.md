# PathPredict AI – Intelligent Travel Time & Recommendation System

PathPredict AI is a machine learning and NLP-powered smart travel assistant that predicts travel time and generates personalized travel recommendations using Artificial Neural Networks (ANN).

The system combines regression modeling, contextual filtering, and lightweight NLP logic to provide accurate travel-time estimation and intelligent trip planning.

---

## 🚀 Key Features

- Travel time prediction using ANN regression (R² = 0.91)
- Personalized destination recommendations
- NLP-based itinerary generation
- Real-time travel duration estimation
- Optimized preprocessing for faster inference

---

## 🧠 Model Architecture

| Component            | Technology Used |
|----------------------|----------------|
| Regression Model     | Artificial Neural Network (ANN) |
| Input Features       | Distance, Traffic Level, Weather, Time of Day |
| Data Preprocessing   | StandardScaler Normalization |
| NLP Engine           | Keyword Extraction + Context Filtering |
| Evaluation Metric    | R² Score |

---

## 📊 Model Performance

| Metric       | Value |
|--------------|--------|
| R² Score     | 0.91 |

The model explains 91% of travel-time variance, indicating strong predictive performance.

---

## ⚙️ Training Workflow

1. Data loading and preprocessing  
2. Feature normalization  
3. Train-test split  
4. ANN training with optimized parameters  
5. Performance evaluation using R²  
6. Model deployment for inference  

---

## 🧪 Example Usage

```python
from tensorflow.keras.models import load_model
import numpy as np
import pickle

model = load_model("travel_time_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

user_input = np.array([[distance, traffic_level, weather_code, time_of_day]])

scaled_input = scaler.transform(user_input)

predicted_time = model.predict(scaled_input)

print("Predicted Travel Time:", predicted_time[0][0], "minutes")
```

---

## 📁 Project Structure

```
PathPredict-AI/
│── travel_time_ANN.ipynb
│── trip_time_dataset.csv
│── travel_time_model.h5
│── scaler.pkl
│── README.md
```

---

## 🔮 Future Enhancements

- Deep learning–based itinerary generation
- LLM-powered conversational travel assistant
- Live traffic & weather API integration
- Region-wise dataset expansion

---

## 🤝 Contributing

Pull requests and issue discussions are welcome.

---

## 👨‍💻 Author

Shubham Chaudhary  
GitHub: https://github.com/shubhamchaudhary-dev
