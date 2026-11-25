# Human Activity Recognition – End-to-End Deep Learning System

Deep learning pipeline for recognising human activities from smartphone sensor data and video.

## Results
| Task | Dataset | Model | Test Accuracy |
|------|---------|-------|---------------|
| Sensor-based HAR | UCI HAR (6 activities) | MLP | 95.1% |
| Sensor-based HAR | UCI HAR | 1D CNN | 94.6% |
| Video action recognition | UCF101 (8 classes) | MobileNetV2 + GRU + Attention | 98.75% |
| Generative modelling | UCF101 frames | Conv Autoencoder, DCGAN | — |

## Tech Stack
TensorFlow/Keras, OpenCV, Scikit-learn, NumPy, Pandas, Streamlit

## Run the dashboard
pip install -r requirements.txt
streamlit run app.py