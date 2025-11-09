Bird Species Classification using Deep Learning (ViT + Streamlit)

This project classifies bird species from images using Transfer Learning. A Vision Transformer (ViT) model is fine-tuned on a labeled bird species image dataset and deployed using Streamlit for real-time predictions.

Features:

Upload an image and get the predicted bird species.

Displays confidence score for the prediction.

Uses Vision Transformer (ViT) model for high accuracy.

Transfer learning reduces training time and improves performance.

Simple and interactive Streamlit web interface.

System Architecture:
User → Streamlit UI → AutoImageProcessor → Fine-Tuned ViT Model → Output Prediction

Project Structure:
app.py → Streamlit web app
model_loader.py → Loads the trained model
train.py → Script to fine-tune the model
segmentations/ → Dataset folder (each class contains images of that bird)
fine_tuned_bird_classifier/ → Saved trained model

Dataset Format:
The dataset must be stored like this:
segmentations/
ClassName_1/
image1.png
image2.png
ClassName_2/
image1.png
image2.png
Each folder represents one bird species.

Installation Steps:

Clone the repository:
git clone https://github.com/Harshitha-79/Bird_Species_Classification_DeepLearning.git

Open project folder:
cd Bird_Species_Classification_DeepLearning

Install required libraries:
pip install streamlit transformers Pillow numpy
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install datasets

Run the Application:
streamlit run app.py
The app will open at: http://localhost:8501

Training the Model (Optional Step):
If you want to fine-tune the model on your dataset:
python train.py
The trained model will be saved in:
fine_tuned_bird_classifier/

Model Used:
Model: Vision Transformer (ViT-Base-Patch16-224)
Training Method: Transfer Learning
Output: Class label + confidence score

Results:

Accurately recognizes bird species from images.

Works on varied lighting conditions and angles.

Demonstrates practical use of deep learning in classification systems.