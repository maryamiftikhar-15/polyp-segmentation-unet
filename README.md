🧠 Colorectal Polyp Segmentation using U-Net
This project uses the U-Net deep learning model to automatically detect and segment colorectal polyps in colonoscopy images. Polyp detection helps in early diagnosis and can potentially save lives by preventing colorectal cancer.

🔍 What this project does

Loads colonoscopy images and their corresponding polyp masks

Trains a U-Net model to highlight polyp areas automatically

Uses improved training techniques like:

Data augmentation

Dice + Focal Loss

IoU and Dice metrics

Early stopping and learning rate scheduler

Evaluates and visualizes the predictions

📁 Dataset Used

Kvasir-SEG
Download from Kaggle:
👉 https://www.kaggle.com/datasets/debeshjha1/kvasirseg/data

📂 Project Files

unet_polyp_segmentation.ipynb – Main training notebook

README.md – Project overview

🚀 Techniques Explored

To improve model performance and training stability, I experimented with the following techniques:

Combined loss functions (Dice + Focal Loss) for better segmentation

Metrics like IoU and Dice to track model quality

Data augmentation using Albumentations

Dropout and BatchNormalization for regularization

EarlyStopping and ReduceLROnPlateau callbacks

Visualization of predictions for qualitative assessment

✅ Why it Matters
This project shows how deep learning can support automatic medical image analysis, making the diagnosis process faster, accurate, and consistent.

