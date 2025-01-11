# Recyclable and Non-Recyclable Waste Classification

## Objective
Develop an automated system using object detection and classification to distinguish between recyclable and non-recyclable household waste. This project leverages a diverse dataset to build robust machine learning models for waste categorization.

---

## Folder Structure

- **`data/processed/`**: Preprocessed data ready for model training and evaluation.
- **`data/README.md`**: Includes the dataset description and download link.
- **`src/`**: Contains Python scripts for various stages of the project:
  - `preprocessing.py`: Preprocessing raw data for training and testing.
  - `model.py`: Defines the model architecture.
  - `train.py`: Handles the training pipeline.
  - `evaluate.py`: Evaluates model performance.
- **`notebooks/`**: Jupyter notebooks for exploratory data analysis and visualization.
- **`results/`**: Contains training results, graphs, and logs.
- **`README.md`**: Overview of the project, setup instructions, and usage.
- **`requirements.txt`**: Lists all required Python dependencies.

---

## Dataset

The dataset used in this project is the **Recyclable and Household Waste Classification Dataset**. It includes 15,000 images across 30 categories, divided into `default` and `real_world` scenarios.

### Dataset Details
- Size: ~920MB
- Categories: Plastic, Paper, Cardboard, Glass, Metal, Organic Waste, Textiles.
- Images: Provided in PNG format with 256x256 resolution.

### Download the Dataset
The dataset is too large to be included directly in this repository. It will automatically download when executing the code files.

Download it from [Kaggle](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification) if needed. 

---

## Requirements

Install project dependencies using the following command:

```bash
pip install -r requirements.txt
