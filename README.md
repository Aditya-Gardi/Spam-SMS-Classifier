# Spam-SMS-Classifier
The Spam SMS Classifier is a machine learning project using NLP and supervised algorithms to detect and classify spam messages. By identifying unsolicited and harmful SMS, this classifier enhances user experience and security, reducing unwanted message clutter and protecting against scams and phishing attempts.

**Features**
- Data Preprocessing: Cleaning and preprocessing SMS text data.
- Feature Extraction: Extracting meaningful features from text using techniques such as TF-IDF and word embeddings.
- Model Training: Training various machine learning models such as Naive Bayes, SVM, and Neural Networks.
- Model Evaluation: Evaluating models using metrics like accuracy, precision, recall, and F1-score.
- Deployment: Deploying the model using Flask for real-time spam detection.

**Requirements**
- Python 3.7+
- Jupyter Notebook
- Flask
- Scikit-learn
- TensorFlow / Keras
- NLTK
- Pandas
- NumPy
- Matplotlib
- Seaborn

**Installation**
1. Clone the repository:
   git clone https://github.com/yourusername/spam-sms-classifier.git
   cd spam-sms-classifier
2. Create a virtual environment:
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
3. Install the dependencies:
   pip install -r requirements.txt
4. Download NLTK data:
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')

**Data**
The dataset used in this project consists of labeled SMS messages, which include examples of both spam and ham. The data can be obtained from sources like the UCI Machine Learning Repository or Kaggle.

**Usage**
1. Data Preprocessing:
   Run data_preprocessing.py to clean and preprocess the SMS text data.
2. Feature Extraction:
   Extract features using feature_extraction.py.
3. Model Training:
   Train the model using train_model.py. You can specify the model type as an argument (e.g., Naive Bayes, SVM, etc.).
4. Model Evaluation:
   Evaluate the trained model using evaluate_model.py.
5. Deployment:
   Deploy the model using Flask.
6. Testing the Deployment:
   Use tools like Postman or curl to send requests to the deployed model and get predictions.

**Contributing**
1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Commit your changes (git commit -am 'Add new feature').
4. Push to the branch (git push origin feature-branch).
5. Create a new Pull Request.

**License**
This project is licensed under the MIT License-
MIT License

Copyright (c) 2024 Aditya Gardi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

**Acknowledgements**
- Datasets from UCI Machine Learning Repository
- NLTK
- Scikit-learn
- TensorFlow and Keras

**Contact**
For any questions or suggestions, please contact cdtaditya.gardi7475@gmail.com.
