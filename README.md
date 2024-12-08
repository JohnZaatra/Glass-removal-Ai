# Glass Removal by Laser Surgery - Predicting Cataract Surgery Success

This repository contains the final project for my Bachelor's degree in **Information and Software Systems Engineering** at **Ben Gurion University**, Negev, Israel. The project is focused on **predicting the success of cataract surgery** through the use of advanced machine learning techniques, combining **tabular patient data** and **eye scan images** to predict post-surgery uncorrected visual acuity.

The model developed as part of this project serves as a **decision support tool for ophthalmologists**, providing them with reliable insights into the chances of surgery success, improving decision-making, and ultimately enhancing patient outcomes.

**About Barzilai Hospital**

Barzilai Hospital, founded in 1961, is a renowned medical facility in Ashkelon, Israel. Named after former Minister of Health Israel Barzilai, the hospital is affiliated with the Faculty of Health Sciences of Ben-Gurion University. Known for its high standards in professional and humane care, medical education, and research, Barzilai Hospital is committed to advancing healthcare in the region.

---

## 🚀 **Project Overview**

The project aims to **predict the likelihood of a successful cataract surgery** by evaluating the **post-surgery visual acuity** of patients. This is achieved through a combination of **tabular data** (patient and health-related information) and **eye scan images** (pre-surgery corneal scans).

### **Key Features:**

- **Tabular Data**: The dataset consists of over 130,000 rows and 46 features, including patient demographics, medical history, and pre-surgery eye conditions.
  - After preprocessing, the dataset was reduced to 53,000 rows and 25 input features.

- **Eye Scan Images**: The dataset includes nearly **107,000 eye scan images** of patients before surgery.
  - After image preprocessing and augmentation, the final dataset contains 53,000 images, ready for deep learning.

- **Combined Model**: The project leverages a hybrid model combining the outputs of an **MLP (Multilayer Perceptron)** for tabular data and **CNN (Convolutional Neural Network)** for image data, followed by a regression model that outputs the predicted success rate of the surgery.

---

## 🎯 **Problem Statement**

The primary goal of the project was to **predict the success of cataract surgery** by analyzing two important data types:
1. **Tabular data**: Medical records and demographic information.
2. **Eye scan images**: Pre-surgery corneal images of patients.

This was accomplished by integrating both data types into a **combined model**, offering **decision support tools** to doctors to predict the **post-surgery visual clarity** of patients, providing a more **accurate and reliable diagnosis**.

---

## 🔬 **Model Description**

### **Tabular Data Approach**:
1. **Data Preprocessing**: 
   - Removed irrelevant features and handled missing data.
   - Applied feature scaling and normalization to prepare the data for machine learning.
   
2. **Model**: 
   - **MLP (Multilayer Perceptron)** was used to model the relationship between the features and the surgery success prediction.
   - Extensive **hyperparameter tuning** was carried out for optimal performance.

### **Image Data Approach**:
1. **Preprocessing**: 
   - Resized, normalized, and augmented **107,000 eye scan images**.
   - Performed image preprocessing to exclude unnecessary IDs and focus on the main eye features.

2. **Model**:
   - **ResNet18, ResNet50, ResNet152** architectures were tested for CNN-based image classification.
   - Applied advanced techniques such as **image augmentation** to enhance the generalization of the model.
   
3. **Hyperparameter Tuning**: 
   - Tuned various hyperparameters to optimize learning rates, batch sizes, and other parameters.

### **Combined Model**:
1. **Integration**:
   - The features from the **MLP** model (tabular data) and **CNN** model (image data) were merged to form a **combined model**.
   
2. **Model Architecture**:
   - The final combined model used **regression analysis** to predict the uncorrected visual acuity post-surgery.
   - The prediction was then fed into a function to generate the final classification output, classifying the likelihood of a successful surgery.

---

**Solution**
![Solution](https://github.com/JohnZaatra/Glass-removal-Ai/blob/87474b1b5271abed5008326874b2785a7dca2024/presentation-and-work/Solution.png)


## 🔧 **Tools and Techniques Used**

- **Python**: Primary programming language used.
- **PyTorch**: For deep learning model development and training.
- **Scikit-learn**: For preprocessing and machine learning algorithms.
- **Pandas & NumPy**: For data manipulation and preprocessing.
- **OpenCV**: For image preprocessing and augmentation.
- **Matplotlib & Seaborn**: For data visualization and model performance evaluation.

---

## 📈 **Results and Achievements**

The model went through several **iterations** to fine-tune its performance. Below is a summary of the improvements made and their effects on the model:

- **Tabular Data (MLP)**: After experimenting with different architectures and hyperparameters, the model showed improved performance, with a significant reduction in overfitting and an increase in the validation accuracy.
  
- **Image Data (CNN)**: 
   - Initially, the model showed **93.83% training accuracy** but required further fine-tuning of architectures and hyperparameters.
   - After hyperparameter tuning, accuracy improved to **89.44% on the training set**, with significant improvements in validation performance as well.
  
- **Combined Model**: 
   - Integration of the tabular and image data led to significant improvements in the final prediction. The combined model outperformed the individual models in terms of both accuracy and generalization.

- **Masked Autoencoder**: Used for pretraining, further enhancing the performance of all models.

## 💡 How to Contribute

I welcome contributions to this project! If you have ideas for improvements, optimizations, or new features, feel free to fork the repository and submit a pull request.

Please follow the **coding standards**, and include **tests** where applicable. Contributions are highly encouraged to help improve the model’s accuracy, generalization, or introduce new features.

---

## 🎓 Acknowledgments

This project is part of a broader exploration into **Human Activity Recognition** and **medical prediction** using machine learning, applying **state-of-the-art techniques** in machine learning to a real-world healthcare problem.

Special thanks to:
- The **original dataset creators** for making the data publicly available.
- The **academic community** for their invaluable research contributions to time-series classification and medical prediction.

---

## 🔮 Conclusion

This project illustrates the **powerful intersection of machine learning** and **wearable technology** in the field of **Human Activity Recognition**. By exploring **multiple models**, **optimizing their performance**, and experimenting with **novel techniques** such as masked autoencoders, this project demonstrates a deep understanding of **time-series data** and **model refinement**.

The skills developed throughout this project are directly applicable to real-world challenges in **data science**, **machine learning**, and **healthcare technology**. I am confident that these experiences will be pivotal in tackling future challenges and developing solutions that improve patient outcomes.

---

## ✨ Thank You for Exploring!

Thank you for exploring this project! Feel free to reach out with any questions, suggestions, or feedback.
