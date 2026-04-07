# 🧠 Candidate Elimination Algorithm (Machine Learning Lab)

## 📌 Overview

This project demonstrates the implementation of the **Candidate Elimination Algorithm**, a supervised learning algorithm used to compute the **version space** of hypotheses that are consistent with training data.

Unlike Find-S, this algorithm maintains both:

* **Specific boundary (S)**
* **General boundary (G)**

---

## 🎯 Objective

* To understand the concept of version space
* To implement Candidate Elimination algorithm using Python
* To update hypotheses based on positive and negative examples
* To derive the most specific and most general hypotheses

---

## 📂 Dataset Description

The dataset (`data.csv`) contains weather-related attributes used to determine whether a sport can be enjoyed.

### Attributes:

* Sky
* Air Temperature
* Humidity
* Wind
* Water
* Forecast

### Target:

* Enjoy Sport (Yes / No)

---

## 🛠️ Technologies Used

* Python 🐍
* Pandas
* NumPy
* Jupyter Notebook

---

## 📜 Algorithm Explanation

The **Candidate Elimination Algorithm** works as follows:

* Initialize:

  * **S** → Most specific hypothesis
  * **G** → Most general hypothesis

* For each training example:

### ✔ If Positive Example:

* Remove inconsistent hypotheses from G
* Generalize S minimally

### ❌ If Negative Example:

* Remove inconsistent hypotheses from S

* Specialize G minimally

* Continue until all examples are processed

👉 This results in a **version space** bounded by S and G 

---

## 💻 Code Implementation

```python
import numpy as np 
import pandas as pd

data = pd.read_csv('data.csv')
concepts = np.array(data.iloc[:,0:-1])
target = np.array(data.iloc[:,-1])  

def learn(concepts, target): 
    specific_h = concepts[0].copy()  
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]     

    for i, h in enumerate(concepts):
        if target[i] == "yes":
            for x in range(len(specific_h)): 
                if h[x] != specific_h[x]:                    
                    specific_h[x] = '?'                     
                    general_h[x][x] = '?'

        if target[i] == "no":            
            for x in range(len(specific_h)): 
                if h[x] != specific_h[x]:                    
                    general_h[x][x] = specific_h[x]                
                else:                    
                    general_h[x][x] = '?'        

    general_h = [h for h in general_h if h != ['?'] * len(specific_h)]
    return specific_h, general_h

s_final, g_final = learn(concepts, target)

print("Final Specific_h:", s_final)
print("Final General_h:", g_final)
```

---

## ✅ Output

```
Final Specific_h:
['sunny' 'warm' '?' 'strong' '?' '?']

Final General_h:
[['sunny', '?', '?', '?', '?', '?'], ['?', 'warm', '?', '?', '?', '?']]
```

---

## 📊 Result Interpretation

* **Specific Hypothesis (S):**
  Represents the most specific rule consistent with all positive examples

* **General Hypothesis (G):**
  Represents the most general rules that still satisfy the training data

👉 Together, they define the **version space**

---

## 🚀 How to Run

1. Create virtual environment
2. Activate environment
3. Install dependencies:

   ```
   pip install pandas numpy
   ```
4. Run the notebook

---

## 📌 Key Learning

* Understanding version space
* Difference between Find-S and Candidate Elimination
* Handling both positive and negative examples
* Hypothesis generalization and specialization

---

## 🔗 Future Improvements

* Visualize hypothesis boundaries
* Implement real-world datasets
* Compare with other ML algorithms

---

## 👩‍💻 Author

**Tejaswini Patil**
📧 [patiltejaswini873@gmail.com](mailto:patiltejaswini873@gmail.com)

---

⭐ If you found this useful, consider giving a star!
