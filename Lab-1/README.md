# 🧠 Find-S Algorithm Implementation (Machine Learning Lab)

## 📌 Overview

This project demonstrates the implementation of the **Find-S Algorithm**, a fundamental concept learning algorithm in Machine Learning.

The algorithm identifies the **most specific hypothesis** that fits all positive training examples.

---

## 🎯 Objective

* To understand the working of the Find-S algorithm
* To implement it using Python
* To train a model using labeled dataset
* To derive the most specific hypothesis

---

## 📂 Dataset Description

The dataset (`ejoysport.csv`) contains weather conditions and whether a sport can be enjoyed.

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

The **Find-S Algorithm** works as follows:

1. Initialize the most specific hypothesis
2. Consider only positive examples
3. For each positive example:

   * Compare with current hypothesis
   * Generalize if needed (replace with '?')
4. Ignore negative examples

---

## 💻 Code Implementation

```python
import pandas as pd
import numpy as np

data = pd.read_csv('enjoysport.csv')

concepts = np.array(data)[:, :-1]
target = np.array(data)[:, -1]

def train(con, tar):
    for i, val in enumerate(tar):
        if val == 'yes':
            specific_h = con[i].copy()
            break
            
    for i, val in enumerate(con):
        if tar[i] == 'yes':
            for x in range(len(specific_h)):
                if val[x] != specific_h[x]:
                    specific_h[x] = '?'
    return specific_h

print(train(concepts, target))
```

---

## ✅ Output

```
['sunny' 'warm' '?' 'strong' '?' '?']
```

---

## 📊 Result Interpretation

The final hypothesis:

* Sky = sunny
* Air Temp = warm
* Humidity = ? (any value)
* Wind = strong
* Water = ? (any value)
* Forecast = ? (any value)

👉 This represents the **most specific generalization** that satisfies all positive examples.

---

## 🚀 How to Run

1. Clone the repository
2. Navigate to project folder
3. Create virtual environment:

   ```
   python -m venv venv
   ```
4. Activate environment:

   ```
   venv\Scripts\activate
   ```
5. Install dependencies:

   ```
   pip install pandas numpy
   ```
6. Run the notebook or script

---

## 📌 Key Learning

* Understanding hypothesis space
* Concept learning basics
* Difference between specific and general hypotheses
* Practical implementation of ML theory

---

## 🔗 Future Improvements

* Visualize hypothesis evolution
* Implement Candidate Elimination Algorithm
* Use larger real-world datasets

---

## 👩‍💻 Author

**Tejaswini Patil**
📧 [patiltejaswini873@gmail.com](mailto:patiltejaswini873@gmail.com)

---

⭐ If you found this useful, consider giving a star!
