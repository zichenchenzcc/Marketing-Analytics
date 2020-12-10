# OpenTable reviews scraping and rating analysis

## **Two restaurants for scraping**

---

 ### 1. Buttermilk & Bourbon 
 
 ([https://www.opentable.com/r/buttermilk-and-bourbon-boston](https://www.opentable.com/r/buttermilk-and-bourbon-boston))

 - *1622 reviews and 4.7 rating*
 
 ![](/images/buttermilk_overview.png)
 
---
  
 - *Rating distribution*
 
 ![](/images/buttermilk_rating.png)

---

 ### 2. Osteria Nino ### ([https://www.opentable.com/osteria-nino](https://www.opentable.com/osteria-nino))
 
 ### 2. [Osteria Nino](https://www.opentable.com/osteria-nino)

 - *864 reviews and 4.2 rating*
 
 ![](/images/osteria_overview.png)
 
---
  
 - *Rating distribution*
 
 ![](/images/osteria_rating.png)
 	
---
  
## **Scraping reviews and ratings**

```python
import pandas as pd
import re
import time
from bs4 import BeautifulSoup
from selenium import webdriver

path_to_driver = 'C:\chromedriver\chromedriver_win32-85\chromedriver.exe'
driver = webdriver.Chrome(executable_path = path_to_driver)
# Buttermilk & Bourbon
url = 'https://www.opentable.com/r/buttermilk-and-bourbon-boston?originId=2&corrid=4bee8a16-0f6a-432c-82f6-5affef6c2098&avt=eyJ2IjoyLCJtIjowLCJwIjowLCJzIjowLCJuIjowfQ&p=2020-12-07T00%3A00%3A00'
# Osteria Nino
#url = 'https://www.opentable.com/osteria-nino?corrid=32a24c34-bd92-43f0-a63c-9484c5293dc5&avt=eyJ2IjoyLCJtIjoxLCJwIjowLCJzIjowLCJuIjowfQ&p=2&sd=2020-12-09T19%3A00%3A00'
driver.get(url)
               
review = []
rating = []                
condition = True
while (condition):
    reviews = driver.find_elements_by_xpath("//div[@class='reviewListItem oc-reviews-91417a38']")
    for i in range(len(reviews)):
        soup = BeautifulSoup(reviews[i].get_attribute('innerHTML'))
        review.append(re.findall('style="">(.*?)</p>',str(soup))[0])
        rating.append(re.findall('div aria-label="(.*?) out of 5 ',str(soup))[0])
    try:
        driver.find_element_by_xpath("//button[@aria-label = 'next-page']").click()
        time.sleep(1)
    except:
        condition = False
        
df = pd.concat([pd.DataFrame(review),pd.DataFrame(rating)],axis = 1)
df.columns = ['review','rating']
df.rating = df.rating.astype('int')
df.to_csv('buttermilk_review.csv')
#df.to_csv('osteria_nino_review.csv')
```

---

 - *Dataframe*
 
 ![](/images/dataframe.png)

---

- *Use all the reviews to make a wordcloud in the shape of a cauldron*
 
 ![](/images/word_cloud.png)
 
---

## **Run different models with different hyperparameters to predict ratings in test data**

- *Import tools*

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.linear_model            import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

```

---

- *Transfer and split data into training, validating, and test set*

``` python
np.random.seed(12)
df = pd.read_csv('buttermilk_review.csv')
df['ML_group']   = np.random.randint(100,size = df.shape[0])
df              = df.sort_values(by='ML_group').reset_index()
df.drop(df.columns[[0,1]],axis = 1,inplace=True)
inx_train         = df.ML_group<80                     
inx_valid         = (df.ML_group>=80)&(df.ML_group<90)
inx_test          = (df.ML_group>=90)
corpus          = df.review.to_list()
ngram_range     = (1,1)
max_df          = 0.85
min_df          = 0.01
vectorizer      = CountVectorizer(lowercase   = True,
                                  ngram_range = ngram_range,
                                  max_df      = max_df     ,
                                  min_df      = min_df     );
                                  
X               = vectorizer.fit_transform(corpus)
Y = df.rating
Y_train   = df.rating[inx_train].to_list()
Y_valid   = df.rating[inx_valid].to_list()
Y_test    = df.rating[inx_test].to_list()
X_train   = X[np.where(inx_train)[0],:].toarray()
X_valid   = X[np.where(inx_valid)[0],:].toarray()
X_test    = X[np.where(inx_test) [0],:].toarray()
```

---

- *Define functions to calculate accuracy score with/without offset*

``` python
def accuracy_offset(cm):
    accuracy = (cm.diagonal(offset=-1).sum()+cm.diagonal(offset=0).sum()+cm.diagonal(offset=1).sum())/cm.sum()
    return accuracy

def accuracy(cm):
    accuracy = cm.diagonal(offset=0).sum()/cm.sum()
    return accuracy

test_score_by_model_accuracy_offset = {}
test_score_by_model_accuracy = {}
```

---

- *1. Linear Model*

``` python
lm  = LinearRegression()
lm.fit(X_train, Y_train)
df['N_star_hat_reg'] = np.concatenate(
        [
                lm.predict(X_train),
                lm.predict(X_valid),
                lm.predict(X_test )
        ]
        ).round().astype(int)

df.loc[df['N_star_hat_reg']>5,'N_star_hat_reg'] = 5
df.loc[df['N_star_hat_reg']<1,'N_star_hat_reg'] = 1
cm_lm = confusion_matrix(df.iloc[1476:1623]['rating'], df.iloc[1476:1623]['N_star_hat_reg'])
test_score_by_model_accuracy_offset['Linear Model'] = accuracy_offset(cm_lm)
test_score_by_model_accuracy['Linear Model'] = accuracy(cm_lm)
```

---

- *2. Logistic Model*

``` python
accuracy_log = []
c_list = [0.01,0.1,1.0,10.0,100.0]
for c in c_list:
    log = LogisticRegression(multi_class="multinomial",solver="sag", 
                             C=c, max_iter=10000).fit(X_train, Y_train)
    cm_log = confusion_matrix(log.predict(X_valid),Y_valid)
    accuracy_log.append(accuracy(cm_log))

log = LogisticRegression(multi_class="multinomial",solver="sag", 
                         C=c_list[np.argmax(accuracy_log)], max_iter=10000).fit(X_train, Y_train)
cm_log = confusion_matrix(log.predict(X_test),Y_test)
test_score_by_model_accuracy_offset['Logistic'] = accuracy_offset(cm_log)
test_score_by_model_accuracy['Logistic'] = accuracy(cm_log)
```

---

- *3. K-Nearest Neighbors*


``` python
accuracy_knn = []
neighbor_list = list(np.arange(2,11))
for k in neighbor_list:
    knn      = KNeighborsClassifier(n_neighbors=k).fit(X_train, Y_train)
    cm_knn = confusion_matrix(knn.predict(X_valid),Y_valid)
    accuracy_knn.append(accuracy(cm_knn))

knn = KNeighborsClassifier(n_neighbors = np.argmax(accuracy_knn)+2).fit(X_train, Y_train)
cm_knn = confusion_matrix(knn.predict(X_test),Y_test)
test_score_by_model_accuracy_offset['KNN'] = accuracy_offset(cm_knn)
test_score_by_model_accuracy['KNN'] = accuracy(cm_knn)
```

---

- *4. Support Vector Classifier*

``` python
accuracy_svc = []
svc_c_list = [0.01,0.1,1.0,10.0,100.0]
g_list = [0.01,0.1,1.0,10.0,100.0]
for c in svc_c_list:
    for g in g_list:
        svc    = SVC(kernel="rbf",C=c,gamma=g).fit(X_train, Y_train)
        cm_svc = confusion_matrix(svc.predict(X_valid),Y_valid)
        accuracy_svc.append(accuracy(cm_svc))
        
svc_best = np.argmax(accuracy_svc)
svc    = SVC(kernel="rbf",C=svc_c_list[svc_best // len(svc_c_list)],
             gamma=g_list[svc_best % len(svc_c_list)]).fit(X_train, Y_train)
cm_svc = confusion_matrix(svc.predict(X_test),Y_test)
test_score_by_model_accuracy_offset['SVC'] = accuracy_offset(cm_svc)
test_score_by_model_accuracy['SVC'] = accuracy(cm_svc)
```

---

- *5. Naive Bayes Classification*

``` python
nb                              = GaussianNB().fit(X_train, Y_train)
df['N_star_hat_NB']             = np.concatenate(
        [
                nb.predict(X_train),
                nb.predict(X_valid),
                nb.predict(X_test)
        ]).round().astype(int)
df.loc[df['N_star_hat_NB']>5,'N_star_hat_NB'] = 5
df.loc[df['N_star_hat_NB']<1,'N_star_hat_NB'] = 1
cm_nb = confusion_matrix(df.iloc[1476:1623]['rating'], df.iloc[1476:1623]['N_star_hat_NB'])
test_score_by_model_accuracy_offset['Naive Bayes'] = accuracy_offset(cm_nb)
test_score_by_model_accuracy['Naive Bayes'] = accuracy(cm_nb)
```

---

- *6. Decision Tree Classifier*

``` python
accuracy_tree         = []
criterion_chosen     = ['entropy','gini']
max_depth_tree = list(range(2,11))
for i in criterion_chosen:
    for depth in max_depth_tree:
        dtree    = tree.DecisionTreeClassifier(criterion    = i, 
                                               max_depth    = depth).fit(X_train, Y_train)
        cm_tree = confusion_matrix(dtree.predict(X_valid),Y_valid)
        accuracy_tree.append(accuracy(cm_tree))

tree_best = np.argmax(accuracy_tree)
dtree    = tree.DecisionTreeClassifier(criterion= criterion_chosen[tree_best // len(max_depth_tree)], 
                                     max_depth = max_depth_tree[tree_best % len(max_depth_tree)]).fit(X_train, Y_train)
cm_tree = confusion_matrix(dtree.predict(X_test),Y_test)
test_score_by_model_accuracy_offset['Decision Tree'] = accuracy_offset(cm_tree)
test_score_by_model_accuracy['Decision Tree'] = accuracy(cm_tree)
```

---

- *7. Random Forest Classifier*

``` python
accuracy_rf         = []
parameter_combination = []
max_depth_rf = list(range(2,12))
n_estimators = [10,20,30,50,100]
max_features = [50,100,150,200,250,300,350,400,487]
for md in max_depth_rf:
    for n in n_estimators:
        for mf in max_features:
            parameter_combination.append([md,n,mf])
            rf    = RandomForestClassifier(max_depth=md, n_estimators=n, 
                                            max_features=mf).fit(X_train, Y_train)
            cm_rf = confusion_matrix(rf.predict(X_valid),Y_valid)
            accuracy_rf.append(accuracy(cm_rf))

rf_best = parameter_combination[np.argmax(accuracy_rf)]
rf = RandomForestClassifier(max_depth=rf_best[0], n_estimators=rf_best[1], 
                            max_features=rf_best[2]).fit(X_train, Y_train)
cm_rf = confusion_matrix(rf.predict(X_test),Y_test)
test_score_by_model_accuracy_offset['Random Forest'] = accuracy_offset(cm_rf)
test_score_by_model_accuracy['Random Forest'] = accuracy(cm_rf)
```

---

### **Feature importances for random forest model**

- *Python code*

```python
review_word = vectorizer.get_feature_names()
word_index = {}
for i in range(len(review_word)):
    word_index[i] = review_word[i]
    
feature_importances_rf = pd.DataFrame(rf.feature_importances_, columns=['importance']).sort_values("importance", ascending = True)
feature_importances_rf = feature_importances_rf.reset_index()
feature_importances_rf['word'] = feature_importances_rf['index'].map(word_index)
feature_importances_rf = feature_importances_rf.drop(feature_importances_rf.columns[0],axis = 1).set_index('word')
feature_importances_rf.tail(20).plot.barh()
```

---

 - *1. Buttermilk & Bourbon*
 
 ![](/images/buttermilk_rf.png)

---

 - *2. Osteria Nino*
 
 ![](/images/osteria_rf.png)

---

### **Models comparison**
 
 ![](/images/model_comparison.png)
 
- ### **1. Compare the accuracy score with and without the offset**

  - #### **The difference between accuracy scores with and without offset is very huge. There is always deviation.**



- ### **2. Compare two restaurants**

  - #### **Most of the models perform better in Buttermilk & Bourbon than in Osteria Nino. Because Buttermilk has 80% 5-star ratings.**



- ### **3. Compare across models**

  - #### **Logistic regression is the best.**
  
  - #### **Linear regression is the worst in Osteria Nino, but pretty good in Buttermilk & Bourbon.**
  
  - #### **Naïve Bayes model does very badly in all of the four situations.**
