# Marketing Analytics

- First item
- Second item
- Third item
    - Indented item
    - Indented item
- Fourth item

> #### The quarterly results look great!
>
> - Revenue was off the chart.
> - Profits were higher than ever.
>
>  *Everything* is going according to **plan**.
>> The Witch bade her clean the pots and kettles and sweep the floor and keep the fire fed with wood.

# OpenTable reviews scraping and rating analysis

## **Two restaurant for scraping**

### 1. Buttermilk & Bourbon

- *1622 reviews and 4.7 rating*

![](/images/buttermilk_overview.png)

- *Rating distribution*

![](/images/buttermilk_rating.png)

### 2. Osteria Nino

- *864 reviews and 4.2 rating*

![](/images/osteria_overview.png)

- *Rating distribution*

![](/images/osteria_rating.png)

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

## **Run different models to predict ratings in test data**

- Import tools


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

- Transfer and split data into training, validating, and test set

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

- Define functions to calculate accuracy score with/without offset
``` python
def accuracy_offset(cm):
    accuracy = (cm.diagonal(offset=-1).sum()+cm.diagonal(offset=0).sum()+cm.diagonal(offset=1).sum())/cm.sum()
    return accuracy

def accuracy(cm):
    accuracy = cm.diagonal(offset=0).sum()/cm.sum()
    return accuracy
```
