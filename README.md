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
python

```

###### Heading level 6
![Tux, the Linux mascot](/assets/images/tux.png)
1.  Open the file.
2.  Find the following code block on line 21:



3.  Update the title to match the name of your website.


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
