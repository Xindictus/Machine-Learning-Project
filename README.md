# Machine-Learning-Project

The purpose of this project is to get familiar with classification and clustering techniques following the key steps:
  1. Data selection
  2. Preprocessing
  3. Transformation
  4. Data Mining
  5. Evaluation

To run any of the `.py`, first unzip the `train_set.7z`. The `train_set.csv` is a collection of news articles in the form of 
**tab-separated CSV**. It contains the fields:
* Id: A unique article number
* Title: The article's title
* Content: The article's body
* Category: The article's category

There are 5 categories:
* Politics
* Film
* Football
* Business
* Technology

# Requirements

Before running the `.py`, you will also need to install certain Python libraries:
* amueller/word_cloud
* numpy
* dateutil
* pytz
* pyparsing
* cycler
* setuptools
* matplotlib
* pandas
* scikit-learn

# WordCloud

Running the `wordCloud.py` will create a WordCloud for each category based on the articles from the `train_set.csv`, saving them in folder named ***WordCloud Export***.
- - -

**Steps taken to increase the WordCloud's quality:**
  1. In the examples images that will follow, we will notice that just taking into account the body of the article is not enough to get the kind of information that will depict the most important stories for each category. As such, I decided to include the title of the article along with the its body in the preprocessing step. 
  1. Apart from the `stopwords` provided from the `word_cloud` library, I have included a .xls of extra `stopwords`, consisting of the most usual english verbs. This will provide us with a better quality WordCloud, since nouns provide us with a better view of what's the most important topics in the articles.
  3. The usual verbs are not included in the WordCloud now, but we can improve it even more, giving more weight to the title. So I try to add the title 20 times this time to the preprocessing
  4. The final improvement has included the title up to 50 times.
  
Let's see an example using the **Football** category of course. :) 
  
WordCloud with default settings |  WordCloud after adding title to preprocessing
:-------------------------:|:-------------------------:
![WordCloud with default settings](https://1ktmhg.by3302.livefilestore.com/y3m341KMmKM9_HwDouyi4pmbzxeBBGK9q9i7fkHksGpxssWof9dkmRDuBsw4omqPsCfPrU_UKHL_MX22ZAWHiwIU5OrhlQbaM4-YmTzbOReS39Y75hsuJZK5GtXTdl6g_1WaZ0JiycjAzbXyfr4rv3nk3K07bOQ2Y91ggNvBTFzTsU?width=418&height=209&cropmode=none)  | ![WordCloud with title](https://1ks35a.by3302.livefilestore.com/y3mNkT2wwuill2LkJxz5qS3gYJ4fy73SRgfsnBSxORxIzJqrMHjG6lci73RnJxacWDugOoVJtTJsR8eMjwx8Fg61rn1ASxNunZl_9ZpfELutlPkyNmFVz6bf4Vc_Mq6C3NW8lbj0ZqhPs_6zvqlbR2YdU46tR0CDr_AZEzD5U7H3Zs?width=418&height=209&cropmode=none)

WordCloud with extra `stopwords` |  WordCloud with title included 20 times
:-------------------------:|:-------------------------:
![WordCloud with extra stopwords](https://1kucbw.by3302.livefilestore.com/y3m-pJoov0HJI6F6f9j16o7pKm12GUYpGcRQxQmYgbjZdvX9YnIMNd7oKz6iIdkkgWKf2dxsevTV03g4JPAklPTy-EM-f9JI6T-Pq9a6u3g2vZrUoF13Ag0XKHOXRTwcaE1HBH1ClEKLZpbyHPDdNUvUAC3FvzfypXn_XjnminovqM?width=418&height=209&cropmode=none)  | ![WordCloud with title x 20](https://1kttzq.by3302.livefilestore.com/y3meKejzTeMm8vyut0VRmfBeWtXXIbquoy4nncZxHylPqlF45_IkhcMeYcMjtBe7hsI3roJon5MFzfTPUkS7MepkbbQoDqb9iCbkJvUQMzLwp-Y4b3T6ku52m7q58r5Zgx7xyHjyPipYPg-5CjA7QJdoY06Bi5Nvb6nMw8lNLb0Jmc?width=418&height=209&cropmode=none)

Final WordCloud with title included 50 times |
:-------------------------:|
![WordCloud with title x 50](https://1kvswa.by3302.livefilestore.com/y3mGM3uILa8Dh0l2xUgyO9cVATDXn4R7apMF6VwkRYyYgX2sa4bBERUCszNM7o8ebw196JkYHbsposE0ZRVMemqVP3HjhmOk2lXTEabENKRNTLrgKwVChdE04xm-iOc6Iqyk_G3So-x0XIHbNJDVnICX3anoxhjuNvnVFcMvcupLy8?width=418&height=209&cropmode=none) |

Although there is not much of a difference between adding the title 20 and 50 times, there is a significant improvement compared to the first WordCloud we created. Teams' names have started to appear and less important words such as player are depicted with a much smaller font, like they should.

# Clustering

# Classification
 
