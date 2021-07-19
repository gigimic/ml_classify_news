
Machine Learning model for classification
-----------------------------------------

A model to classify the news as real or fake.

Using sklearn, we build a TfidfVectorizer on the dataset. Then, we initialize a PassiveAggressive Classifier and fit the model. Passive Aggressive algorithms are online learning algorithms. Such an algorithm remains passive for a correct classification outcome, and turns aggressive in the event of a miscalculation, updating and adjusting. Unlike most other algorithms, it does not converge. Its purpose is to make updates that correct the loss, causing very little change in the norm of the weight vector.

The dataset has a shape of 7796×4. The first column identifies the news, the second and third are the title and text, and the fourth column has labels denoting whether the news is REAL or FAKE. 

Text is cleaned. stop words are removed. tfidf vectorizer is used with a maximum document frequency 0.7 (term with a higher document frequency are discarded). tfidf vectorizer turns a collection of raw documents into a matrix of TF-IDF features.

Then fit and transform the vectorizer on the train set, and transform the vectorizer on the test set.


https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/