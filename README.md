# IMDB-Movie-Sentiment-Analysis
Classifying IMDb Movie Reviews.
Classifying IMDb Movie Reviews


Sentiment Analysis is a common NLP task that Data Scientists need to perform.

Sentiment Analysis is a common NLP task that Data Scientists need to perform.In this project I have created barebones of  movie review classifier in Python. 

Data Overview
For this analysis we’ll be using a dataset of 50,000 movie reviews taken from IMDb.
The data is split evenly with 25k reviews intended for training and 25k for testing our classifier. Moreover, each set has 12.5k positive and 12.5k negative reviews.

IMDb lets users rate movies on a scale from 1 to 10. To label these reviews the curator of the data labeled anything with ≤ 4 stars as negative and anything with ≥ 7 stars as positive. Reviews with 5 or 6 stars were left out.

**Step 1: Download and Combine Movie Reviews**
We will go to IMDb Reviews and click on “Large Movie Review Dataset v1.0”. Once that is complete we willll have a file called aclImdb_v1.tar.gz in our downloads folder.

Unpacking and Merging
Follow these steps or run the shell script here: Preprocessing Script

1. Move the tar file to the directory where you want this data to be stored.
2. Open a terminal window and cd to the directory that you put aclImdb_v1.tar.gz in.
3. gunzip -c aclImdb_v1.tar.gz | tar xopf -
4. cd aclImdb && mkdir movie_data
5. for split in train test; do for sentiment in pos neg; do for file in $split/$sentiment/*; do cat $file >>  movie_data/full_${split}.txt; echo >> movie_data/full_${split}.txt; done; done; done;
**Step 2: Read into Python**
For most of what we want to do in this project we’ll only need our reviews to be in a Python list. We have to make sure to point open to the directory where we put the movie data.

## Step 3: Clean and Preprocess
The raw text is pretty messy for these reviews so before we can do any analytics we need to clean things up.
**Note:** There are a lot of different and more sophisticated ways to clean text data that would likely produce better results than what I’ve done here.I generally think it’s best to get baseline predictions with the simplest possible solution before spending time doing potentially unnecessary transformations.
**Vectorization**
In order for this data to make sense to our machine learning algorithm we’ll need to convert each review to a numeric representation, which we call vectorization.

The simplest form of this is to create one very large matrix with one column for every unique word in your corpus (where the corpus is all 50k reviews in our case). Then we transform each review into one row containing 0s and 1s, where 1 means that the word in the corpus corresponding to that column appears in that review. That being said, each row of the matrix will be very sparse (mostly zeros). This process is also known as one hot encoding.

Step 4: Build Classifier
Now that we’ve transformed our dataset into a format suitable for modeling we can start building a classifier. Logistic Regression is a good baseline model for us to use for several reasons: 
(1) They’re easy to interpret, (2) linear models tend to perform well on sparse datasets like this one, and (3) they learn very fast compared to other algorithms.

To keep things simple I’m only going to worry about the hyperparameter C, which adjusts the regularization.
   
#     Accuracy for C=0.01: 0.87472
#     Accuracy for C=0.05: 0.88368
#     Accuracy for C=0.25: 0.88016
#     Accuracy for C=0.5: 0.87808
#     Accuracy for C=1: 0.87648
It looks like the value of C that gives us the highest accuracy is 0.05.

**Train Final Model**
Now that we’ve found the optimal value for C, we should train a model using the entire training set and evaluate our accuracy on the 25k test reviews.
# Final Accuracy: 0.88128
As a sanity check, let’s look at the 5 most discriminating words for both positive and negative reviews. We’ll do this by looking at the largest and smallest coefficients, respectively.
    
#     ('excellent', 0.9288812418118644)
#     ('perfect', 0.7934641227980576)
#     ('great', 0.675040909917553)
#     ('amazing', 0.6160398142631545)
#     ('superb', 0.6063967799425831)    
#     ('worst', -1.367978497228895)
#     ('waste', -1.1684451288279047)
#     ('awful', -1.0277001734353677)
#     ('poorly', -0.8748317895742782)
#     ('boring', -0.8587249740682945)
And there it is. A very simple classifier with pretty decent accuracy out of the box.
**Text Processing**
For our first iteration we did very basic text processing like removing punctuation and HTML tags and making everything lower-case. We can clean things up further by removing stop words and normalizing the text.

To make these transformations we’ll use libraries from the Natural Language Toolkit (NLTK). This is a very popular NLP library for Python.
**Removing Stop Words**
Stop words are the very common words like ‘if’, ‘but’, ‘we’, ‘he’, ‘she’, and ‘they’. We can usually remove these words without changing the semantics of a text and doing so often (but not always) improves the performance of a model. Removing these stop words becomes a lot more useful when we start using longer word sequences as model features (see n-grams below).

**Normalization**
A common next step in text preprocessing is to normalize the words in your corpus by trying to convert all of the different forms of a given word into one. Two methods that exist for this are Stemming and Lemmatization.

**Stemming**

Stemming is considered to be the more crude/brute-force approach to normalization (although this doesn’t necessarily mean that it will perform worse). There’s several algorithms, but in general they all use basic rules to chop off the ends of words.

**Lemmatization**

Lemmatization works by identifying the part-of-speech of a given word and then applying more complex rules to transform the word into its true root.

**n-grams**
Last time we used only single word features in our model, which we call 1-grams or unigrams. We can potentially add more predictive power to our model by adding two or three word sequences (bigrams or trigrams) as well. For example, if a review had the three word sequence “didn’t love movie” we would only consider these words individually with a unigram-only model and probably not capture that this is actually a negative sentiment because the word ‘love’ by itself is going to be highly correlated with a positive review.

The scikit-learn library makes this really easy to play around with. Just use the ngram_range argument with any of the ‘Vectorizer’ classes.
    
# Accuracy for C=0.01: 0.88416
# Accuracy for C=0.05: 0.892
# Accuracy for C=0.25: 0.89424
# Accuracy for C=0.5: 0.89456
# Accuracy for C=1: 0.8944

# Final Accuracy: 0.898
Getting pretty close to 90%! So, simply considering 2-word sequences in addition to single words increased our accuracy by more than 1.6 percentage points.

**Note:** There’s technically no limit on the size that n can be for your model, but there are several things to consider. First, increasing the number of grams will not necessarily give you better performance. Second, the size of your matrix grows exponentially as you increment n, so if you have a large corpus that is comprised of large documents your model may take a very long time to train.
## Representations

While this simple approach can work very well, there are ways that we can encode more information into the vector.

**Word Counts**
Instead of simply noting whether a word appears in the review or not, we can include the number of times a given word appears. This can give our sentiment classifier a lot more predictive power. For example, if a movie reviewer says ‘amazing’ or ‘terrible’ multiple times in a review it is considerably more probable that the review is positive or negative, respectively.

# Accuracy for C=0.01: 0.87456
# Accuracy for C=0.05: 0.88016
# Accuracy for C=0.25: 0.87936
# Accuracy for C=0.5: 0.87936
# Accuracy for C=1: 0.87696
# Final Accuracy: 0.88184
**TF-IDF**
Another common way to represent each document in a corpus is to use the tf-idf statistic (term frequency-inverse document frequency) for each word, which is a weighting factor that we can use in place of binary or word count representations.

There are several ways to do tf-idf transformation but in a nutshell, tf-idf aims to represent the number of times a given word appears in a document (a movie review in our case) relative to the number of documents in the corpus that the word appears in — where words that appear in many documents have a value closer to zero and words that appear in less documents have values closer to 1.

**Note:** Now that we’ve gone over n-grams, when I refer to ‘words’ I really mean any n-gram (sequence of words) if the model is using an n greater than one.



# Accuracy for C=0.01: 0.79632
# Accuracy for C=0.05: 0.83168
# Accuracy for C=0.25: 0.86768
# Accuracy for C=0.5: 0.8736
# Accuracy for C=1: 0.88432

# Final Accuracy: 0.882
## Algorithms
So far we’ve chosen to represent each review as a very sparse vector (lots of zeros!) with a slot for every unique n-gram in the corpus (minus n-grams that appear too often or not often enough). Linear classifiers typically perform better than other algorithms on data that is represented in this way.

**Support Vector Machines (SVM)**
Recall that linear classifiers tend to work well on very sparse datasets (like the one we have). Another algorithm that can produce great results with a quick training time are Support Vector Machines with a linear kernel.
    
# Accuracy for C=0.01: 0.89104
# Accuracy for C=0.05: 0.88736
# Accuracy for C=0.25: 0.8856
# Accuracy for C=0.5: 0.88608
# Accuracy for C=1: 0.88592

# Final Accuracy: 0.89174
## Final Model

I found that removing a small set of stop words along with an n-gram range from 1 to 3 and a linear support vector classifier gave me the best results.
    
# Accuracy for C=0.001: 0.88784
# Accuracy for C=0.005: 0.89456
# Accuracy for C=0.01: 0.89376
# Accuracy for C=0.05: 0.89264
# Accuracy for C=0.1: 0.8928

# Final Accuracy: 0.90064

We broke the 90% mark!
**Summary**
We’ve gone over several options for transforming text that can improve the accuracy of an NLP model. Which combination of these techniques will yield the best results will depend on the task, data representation, and algorithms you choose. It’s always a good idea to try out many different combinations to see what works.
