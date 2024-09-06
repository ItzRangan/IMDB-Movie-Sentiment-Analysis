# IMDB-Movie-Sentiment-Analysis
Classifying IMDb Movie Reviews.
Classifying IMDb Movie Reviews


Sentiment Analysis is a common NLP task that Data Scientists need to perform.

Sentiment Analysis is a common NLP task that Data Scientists need to perform. This is a straightforward guide to creating a barebones movie review classifier in Python. Future parts of this series will focus on improving the classifier.
Data Overview
For this analysis we’ll be using a dataset of 50,000 movie reviews taken from IMDb.
The data is split evenly with 25k reviews intended for training and 25k for testing your classifier. Moreover, each set has 12.5k positive and 12.5k negative reviews.

IMDb lets users rate movies on a scale from 1 to 10. To label these reviews the curator of the data labeled anything with ≤ 4 stars as negative and anything with ≥ 7 stars as positive. Reviews with 5 or 6 stars were left out.
## Step 1: Download and Combine Movie Reviews
If you haven’t yet, go to IMDb Reviews and click on “Large Movie Review Dataset v1.0”. Once that is complete you’ll have a file called aclImdb_v1.tar.gz in your downloads folder.

Shortcut: If you want to get straight to the data analysis and/or aren’t super comfortable with the terminal, I’ve put a tar file of the final directory that this step creates here: Merged Movie Data. Double clicking this file should be sufficient to unpack it (at least on a Mac), otherwise gunzip -c movie_data.tar.gz | tar xopf — in a terminal will do it.
## Unpacking and Merging
Follow these steps or run the shell script here: Preprocessing Script

1. Move the tar file to the directory where you want this data to be stored.
2. Open a terminal window and cd to the directory that you put aclImdb_v1.tar.gz in.
3. gunzip -c aclImdb_v1.tar.gz | tar xopf -
4. cd aclImdb && mkdir movie_data
5. for split in train test; do for sentiment in pos neg; do for file in $split/$sentiment/*; do cat $file >>  movie_data/full_${split}.txt; echo >> movie_data/full_${split}.txt; done; done; done;
## Step 2: Read into Python
For most of what we want to do in this walkthrough we’ll only need our reviews to be in a Python list. Make sure to point open to the directory where you put the movie data.
reviews_train = []
for line in open('./data/full_train.txt', 'r', encoding="utf8"):
    reviews_train.append(line.strip())
    
reviews_test = []
for line in open('./data/full_test.txt', 'r', encoding="utf8"):
    reviews_test.append(line.strip())
## Step 3: Clean and Preprocess
The raw text is pretty messy for these reviews so before we can do any analytics we need to clean things up. Here’s one example:
"This isn't the comedic Robin Williams, nor is it the quirky/insane Robin Williams of recent thriller fame. This is a hybrid of the classic drama without over-dramatization, mixed with Robin's new love of the thriller. But this isn't a thriller, per se. This is more a mystery/suspense vehicle through which Williams attempts to locate a sick boy and his keeper.<br /><br />Also starring Sandra Oh and Rory Culkin, this Suspense Drama plays pretty much like a news report, until William's character gets close to achieving his goal.<br /><br />I must say that I was highly entertained, though this movie fails to teach, guide, inspect, or amuse. It felt more like I was watching a guy (Williams), as he was actually performing the actions, from a third person perspective. In other words, it felt real, and I was able to subscribe to the premise of the story.<br /><br />All in all, it's worth a watch, though it's definitely not Friday/Saturday night fare.<br /><br />It rates a 7.7/10 from...<br /><br />the Fiend :."
Note: Understanding and being able to use regular expressions is a prerequisite for doing any Natural Language Processing task. 
import re

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews

reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)
And this is what the same review looks like now:
"this isnt the comedic robin williams nor is it the quirky insane robin williams of recent thriller fame this is a hybrid of the classic drama without over dramatization mixed with robins new love of the thriller but this isnt a thriller per se this is more a mystery suspense vehicle through which williams attempts to locate a sick boy and his keeper also starring sandra oh and rory culkin this suspense drama plays pretty much like a news report until williams character gets close to achieving his goal i must say that i was highly entertained though this movie fails to teach guide inspect or amuse it felt more like i was watching a guy williams as he was actually performing the actions from a third person perspective in other words it felt real and i was able to subscribe to the premise of the story all in all its worth a watch though its definitely not friday saturday night fare it rates a   from the fiend"
**Note:** There are a lot of different and more sophisticated ways to clean text data that would likely produce better results than what I’ve done here. I wanted part 1 of this tutorial to be as simple as possible. Also, I generally think it’s best to get baseline predictions with the simplest possible solution before spending time doing potentially unnecessary transformations.
**Vectorization**
In order for this data to make sense to our machine learning algorithm we’ll need to convert each review to a numeric representation, which we call vectorization.

The simplest form of this is to create one very large matrix with one column for every unique word in your corpus (where the corpus is all 50k reviews in our case). Then we transform each review into one row containing 0s and 1s, where 1 means that the word in the corpus corresponding to that column appears in that review. That being said, each row of the matrix will be very sparse (mostly zeros). This process is also known as one hot encoding.
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary=True)
cv.fit(reviews_train_clean)
X = cv.transform(reviews_train_clean)
X_test = cv.transform(reviews_test_clean)
## Step 4: Build Classifier
Now that we’ve transformed our dataset into a format suitable for modeling we can start building a classifier. Logistic Regression is a good baseline model for us to use for several reasons: (1) They’re easy to interpret, (2) linear models tend to perform well on sparse datasets like this one, and (3) they learn very fast compared to other algorithms.

To keep things simple I’m only going to worry about the hyperparameter C, which adjusts the regularization.
**Note:** The targets/labels we use will be the same for training and testing because both datasets are structured the same, where the first 12.5k are positive and the last 12.5k are negative.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = [1 if i < 12500 else 0 for i in range(25000)]

X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = 0.75
)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=c, solver='lbfgs', max_iter=1000)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_val, lr.predict(X_val))))
    
#     Accuracy for C=0.01: 0.87472
#     Accuracy for C=0.05: 0.88368
#     Accuracy for C=0.25: 0.88016
#     Accuracy for C=0.5: 0.87808
#     Accuracy for C=1: 0.87648
It looks like the value of C that gives us the highest accuracy is 0.05.
## Train Final Model
Now that we’ve found the optimal value for C, we should train a model using the entire training set and evaluate our accuracy on the 25k test reviews.
final_model = LogisticRegression(C=0.05, solver='lbfgs', max_iter=1000)
final_model.fit(X, target)
print ("Final Accuracy: %s" 
       % accuracy_score(target, final_model.predict(X_test)))
# Final Accuracy: 0.88128
As a sanity check, let’s look at the 5 most discriminating words for both positive and negative reviews. We’ll do this by looking at the largest and smallest coefficients, respectively.
feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names_out(), final_model.coef_[0]
    )
}
for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:5]:
    print (best_positive)
    
#     ('excellent', 0.9288812418118644)
#     ('perfect', 0.7934641227980576)
#     ('great', 0.675040909917553)
#     ('amazing', 0.6160398142631545)
#     ('superb', 0.6063967799425831)
    
for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:5]:
    print (best_negative)
    
#     ('worst', -1.367978497228895)
#     ('waste', -1.1684451288279047)
#     ('awful', -1.0277001734353677)
#     ('poorly', -0.8748317895742782)
#     ('boring', -0.8587249740682945)
And there it is. A very simple classifier with pretty decent accuracy out of the box.
## Text Processing
For our first iteration we did very basic text processing like removing punctuation and HTML tags and making everything lower-case. We can clean things up further by removing stop words and normalizing the text.

To make these transformations we’ll use libraries from the Natural Language Toolkit (NLTK). This is a very popular NLP library for Python.
**Removing Stop Words**
Stop words are the very common words like ‘if’, ‘but’, ‘we’, ‘he’, ‘she’, and ‘they’. We can usually remove these words without changing the semantics of a text and doing so often (but not always) improves the performance of a model. Removing these stop words becomes a lot more useful when we start using longer word sequences as model features (see n-grams below).
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords

english_stop_words = stopwords.words('english')
def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in english_stop_words])
        )
    return removed_stop_words

no_stop_words = remove_stop_words(reviews_train_clean)
**Before**

"bromwell high is a cartoon comedy it ran at the same time as some other programs about school life such as teachers my years in the teaching profession lead me to believe that bromwell high’s satire is much closer to reality than is teachers the scramble to survive financially the insightful students who can see right through their pathetic teachers’ pomp the pettiness of the whole situation all remind me of the schools i knew and their students when i saw the episode in which a student repeatedly tried to burn down the school i immediately recalled at high a classic line inspector i’m here to sack one of your teachers student welcome to bromwell high i expect that many adults of my age think that bromwell high is far fetched what a pity that it isn’t"
**After**

"bromwell high cartoon comedy ran time programs school life teachers years teaching profession lead believe bromwell high's satire much closer reality teachers scramble survive financially insightful students see right pathetic teachers' pomp pettiness whole situation remind schools knew students saw episode student repeatedly tried burn school immediately recalled high classic line inspector i'm sack one teachers student welcome bromwell high expect many adults age think bromwell high far fetched pity"
**Note:** In practice, an easier way to remove stop words is to just use the stop_words argument with any of scikit-learn’s ‘Vectorizer’ classes. If you want to use NLTK’s full list of stop words you can do stop_words='english’. In practice I’ve found that using NLTK’s list actually decreases my performance because its too expansive, so I usually supply my own list of words. For example, stop_words=['in','of','at','a','the'] .
**Normalization**
A common next step in text preprocessing is to normalize the words in your corpus by trying to convert all of the different forms of a given word into one. Two methods that exist for this are Stemming and Lemmatization.

**Stemming**

Stemming is considered to be the more crude/brute-force approach to normalization (although this doesn’t necessarily mean that it will perform worse). There’s several algorithms, but in general they all use basic rules to chop off the ends of words.

NLTK has several stemming algorithm implementations.
def get_stemmed_text(corpus):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

stemmed_reviews = get_stemmed_text(reviews_train_clean)
**Lemmatization**

Lemmatization works by identifying the part-of-speech of a given word and then applying more complex rules to transform the word into its true root.
def get_lemmatized_text(corpus):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

lemmatized_reviews = get_lemmatized_text(reviews_train_clean)

**Results**

**No Normalization**

"this is not the typical mel brooks film it was much less slapstick than most of his movies and actually had a plot that was followable leslie ann warren made the movie she is such a fantastic under rated actress there were some moments that could have been fleshed out a bit more and some scenes that could probably have been cut to make the room to do so but all in all this is worth the price to rent and see it the acting was good overall brooks himself did a good job without his characteristic speaking to directly to the audience again warren was the best actor in the movie but fume and sailor both played their parts well"
**Stemmed**

"thi is not the typic mel brook film it wa much less slapstick than most of hi movi and actual had a plot that wa follow lesli ann warren made the movi she is such a fantast under rate actress there were some moment that could have been flesh out a bit more and some scene that could probabl have been cut to make the room to do so but all in all thi is worth the price to rent and see it the act wa good overal brook himself did a good job without hi characterist speak to directli to the audienc again warren wa the best actor in the movi but fume and sailor both play their part well"
**Lemmatized**

"this is not the typical mel brook film it wa much le slapstick than most of his movie and actually had a plot that wa followable leslie ann warren made the movie she is such a fantastic under rated actress there were some moment that could have been fleshed out a bit more and some scene that could probably have been cut to make the room to do so but all in all this is worth the price to rent and see it the acting wa good overall brook himself did a good job without his characteristic speaking to directly to the audience again warren wa the best actor in the movie but fume and sailor both played their part well"
**n-grams**
Last time we used only single word features in our model, which we call 1-grams or unigrams. We can potentially add more predictive power to our model by adding two or three word sequences (bigrams or trigrams) as well. For example, if a review had the three word sequence “didn’t love movie” we would only consider these words individually with a unigram-only model and probably not capture that this is actually a negative sentiment because the word ‘love’ by itself is going to be highly correlated with a positive review.

The scikit-learn library makes this really easy to play around with. Just use the ngram_range argument with any of the ‘Vectorizer’ classes.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(reviews_train_clean)
X = ngram_vectorizer.transform(reviews_train_clean)
X_test = ngram_vectorizer.transform(reviews_test_clean)

X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = 0.75
)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=c, solver='lbfgs', max_iter=1000)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_val, lr.predict(X_val))))
    
# Accuracy for C=0.01: 0.88416
# Accuracy for C=0.05: 0.892
# Accuracy for C=0.25: 0.89424
# Accuracy for C=0.5: 0.89456
# Accuracy for C=1: 0.8944
    
final_ngram = LogisticRegression(C=0.5, solver='lbfgs', max_iter=1000)
final_ngram.fit(X, target)
print ("Final Accuracy: %s" 
       % accuracy_score(target, final_ngram.predict(X_test)))

# Final Accuracy: 0.898
Getting pretty close to 90%! So, simply considering 2-word sequences in addition to single words increased our accuracy by more than 1.6 percentage points.

**Note:** There’s technically no limit on the size that n can be for your model, but there are several things to consider. First, increasing the number of grams will not necessarily give you better performance. Second, the size of your matrix grows exponentially as you increment n, so if you have a large corpus that is comprised of large documents your model may take a very long time to train.
## Representations

While this simple approach can work very well, there are ways that we can encode more information into the vector.

**Word Counts**
Instead of simply noting whether a word appears in the review or not, we can include the number of times a given word appears. This can give our sentiment classifier a lot more predictive power. For example, if a movie reviewer says ‘amazing’ or ‘terrible’ multiple times in a review it is considerably more probable that the review is positive or negative, respectively.


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

wc_vectorizer = CountVectorizer(binary=False)
wc_vectorizer.fit(reviews_train_clean)
X = wc_vectorizer.transform(reviews_train_clean)
X_test = wc_vectorizer.transform(reviews_test_clean)

X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = 0.75, 
)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=c, solver='lbfgs', max_iter=1000)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_val, lr.predict(X_val))))
    
# Accuracy for C=0.01: 0.87456
# Accuracy for C=0.05: 0.88016
# Accuracy for C=0.25: 0.87936
# Accuracy for C=0.5: 0.87936
# Accuracy for C=1: 0.87696
    
final_wc = LogisticRegression(C=0.05, solver='lbfgs', max_iter=1000)
final_wc.fit(X, target)
print ("Final Accuracy: %s" 
       % accuracy_score(target, final_wc.predict(X_test)))

# Final Accuracy: 0.88184
**TF-IDF**
Another common way to represent each document in a corpus is to use the tf-idf statistic (term frequency-inverse document frequency) for each word, which is a weighting factor that we can use in place of binary or word count representations.

There are several ways to do tf-idf transformation but in a nutshell, tf-idf aims to represent the number of times a given word appears in a document (a movie review in our case) relative to the number of documents in the corpus that the word appears in — where words that appear in many documents have a value closer to zero and words that appear in less documents have values closer to 1.

**Note:** Now that we’ve gone over n-grams, when I refer to ‘words’ I really mean any n-gram (sequence of words) if the model is using an n greater than one.


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(reviews_train_clean)
X = tfidf_vectorizer.transform(reviews_train_clean)
X_test = tfidf_vectorizer.transform(reviews_test_clean)

X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = 0.75
)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=c, solver='lbfgs', max_iter=1000)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_val, lr.predict(X_val))))

# Accuracy for C=0.01: 0.79632
# Accuracy for C=0.05: 0.83168
# Accuracy for C=0.25: 0.86768
# Accuracy for C=0.5: 0.8736
# Accuracy for C=1: 0.88432
    
final_tfidf = LogisticRegression(C=1, solver='lbfgs', max_iter=1000)
final_tfidf.fit(X, target)
print ("Final Accuracy: %s" 
       % accuracy_score(target, final_tfidf.predict(X_test)))

# Final Accuracy: 0.882
## Algorithms
So far we’ve chosen to represent each review as a very sparse vector (lots of zeros!) with a slot for every unique n-gram in the corpus (minus n-grams that appear too often or not often enough). Linear classifiers typically perform better than other algorithms on data that is represented in this way.

**Support Vector Machines (SVM)**
Recall that linear classifiers tend to work well on very sparse datasets (like the one we have). Another algorithm that can produce great results with a quick training time are Support Vector Machines with a linear kernel.

Here’s an example with an n-gram range from 1 to 2:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(reviews_train_clean)
X = ngram_vectorizer.transform(reviews_train_clean)
X_test = ngram_vectorizer.transform(reviews_test_clean)

X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = 0.75
)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    svm = LinearSVC(C=c, max_iter=1500)
    svm.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_val, svm.predict(X_val))))
    
# Accuracy for C=0.01: 0.89104
# Accuracy for C=0.05: 0.88736
# Accuracy for C=0.25: 0.8856
# Accuracy for C=0.5: 0.88608
# Accuracy for C=1: 0.88592
    
final_svm_ngram = LinearSVC(C=0.01, max_iter=1500)
final_svm_ngram.fit(X, target)
print ("Final Accuracy: %s" 
       % accuracy_score(target, final_svm_ngram.predict(X_test)))

# Final Accuracy: 0.8974
## Final Model

The goal of this post was to give you a toolbox of things to try and mix together when trying to find the right model + data transformation for your project. I found that removing a small set of stop words along with an n-gram range from 1 to 3 and a linear support vector classifier gave me the best results.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC


stop_words = ['in', 'of', 'at', 'a', 'the']
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)
ngram_vectorizer.fit(reviews_train_clean)
X = ngram_vectorizer.transform(reviews_train_clean)
X_test = ngram_vectorizer.transform(reviews_test_clean)

X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = 0.75
)

for c in [0.001, 0.005, 0.01, 0.05, 0.1]:
    
    svm = LinearSVC(C=c, max_iter=1000)
    svm.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_val, svm.predict(X_val))))
    
# Accuracy for C=0.001: 0.88784
# Accuracy for C=0.005: 0.89456
# Accuracy for C=0.01: 0.89376
# Accuracy for C=0.05: 0.89264
# Accuracy for C=0.1: 0.8928
    
final = LinearSVC(C=0.01, max_iter=1000)
final.fit(X, target)
print ("Final Accuracy: %s" 
       % accuracy_score(target, final.predict(X_test)))

# Final Accuracy: 0.90064

We broke the 90% mark!
**Summary**
We’ve gone over several options for transforming text that can improve the accuracy of an NLP model. Which combination of these techniques will yield the best results will depend on the task, data representation, and algorithms you choose. It’s always a good idea to try out many different combinations to see what works.
