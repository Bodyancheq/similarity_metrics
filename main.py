import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from text_news import text as news_text
from text_science import text as science_text
from text_sport import text as sport_text

with open("news.csv", "r", encoding='utf-8') as newsfile:
    news_keywords = set()
    newsfile.__next__()
    for line in newsfile:
        words = line.replace('"', '').split()
        news_keywords.update(words)

with open("science.csv", "r", encoding='utf-8') as sciencefile:
    science_keywords = set()
    sciencefile.__next__()
    for line in sciencefile:
        words = line.replace('"', '').split()
        science_keywords.update(words)

with open("shopping.csv", "r", encoding='utf-8') as shoppingfile:
    shopping_keywords = set()
    shoppingfile.__next__()
    for line in shoppingfile:
        words = line.replace('"', '').split()
        shopping_keywords.update(words)

with open("sport.csv", "r", encoding='utf-8') as sportfile:
    sport_keywords = set()
    sportfile.__next__()
    for line in sportfile:
        words = line.replace('"', '').split()
        sport_keywords.update(words)

re_pattern = re.compile(r"[\w’]+")
list_news_text = list(re_pattern.findall(news_text))
# list_science_text = list(re_pattern.findall(science_text))
# list_sport_text = list(re_pattern.findall(sport_text))


def jaccard_index(text: list, keywords: set):
    text_keywords = set(text)
    intersection = len(text_keywords.intersection(keywords))
    union = len(text_keywords.union(keywords))
    return intersection / union


def cosine_similarity_impl(text: str, keywords: str):
    """
    CountVectorizer transforms two strings to vectors, where i'th position corresponds with the
    number of times that i'th word occurred in text

    Then cosine similarity is counted between two vectors
    """
    count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform([text, keywords])
    return cosine_similarity(sparse_matrix, sparse_matrix)


news_news = jaccard_index(list_news_text, news_keywords)
news_sport = jaccard_index(list_news_text, sport_keywords)
news_science = jaccard_index(list_news_text, science_keywords)
news_shopping = jaccard_index(list_news_text, shopping_keywords)
print(f"Jaccard index of news text with news keywords:     {news_news}")
print(f"Jaccard index of news text with sport keywords:    {news_sport}")
print(f"Jaccard index of news text with science keywords:  {news_science}")
print(f"Jaccard index of news text with shopping keywords: {news_shopping}\n")

news_news = cosine_similarity_impl(news_text, " ".join(list(news_keywords)))[0][1]
news_sport = cosine_similarity_impl(news_text, " ".join(list(sport_keywords)))[0][1]
news_science = cosine_similarity_impl(news_text, " ".join(list(science_keywords)))[0][1]
news_shopping = cosine_similarity_impl(news_text, " ".join(list(shopping_keywords)))[0][1]
print(f"Cosine metric of news text with news keywords:    {news_news}")
print(f"Cosine metric of news text with sport keywords:   {news_sport}")
print(f"Cosine metric of news text with science keywords: {news_science}")
print(f"Cosine metric of news text with shoping keywords: {news_shopping}")
