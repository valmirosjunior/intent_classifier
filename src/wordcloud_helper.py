import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud

from .distribution_data import *

nltk.download('stopwords')
STOPWORDS = nltk.corpus.stopwords.words('portuguese')


def plot_word_cloud(words, name_file=None):
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color="black",
                          width=1600, height=800).generate(words)

    if name_file:
        wordcloud.to_file("{}.png".format(name_file))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_axis_off()

    plt.imshow(wordcloud)
    plt.show(block=False)


def print_wordcloud_of_sentences(sentences):
    if not sentences.empty:
        plot_word_cloud(" ".join(s for s in sentences))


def print_word_clouds_of_each_label(data_helper, num_sentences=10):
    labels = data_helper.get_labels()
    df = data_helper.df

    for label in labels:
        sentences = df[df['label'] == label]['txt']
        print(f"Total of sentences is: {len(sentences)} for the label: {label}", )
        print('\n=======================================================\n')
        print(sentences[:num_sentences].to_numpy())

        print_wordcloud_of_sentences(sentences)

        print('\n\n========================================================\n\n')


def analyze_cluster_info(df, cluster_label, num_sentences=10):
    cluster_data = df[df['label'] == cluster_label].sort_values('distance')[['txt', 'distance']]

    distribution_data = get_distribution_data(cluster_data['distance'])

    print(f'This cluster has {len(cluster_data)}')
    print(distribution_data)

    best_sentences = cluster_data[cluster_data['distance'] <= distribution_data.upper_bound][['txt', 'distance']]
    outliers_sentences = cluster_data[cluster_data['distance'] > distribution_data.upper_bound][['txt', 'distance']]

    print_wordcloud_of_sentences(best_sentences['txt'])
    print('Sentenças mais poróximas ao centroide:')
    print(best_sentences[:num_sentences].to_numpy())

    print('\n\n***************************\n')
    print('Sentenças mais distantes ao centroide:')
    print(best_sentences[-num_sentences:].to_numpy())

    print('\n-------------------------------')

    if len(outliers_sentences) > 0:
        print_wordcloud_of_sentences(outliers_sentences['txt'])
    print(f'Amount of outliers: {len(outliers_sentences)}')
    print(outliers_sentences[:num_sentences].to_numpy())

    return best_sentences, outliers_sentences
