import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image    
from wordcloud import WordCloud, STOPWORDS


def count_plot(tweet):
    plt.figure(figsize=(8,4))
    sns.countplot(x='target', data=tweet,palette='viridis_r')
    plt.title('General or Disaster Tweets',fontsize = 20)
    plt.xlabel('class',fontsize = 15)
    plt.ylabel('count',fontsize = 15)
    plt.show()

def hist_plot(tweet , label ):
    tweet['char_counts'] = tweet['text'].str.len()
    tweet['word_counts'] = tweet['text'].str.split().str.len()

    if label == 'char' :
        plt.figure(figsize=(8,4))
        sns.histplot(tweet['char_counts'])
        plt.title('Character Count Distribution',fontsize = 20)
        plt.xlabel('Number of Characters',fontsize = 15)
        plt.ylabel('Count',fontsize = 15)
        plt.show()

        plt.figure(figsize=(8,4))
        sns.kdeplot(tweet[tweet['target']==1]['char_counts'], shade=True, color='blue')
        sns.kdeplot(tweet[tweet['target']==0]['char_counts'], shade=True, color='green')
        plt.title('Disaster vs. General Tweets',fontsize = 20)
        plt.xlabel('Number of Characters',fontsize = 15)
        plt.ylabel('Count',fontsize = 15)
        plt.legend(labels= ["Disaster" , "General"], loc = "upper left")
        plt.show()

    if label == 'word' :
        plt.figure(figsize=(8,4))
        sns.histplot(tweet['word_counts'])
        plt.title('Word Count Distribution',fontsize = 20)
        plt.xlabel('Number of Words',fontsize = 15)
        plt.ylabel('Count',fontsize = 15)
        plt.show()

        plt.figure(figsize=(8,4))
        sns.kdeplot(tweet[tweet['target']==1]['word_counts'], shade=True, color='blue')
        sns.kdeplot(tweet[tweet['target']==0]['word_counts'], shade=True, color='green')
        plt.title('Disaster vs. General Tweets',fontsize = 20)
        plt.xlabel('Number of Words',fontsize = 15)
        plt.ylabel('Count',fontsize = 15)
        plt.legend(labels= ["Disaster" , "General"], loc = "upper left")
        plt.show()

def plot_cost(cost_list):
    """To plot the cost vs iteration plot """
    plt.title( "cost vs iteration")
    plt.plot( cost_list , 'blue')
    plt.show()

    
def plot_output(file_path):
        
    with open (file_path) as f:
        lines = f.readlines()
    
    cost_list = []
    for l in lines:
        l = float(l[:-2])
        cost_list.append(l)
    
    plt.title( "cost vs iteration")
    plt.plot( cost_list , 'blue')
    plt.show()
    

def plot_cloud(wordcloud,title):
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud) 
    plt.title(title ,fontsize = 20)
    plt.axis("off")



def word_cloud(keyAndListOfWords,title):
    from wordcloud import WordCloud, STOPWORDS
    newStopWords = ['may','would','also','th','one','two','us','to', 'due','new','via','many','much','however','since','although','often','m','s','ll','ve','tweet','tweeter','blog','amp']
    STOPWORDS = STOPWORDS.union(newStopWords )
    list0fWords = keyAndListOfWords.flatMap(lambda x: (x[2])).collect()
    string = pd.Series(list0fWords).str.cat(sep=' ')
    mask = np.array(Image.open('./tweet.png'))
    wordcloud = WordCloud(width = 800, height = 800, random_state=1, background_color='Navy', colormap='Blues', collocations=False, stopwords = STOPWORDS, mask=mask).generate(string)
    plot_cloud(wordcloud,title)
    

    """
    supported colors are 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'
    
    """
    
def read_output(file_path):
        
    with open (file_path) as f:
        lines = f.readlines()
    cost_list = []
    for l in lines:
        l = float(l[:-2])
        cost_list.append(l)
    return cost_list

def plot_cost_optimizer(path, bold_driver = 'without', name='plot.png'):
    cost_array_list = []
    optimizers = ['SGD','Momentum','Nesterov','Adam', 'Adagrad' ,'RMSprop' ]
    for optimizer in optimizers:

        cost_array_list.append(read_output(f'{path}{optimizer}/part-00000'))

    iter_ = range(0,200)
    plt.figure(figsize=(10,8))
    plt.plot(iter_ ,cost_array_list[0] , label = optimizers[0], linewidth=2)
    plt.plot(iter_ ,cost_array_list[1] , label = optimizers[1], linewidth=2)
    # plt.plot(iter_ ,cost_array_list[2] , label = optimizers[2], linewidth=2)
    plt.plot(iter_ ,cost_array_list[3] , label = optimizers[3], linewidth=2)
    plt.plot(iter_ ,cost_array_list[4] , label = optimizers[4], linewidth=2)
    plt.plot(iter_ ,cost_array_list[5] , label = optimizers[5], linewidth=2)
    plt.legend(loc = 'best')
    plt.title(f'Cost Comparison among Gradient Descent Optimizers {bold_driver} bold driver',fontsize = 15)
    plt.xlabel('Iteration',fontsize = 15)
    plt.ylabel('Cost',fontsize = 15)
    plt.savefig(name,bbox_inches = 'tight')
    plt.show()
    
    
    
def plot_cost_SGD_Adam( cost_array1 , cost_array2 , name='plot.png'):
    iter_ = range(0,300)
    plt.figure(figsize=(10,8))
    plt.plot(iter_ ,cost_array1 , label = 'SGD', linewidth=2)
    plt.plot(iter_ ,cost_array2 , label = 'Adam', linewidth=2)
    plt.legend(loc = 'best')
    plt.title(f'Cost Comparison : SGD vs. Adam',fontsize = 15)
    plt.xlabel('Iteration',fontsize = 15)
    plt.ylabel('Cost',fontsize = 15)
    plt.savefig(name,bbox_inches = 'tight')
    plt.show()
    
    

