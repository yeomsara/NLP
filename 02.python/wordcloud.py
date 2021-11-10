import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
from io import BytesIO 
import seaborn as sns

def read_img_from_url_and_return_matrix(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_matrix = np.array(img)
    return img_matrix


def make_colors(word,font_size,position,orientation,random_state,**kwargs):
    r = random_state.randint(0,105)
    g = random_state.randint(82,155)
    b = random_state.randint(33,103)    
    color = f'rgb({r},{g},{b})'
    
    return color

def make_ImageColoredWordcloud(text, img_matrix, outputfile_name,color):
    wc = WordCloud(width = 800, height = 800,background_color="white",
                   max_words=3000, mask=img_matrix, min_font_size=4, 
                   max_font_size=80, random_state=42,stopwords=stop_words,
                  relative_scaling =0,colormap=color)
#     ,colormap=color

    # generate word cloud
#     wc.generate(text)
    wc.generate_from_frequencies(text)
    print(make_colors)
    wc.recolor(color_func=make_colors,random_state=True)
    f = plt.figure(figsize=(20, 20))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(outputfile_name)
    
    
keyword = pd.read_csv('eu_pos_keyword.csv')

fig, axes = plt.subplots(2,1,figsize=(10,20))
ax = sns.barplot(ax=axes[0],x='count',y='word',data=keyword.sort_values('count',ascending = False).head(30),palette='Greens_d')


url        = "temp/img/circle3.png"
img_matrix = np.array(Image.open(url))
keyword_df = df[(df['sentiment'] == 'positive') ].groupby(['word'])['count'].sum().reset_index()
pos_text       = pd.Series(keyword_df['count'].values,index=keyword_df.word).to_dict()

savepath = 'temp/img/coloredWordCloud.png'
make_ImageColoredWordcloud(pos_text, img_matrix,savepath ,'Greens')