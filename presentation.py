import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#from bokeh.plotting import figure, show, output_notebook
#from bokeh.models import ColumnDataSource, HoverTool
#from bokeh.io import push_notebook
#from bokeh.transform import factor_cmap
#from bokeh.palettes import Category10
import ast
import plotly.express as px


st.title("Are epistemic and moral virtues semantically distinct?")

'''
I investigate whether moral and epistemic concepts are semantically distinct in ordinary language. Some philosophers argue that these concepts overlap considerably, while others maintain that they are fundamentally different. To empirically explore this issue, we employ word embeddings of moral and epistemic virtue terms to examine the contextual usage of these terms. The methodology involves calculating the cosine similarity between word embeddings, conducting clustering analyses, and performing statistical tests to assess the distinctiveness of these concepts.

We first demonstrate that the methodology works by showing that words representing big-5 personality traits form distinct clusters. Then we apply the technique to moral and epistemic virtue words. We find two clearly distinct clusters. This shows that moral and epistemic virtues are semantically distinct.

'''

# Functions
def get_embedding(word):
    """
    Given a word, return its embedding using OpenAI API.
    """
    response = openai.embeddings.create(
        model="text-embedding-3-large",
        input=word
    )
    return response.data[0].embedding

def apply_get_embedding(df, column_name):
    """
    Apply get_embedding function to a column in a pandas dataframe row-wise.
    """
    df['embedding'] = df[column_name].apply(get_embedding)
    return df

def extract_pca_loadings(df, embedding_column, n_components=2):
    """
    Apply PCA to a column of word embeddings and add the PCA loadings to the dataframe.
    """
    pca = PCA(n_components=n_components)
    embeddings = np.array(df[embedding_column].tolist())
    pca_loadings = pca.fit_transform(embeddings)
    for i in range(n_components):
        df[f'PCA_{i+1}'] = pca_loadings[:, i]
    return df, pca

def plot_pca(df, word_column, category_column, pca_columns):
    """
    Create a 2D graph of the PCA loadings with words and categories.
    """
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=df[pca_columns[0]], 
        y=df[pca_columns[1]], 
        hue=df[category_column], 
        palette='tab10', 
        s=100, 
        alpha=0.7,
        legend='full'
    )
        
    # Annotate each point with the word
    #for i in range(df.shape[0]):
    #    plt.text(df[pca_columns[0]].iloc[i], df[pca_columns[1]].iloc[i], df[word_column].iloc[i], 
    #             fontsize=9, alpha=0.9)
    
    plt.title('PCA of Word Embeddings')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(title=category_column)
    plt.grid(True)
    plt.show()



def plot_pca_interactive(df, word_column, category_column, pca_columns):
    """
    Create an interactive 2D graph of the PCA loadings with words and categories using Plotly.
    """
    num_categories = df[category_column].nunique()
    
    if num_categories == 2:
        color_discrete_sequence = ["#1f77b4", "#ff7f0e"]  # Default colors for two categories
    else:
        color_discrete_sequence = px.colors.qualitative.Plotly

    fig = px.scatter(df, x=pca_columns[0], y=pca_columns[1],
                     color=category_column, hover_name=word_column,
                     labels={pca_columns[0]: 'PCA 1', pca_columns[1]: 'PCA 2'},
                     title="PCA of Word Embeddings",
                     color_discrete_sequence=color_discrete_sequence)

    fig.update_traces(marker=dict(size=10, opacity=0.6),
                      selector=dict(mode='markers'))
    fig.update_layout(showlegend=True)
    return fig

'''
## Big Five

Let's begin with showing the methodology using words associated with the big five personality traits. I used GPT to create a database of 30 adjectives associated with each of the Big-five personality traits. Below is a sample of the dataset:

'''

# Load the Big Six Adjectives dataset
df_big_five = pd.read_csv('./big_five_adjectives.csv')

st.table(df_big_five.groupby('Category').head(2))

'''
[Full Dataset](https://github.com/marco2meyer/moral-epistemic/blob/main/big_five_adjectives.csv)
'''

'''
I use OpenAI to extract word embeddings for each of the words. Each word embedding is an array with 3072 dimensions. In the next step, I reduce dimensionality to 2 dimensions using principal component analysis. Below is an interactive plot of the embeddings. 

'''

# Apply the embedding extraction
#df = apply_get_embedding(df, 'Adjective')
df = pd.read_csv("./big_five_cum_embeddings.csv")
df['embedding'] = df['embedding'].apply(ast.literal_eval)

# Extract PCA loadings
df, pca = extract_pca_loadings(df, 'embedding')

# Plot the PCA loadings
st.plotly_chart(plot_pca_interactive(df, 'Adjective', 'Category', ['PCA_1', 'PCA_2']))

'''
The clusters are clearly visible, showing that the words associated with each of the Big-5 personality traits are indeed semantically distinct. This result can be made more precise by computing cosine similarities within and between word in different categories.  
'''

'''
## Epistemic vs. Moral virtues 

Now we use the same methodology to investigate whether moral and epistemic virtues are semantically distinct. First, I asked GPT to constract a dataset with 50 adjectives associated with epistemic virtue, and 50 adjectives associated with moral virtue. Below is a sample: 
'''

# Load the Big Six Adjectives dataset
df_moral_epistemic = pd.read_csv('./moral_epistemic_adjectives.csv')
st.table(df_moral_epistemic.groupby('Category').head(10))
'''
[Full Dataset](https://github.com/marco2meyer/moral-epistemic/blob/main/moral_epistemic_adjectives.csv)
'''

# Apply the embedding extraction
#df2 = apply_get_embedding(df2, 'Adjective')
df2 = pd.read_csv('./moral_epistemic_cum_embeddings.csv')
df2['embedding'] = df2['embedding'].apply(ast.literal_eval)

# Extract PCA loadings
df2, pca2 = extract_pca_loadings(df2, 'embedding')

# Plot the PCA loadings
st.plotly_chart(plot_pca_interactive(df2, 'Adjective', 'Category', ['PCA_1', 'PCA_2']))

'''The graph clearly shows that there are semantic differences between epistemic and moral virtues, with very little overlap. Epistemic and moral virtues are indeed semantically distinct.
'''

'''
## Next steps

- Better corpus of epistemic and moral terms
- Add quantitative metrics
- Compare conceptual, semantic, and psychological distinctions 
- Relate to existing literature on epistemic and moral virtues.
'''