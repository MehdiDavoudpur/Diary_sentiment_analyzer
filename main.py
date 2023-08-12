from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as sl
import plotly.express as px

sl.title('Diary Tone')

# create a list of diary texts & a list of dates of diaries:
diary_list = []
date_list = []
for i in range(7):
    path = f'2023-10-2{i + 1}.txt'
    date_string = path.strip('.txt')
#    new_date_string = date_string.replace('-', '')
#    date_int = int(date_string)
    date_object = datetime.strptime(date_string, '%Y-%m-%d').date()
    date_list.append(date_object)
    with open(path, 'r') as file:
        my_diary = file.read()
        diary_list.append(my_diary)

# create a list of polarity score of each diary:
analyzed_diary_list = []
analyzer = SentimentIntensityAnalyzer()
for diary in diary_list:
    my_analyzed_diary = analyzer.polarity_scores(diary)
    analyzed_diary_list.append(my_analyzed_diary)

print(analyzed_diary_list)

# create proper date for positivity plot chart
sl.subheader('Positivity')

x_dots = date_list
y_dots = [item['pos']*100 for item in analyzed_diary_list]

print(y_dots)

my_plot = px.line(x=x_dots, y=y_dots, labels={'x': 'date', 'y': 'Positivity%'})
sl.plotly_chart(my_plot)

# create proper date for negativity plot chart
sl.subheader('Negativity')

x_dots = date_list
y_dots = [item['neg']*100 for item in analyzed_diary_list]

print(y_dots)

my_plot = px.line(x=x_dots, y=y_dots, labels={'x': 'date', 'y': 'Negativity%'})
sl.plotly_chart(my_plot)

# create proper date for neutrality plot chart
sl.subheader('Neutrality')

x_dots = date_list
y_dots = [item['neu']*100 for item in analyzed_diary_list]

print(y_dots)

my_plot = px.line(x=x_dots, y=y_dots, labels={'x': 'date', 'y': 'Neutrality%'})
sl.plotly_chart(my_plot)


# create proper date for sentiment plot chart
sl.subheader('Sentiment')

x_dots = date_list
y_dots = [item['compound']*100 for item in analyzed_diary_list]

print(y_dots)

my_plot = px.line(x=x_dots, y=y_dots, labels={'x': 'date', 'y': 'Sentiment%'})
sl.plotly_chart(my_plot)
