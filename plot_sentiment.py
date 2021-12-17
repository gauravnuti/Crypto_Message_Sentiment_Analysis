import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Retreiving results from crypto_sentiment.py
file_name = "doge_sentiment_summary.pkl"
open_file = open(file_name, "rb")
doge_scores = pickle.load(open_file)
open_file.close()

file_name = "shiba_sentiment_summary.pkl"
open_file = open(file_name, "rb")
shiba_scores = pickle.load(open_file)
open_file.close()

doge_pos_counts = [doge_scores[i][0] for i in range(15)]
doge_neg_counts = [doge_scores[i][1] for i in range(15)]
shiba_pos_counts = [shiba_scores[i][0] for i in range(15)]
shiba_neg_counts = [shiba_scores[i][1] for i in range(15)]

doge_counts = doge_pos_counts + doge_neg_counts
shiba_counts = shiba_pos_counts + shiba_neg_counts

pos_counts = [doge_scores[i][0] + shiba_scores[i][0] for i in range(15)]
neg_counts = [doge_scores[i][1] + shiba_scores[i][1] for i in range(15)]

num_doge = [doge_scores[i][0] + doge_scores[i][1] for i in range(15)]
num_shiba = [shiba_scores[i][0] + shiba_scores[i][1] for i in range(15)]
tot_crypto = [num_doge[i] + num_shiba[i] for i in range(15)]
avg_scores = [doge_scores[i][2]*num_doge[i]
              + shiba_scores[i][2]*num_shiba[i] for i in range(15)]
avg_scores = [avg_scores[i]/tot_crypto[i] for i in range(15)]

# The following is the plot for Doge vs Shiba for total
# messages, further split by positive and negative.
df = pd.DataFrame(
    dict(
        day=[i+1 for i in range(15)] * 2 * 2,
        coins=["doge"] * 30 + ["shiba"] * 30,
        response=(["Positive"]*15 + ["Negative"]*15)*2,
        cnt=doge_counts + shiba_counts,
    )
)

fig = go.Figure()

fig.update_layout(
    template="simple_white",
    xaxis=dict(title_text="Day"),
    yaxis=dict(title_text="Count"),
    barmode="stack",
)

colors = ["#2A66DE", "#FFC32B"]

for r, c in zip(df.response.unique(), colors):
    plot_df = df[df.response == r]
    fig.add_trace(
        go.Bar(
            x=[plot_df.day, plot_df.coins],
            y=plot_df.cnt, name=r, marker_color=c
            ),
    )

fig.show()

# For plotting total number of messages for doge
# and shiba, uncomment the following:
# fig = go.Figure(data=[
#     go.Bar(name='Positive', x=[str(i+1) for i in range(15)], y=pos_counts),
#     go.Bar(name='Negative', x=[str(i+1) for i in range(15)], y=neg_counts)
# ])
# fig.update_layout(
#     barmode='stack',
#     xaxis=dict(title_text='Day'),
#     yaxis=dict(title_text='Num Messages'),
# )
# fig.show()

# For plotting avg_score of messages for doge
# and shiba, uncomment the following:
# fig = go.Figure(data=[
#     go.Bar(x=[str(i+1) for i in range(15)], y=avg_scores)
# ])
# fig.update_layout(
#     barmode='stack',
#     xaxis=dict(title_text='Day'),
#     yaxis=dict(title_text='Avg Score'),
# )
# fig.show()
