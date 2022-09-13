# import os
# PATH = os.path.join(os.getcwd(), "ml_algo")
# os.chdir(PATH)

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import xgboost as xgb
from dotenv import load_dotenv
from flask import Flask, render_template
from plotly.subplots import make_subplots
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.create_dataset import main

# Load model
MODEL_PATH = os.path.join(Path(os.getcwd()), "model", "xgboost.txt")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

# Load features

FEATURES_PATH = os.path.join(Path(os.getcwd()), "model", "features.txt")
features = []
with open(FEATURES_PATH, "r") as f:
    lines = f.readlines()
    for line in lines:
        features.append(line.strip("\n"))

# Get dataset with features

load_dotenv()
SECTOR = os.getenv("SECTOR")
HOST = os.getenv("HOST")
df = main(sector=SECTOR, host=HOST)

# Create pipeline to transform features and predict output

preprocessor = ColumnTransformer(
    remainder="drop", transformers=[("mm", MinMaxScaler(), features)]
)

pipe = Pipeline([("preprocessor", preprocessor)])

X = pd.DataFrame(pipe.fit_transform(df), columns=features)
ppred = model.predict_proba(X)[:, 1]

# Extract dates & index for plot

dates = df.loc[:, "Date"].values
index = df.loc[:, "Index"].values

# Set flask server to host the chart

FLASK_HOST = os.getenv("FLASK_HOST")
FLASK_PORT = os.getenv("FLASK_PORT")

app = Flask(__name__)


@app.route("/")
def xgboost_pred():
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(template="plotly_white", width=1500, height=750)
    fig.add_trace(
        go.Line(x=dates, y=index, name="Index", line=dict(color="#385A72", width=1.5))
    )
    fig.add_trace(
        go.Line(
            x=dates,
            y=ppred,
            name="Probability",
            line=dict(color="#24E6C7", width=0.75),
        ),
        secondary_y=True,
    )
    fig.update_yaxes(
        range=[4.175, np.max(index) + 0.1], secondary_y=False, showgrid=False
    )
    fig.update_yaxes(range=[0, 2], secondary_y=True, showgrid=True)
    fig.update_xaxes(showgrid=True)
    fig.layout.yaxis.update(showticklabels=False)
    fig.layout.yaxis2.update(showticklabels=False)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("xgboost_pred.html", graphJSON=graphJSON)


if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)
