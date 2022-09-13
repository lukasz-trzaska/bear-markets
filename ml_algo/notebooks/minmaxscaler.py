import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def mmscaler(columns, df):

    preprocessor = ColumnTransformer(
        remainder="passthrough", transformers=[("mm", MinMaxScaler(), columns)]
    )
    pipe = Pipeline(steps=[("preprocessor", preprocessor)])
    pipe.fit(df)

    format = lambda x: x.replace("mm__", "").replace("remainder__", "")
    format_col = list(map(format, pipe.get_feature_names_out()))
    output = pd.DataFrame(pipe.fit_transform(df), columns=format_col)
    output = output.astype({"sell_signal": bool, "vs_quantile_binary": bool})
    return output
