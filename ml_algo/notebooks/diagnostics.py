import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
import seaborn as sns
import numpy as np
import pandas as pd


class ClassificationDiagnostics:
    def __init__(self, y, yhat, yhat_prob):
        self.y = y
        self.yhat = yhat
        self.yhat_prob = yhat_prob

    labels = ["Don't Sell", "Sell"]
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    plt.close()

    def plotConfusionMatrix(self):
        self.ax[0, 0].cla()
        cf_matrix = confusion_matrix(self.y, self.yhat)
        self.ax[0, 0] = sns.heatmap(
            np.eye(2),
            annot=cf_matrix,
            fmt="g",
            cbar=False,
            linewidths=20,
            cmap=ListedColormap(["#24E6C7", "#385A72"]),
            ax=self.ax[0, 0],
        )

        plt.sca(self.ax[0, 0])
        plt.title("Confusion Matrix \n")
        plt.xlabel("Predicted Values")
        plt.ylabel("Actual Values")
        plt.close()

        self.ax[0, 0].xaxis.set_ticklabels(self.labels)
        self.ax[0, 0].yaxis.set_ticklabels(self.labels)

    def plotClassificationReport(self):
        self.ax[0, 1].cla()
        clf_report = classification_report(
            self.y, self.yhat, target_names=self.labels, output_dict=True
        )
        clf_report = pd.DataFrame(clf_report).iloc[:-1, :].T
        cmap = ListedColormap(["#24E6C7", "#2EA09D", "#385A72"])
        clmns = clf_report.columns
        output = 1
        for clmn in clmns:
            for idx, row in clf_report.iterrows():
                if row[clmn] < output:
                    output = row[clmn]
                else:
                    pass

        sns.heatmap(
            clf_report,
            cmap=cmap,
            vmin=round(output, 2),
            vmax=1,
            linewidth=10,
            annot=True,
            fmt=".2f",
            ax=self.ax[0, 1],
            cbar=False,
        )

        plt.sca(self.ax[0, 1])
        plt.title("Classification Report \n")

    def plotROC(self):
        self.ax[1, 0].cla()
        fpr, tpr, _ = roc_curve(self.y, self.yhat_prob, drop_intermediate=False)

        sns.lineplot(x=fpr, y=tpr, color="#385A72", legend=False, ax=self.ax[1, 0])
        self.ax[1, 0].fill_between(fpr, tpr, color="#24E6C7")
        sns.lineplot(
            x=range(0, 2),
            y=range(0, 2),
            style=True,
            dashes=[(5, 5)],
            legend=False,
            color="#385A72",
            ax=self.ax[1, 0],
        )

        plt.sca(self.ax[1, 0])
        plt.box(False)
        plt.xticks([0, 0.5, 1])
        plt.title(f"ROC Curve (AUC={auc(fpr, tpr):.4f})", loc="center")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

    def plotPR(self):
        self.y = pd.Series(self.y)
        self.ax[1, 1].cla()
        precision, recall, _ = precision_recall_curve(self.y, self.yhat_prob)
        noskill_classifier = len(self.y[self.y == True]) / len(self.y)

        sns.lineplot(
            x=recall, y=precision, color="#385A72", legend=False, ax=self.ax[1, 1]
        )
        self.ax[1, 1].fill_between(recall, precision, color="#24E6C7")
        sns.lineplot(
            x=range(0, 2),
            y=noskill_classifier,
            style=True,
            dashes=[(5, 5)],
            legend=False,
            color="#385A72",
            ax=self.ax[1, 1],
        )

        plt.sca(self.ax[1, 1])
        plt.box(False)
        plt.xticks([0, 0.5, 1])
        plt.title(f"PR Curve (AUC={auc(recall, precision):.4f})", loc="center")
        plt.xlabel("Recall")
        plt.ylabel("Precision")

    def plotDiagnostics(self):
        # plt.close()
        self.plotConfusionMatrix()
        self.plotClassificationReport()
        self.plotROC()
        self.plotPR()
        self.fig.tight_layout(pad=3.0)
        self.fig.suptitle("Classification Diagnostics")
        plt.show()
