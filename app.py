from flask import Flask, Flask, redirect, url_for, render_template, request, flash
from flask import send_from_directory, abort
import os

from sklearn import datasets
from statistic import DataSet
import pandas as pd
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

global dataset
global sparsity

app.config["img"]="/Users/damiangorcak/Desktop/BP-app/static/img"

@app.route("/")
def index():
    return render_template("main.html")

@app.route("/main")
def main():
    return render_template("main.html")


app.config["uploads_dataset"] = "/Users/damiangorcak/Desktop/BP-app/static/datasets"
app.config['SECRET_KEY'] = 'anystringthatyoulike'

@app.route("/upload", methods=["GET", "POST"])
def upload():

    if request.method == "POST":
        print("here")
        if request.files:

            file = request.files["file"]
            user_col = request.form.get("user_id")
            item_col = request.form.get("item_id")
            rating_col = request.form.get("rating")
            print(user_col,item_col,rating_col)
            if "csv" in file.content_type or "json" in file.content_type:
                file.save(os.path.join(app.config["uploads_dataset"], file.filename))
                path_to_file = os.path.join(app.config["uploads_dataset"], file.filename)
                flash(u'data uspesne ulozene', 'ok')
            else: 
                flash(u'data niesu ani v json ani v csv formate', 'error')
                return render_template("upload.html")

            df = pd.read_csv(path_to_file)
            
            global dataset 
            dataset = DataSet(dataset=df,user="user_id",item="movie_id",rating="rating")

    return render_template("upload.html")

@app.route("/statistic")
def statistic():
    if len(os.listdir(app.config["uploads_dataset"]) ) == 0:
        return render_template("empty.html")
    else:
        return render_template("statistic.html")

@app.route("/get_statistic", methods=["GET"])
def get_statistic():
    global dataset
    global sparsity

    print("hello")
    if not "dataset" in globals():
        df = pd.read_csv(os.path.join(app.config["uploads_dataset"],os.listdir(app.config["uploads_dataset"])[0]))
        dataset = DataSet(dataset=df,user="user_id",item="movie_id",rating="rating")

    if len(os.listdir(app.config["img"])) == 4:

        if not "sparsity" in globals():
            sparsity = '{:.2f}%'.format(dataset.sparsity())

        return render_template("stats.html", sparsity=sparsity)

    sparsity = '{:.2f}%'.format(dataset.sparsity())
    dataset.sparsity_graph(
        fig_location="/Users/damiangorcak/Desktop/BP-app/static/img/sparsity.png"
    )

    dataset.ratings_graph(all=True, all_path="/Users/damiangorcak/Desktop/BP-app/static/img/all.png",
                            scatter=True, scatter_path="/Users/damiangorcak/Desktop/BP-app/static/img/scatter.png",
                            reduced=True, reduced_path="/Users/damiangorcak/Desktop/BP-app/static/img/reduced.png")

    return render_template("stats.html", sparsity=sparsity)



if __name__ == "__main__":
    app.run(debug=True)
