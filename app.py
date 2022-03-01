import pkgutil
from tkinter import N
from unicodedata import name
from flask import Flask, Flask, redirect, url_for, render_template, request, flash
from flask import send_from_directory, abort, session
import os
from flask_sqlalchemy import SQLAlchemy
from numpy import delete
from sklearn import datasets
from statistic import DataSet
import pandas as pd
import matplotlib
import pickle
matplotlib.use('Agg')

app = Flask(__name__)

app.secret_key = "xxxxx"
app.config["img"]="/Users/damiangorcak/Desktop/BP-app/static/img"
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///datasets.sqlite3'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

class datasets(db.Model):
    _id = db.Column("id", db.Integer, primary_key=True)
    name = db.Column("name", db.String(100))
    user_row = db.Column("user_row", db.String(100))
    item_row = db.Column("item_row", db.String(100))
    rating = db.Column("rating", db.String(100))
    path_to_file = db.Column("path_to_file", db.String(100))
    sparsity = db.Column("sparsity", db.String(100))
    path_img_stats_all = db.Column("path_img_stats_all", db.String(200))
    path_img_stats_reduced = db.Column("path_img_stats_reduced", db.String(200))
    path_img_stats_scatter = db.Column("path_img_stats_scatter", db.String(200))
    path_img_stats_sparsity = db.Column("path_img_stats_sparsity", db.String(200))
    path_train_graph = db.Column("path_train_graph", db.String(200))
    train_steps = db.Column("train_steps", db.Integer)
    train_lr = db.Column("train_lr", db.Float)
    train_alg = db.Column("train_alg", db.String(10))
    path_to_model_svd = db.Column("path_to_model_svd", db.String(200))
    path_to_model_sgd = db.Column("path_to_model_sgd", db.String(200))
    path_to_model_als = db.Column("path_to_model_als", db.String(200))

db.create_all()
global dataset
global sparsity



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
            #db.drop_all()
            #db.create_all()
            file = request.files["file"]
            user_col = request.form.get("user_id")
            item_col = request.form.get("item_id")
            rating_col = request.form.get("rating")

            print(user_col,item_col,rating_col)
            new = datasets(user_row=user_col, item_row=item_col, rating=rating_col)

            db.session.add(new)
            db.session.commit()

            new.name = file.filename.split(".")[0] + str(new._id) + "." + file.filename.split(".")[1]

            db.session.commit()
            
            if "csv" in file.content_type or "json" in file.content_type:
                file.save(os.path.join(app.config["uploads_dataset"], new.name))
                path_to_file = os.path.join(app.config["uploads_dataset"], new.name)

                if "csv" in file.content_type:
                    df = pd.read_csv(path_to_file)
                else:
                    df = pd.read_json(path_to_file)

                if  not all(x in list(df.columns) for x in [user_col, item_col, rating_col]):
                    flash(u'zadane stlpce niesu v zadanom datasete', 'error')
                    return render_template("upload.html")

                pkl_path = path_to_file.split(".")[0] + ".pkl"

                df.to_pickle(pkl_path)
                
                flash(u'data uspesne ulozene', 'ok')
            else: 
                flash(u'data niesu ani v json ani v csv formate', 'error')
                return render_template("upload.html")

            new.path_to_file = pkl_path

            db.session.commit()

            
            
            global dataset 
            dataset = DataSet(dataset=df,user=user_col,item=item_col,rating=rating_col)
            session["df"] = new.name
            

            print(datasets.query.all())
            print(datasets.query.filter_by(name=session["df"]).all())
            
    
    return render_template("upload.html")

@app.route("/statistic")
def statistic():

    if session["df"] == None:
        return render_template("empty.html")
    else:
        x = datasets.query.filter_by(name=session["df"]).all()[0]

        
        if x.path_img_stats_sparsity is not None:
            return render_template("stats.html", x=datasets.query.filter_by(name=session["df"]).all()[0])
        else:
            return render_template("statistic.html")

@app.route("/get_statistic", methods=["GET"])
def get_statistic():
    global dataset

    x = datasets.query.filter_by(name=session["df"]).all()[0]

    if x.path_img_stats_sparsity == None:
        print("tu1")
        print(x.path_to_file)
        print(x.item_row)
        if not "dataset" in globals():
            dataset = DataSet(dataset=pd.read_pickle(x.path_to_file),
                            user=x.user_row,
                            item=x.item_row,
                            rating=x.rating)
        print("tu2")
        x.sparsity = '{:.2f}%'.format(dataset.sparsity())
        print("tu3")
        name = x.path_to_file.split("datasets/")[1].split(".")[0] + str(x._id) + ".png"
        print("tu4")
        x.path_img_stats_all = "static/img/all_" + name
        x.path_img_stats_reduced = "static/img/reduced_" + name
        x.path_img_stats_scatter = "static/img/scatter_" + name
        x.path_img_stats_sparsity = "static/img/sparsity_" + name
        db.session.commit()
        print("tu5")
        dataset.sparsity_graph(
            fig_location=x.path_img_stats_sparsity
        )


        dataset.ratings_graph(all=True, all_path=x.path_img_stats_all,
                            scatter=True, scatter_path=x.path_img_stats_scatter,
                            reduced=True, reduced_path=x.path_img_stats_reduced )

    #return render_template("stats.html", sparsity=x.sparsity,
    #                        img_sparsity=x.path_img_stats_sparsity,
    #                        img_all=x.path_img_stats_all,
    #                        img_scatter=x.path_img_stats_scatter,
    #                        img_reduced=x.path_img_stats_reduced)

    return render_template("stats.html",x=x)

@app.route("/train")
def train():
    if session["df"] == None:
        return render_template("empty.html")
    else:
        x = datasets.query.filter_by(name=session["df"]).all()[0]
        if x.path_train_graph is not None:
            return render_template("train_stats.html", x=x)

        return render_template("train.html")

@app.route("/find_hyperparams", methods=["GET","POST"])
def find_hyperparams():
    print("hello")

    global dataset

    x = datasets.query.filter_by(name=session["df"]).all()[0]
    print(x.path_train_graph)
    if request.method == "POST":
        print("hello")
        if not "dataset" in globals():
            dataset = DataSet(dataset=pd.read_pickle(x.path_to_file),
                                user=x.user_row,item=x.item_row,
                                rating=x.rating)

        min = dataset.find_hyperparams(alg=request.form.get("type"))
        print(min)
        

        path = x.path_to_file.split("datasets/")[1].split(".")[0] + str(x._id) + ".png"
        x.path_train_graph = "static/train_img/train_" + path
        x.train_steps = min["params"]['k']
        x.train_lr = min["params"]["learning_rate"]
        x.train_alg = min["alg"]
        x.train_rmse = min["target"]
        db.session.commit()

        if request.form.get("type") == "als":
            dataset.graphs_alghoritms(als=True, 
                                lr=True,   time=True, eval=True,save_path=x.path_train_graph)
        elif request.form.get("type") == "sgd":
            dataset.graphs_alghoritms(sgd=True, 
                                lr=True,   time=True, eval=True,save_path=x.path_train_graph)
        elif request.form.get("type") == "svd":
            dataset.graphs_alghoritms(svd=True, 
                                lr=True,   time=True, eval=True,save_path=x.path_train_graph)
        else:
            dataset.graphs_alghoritms(sgd=True, als=True, svd=True,
                                lr=True,   time=True, eval=True,save_path=x.path_train_graph)

    return render_template("train_stats.html", x=x)

@app.route("/train_model", methods=["GET","POST"])
def train_model():
    if request.method == "POST":

        if session["df"] == None:
            return render_template("empty.html")
        else:
            x = datasets.query.filter_by(name=session["df"]).all()[0]
        
        global dataset

        alg = request.form.get("alg")

        path = x.path_to_file.split("datasets/")[0] + x.path_to_file.split("datasets/")[1].split(".")[0] + str(x._id) + "-" + alg + "_pkl"

        if not "dataset" in globals():
            dataset = DataSet(dataset=pd.read_pickle(x.path_to_file),
                                user=x.user_row,item=x.item_row,
                                rating=x.rating)
        if alg == "svd":
            model = dataset.train_model_svd(lr=request.form.get("lr"), steps=request.form.get("steps"))
            x.path_to_model_svd = path
        elif alg == "sgd":
            model = dataset.train_model_sgd(lr=request.form.get("lr"), steps=request.form.get("steps"))
            x.path_to_model_sgd = path
        else:
            model = dataset.train_model_als(lr=request.form.get("lr"), steps=request.form.get("steps"))
            x.path_to_model_als = path

        with open(path, 'wb') as files:
            pickle.dump(model, files)
       
    return render_template("train_model.html")



if __name__ == "__main__":
    app.run(debug=True)
