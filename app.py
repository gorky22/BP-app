from tkinter.font import names
from tkinter.ttk import Separator
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
import surprise
import json

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
    names_row = db.Column("names_row", db.String(100))
    rating = db.Column("rating", db.String(100))
    path_to_file = db.Column("path_to_file", db.String(100))
    sparsity = db.Column("sparsity", db.String(100))
    path_img_stats_all = db.Column("path_img_stats_all", db.String(200))
    path_img_stats_reduced = db.Column("path_img_stats_reduced", db.String(200))
    path_img_stats_scatter = db.Column("path_img_stats_scatter", db.String(200))
    path_img_stats_sparsity = db.Column("path_img_stats_sparsity", db.String(200))
    path_to_img_stats = db.Column("path_to_img_stats", db.String(200))
    path_train_graph = db.Column("path_train_graph", db.String(200))
    train_steps = db.Column("train_steps", db.Integer)
    train_lr = db.Column("train_lr", db.Float)
    train_alg = db.Column("train_alg", db.String(10))
    path_to_model_svd = db.Column("path_to_model_svd", db.String(200))
    path_to_model_sgd = db.Column("path_to_model_sgd", db.String(200))
    path_to_model_als = db.Column("path_to_model_als", db.String(200))
    path_to_dataset = db.Column("path_to_dataset", db.String(200))
    path_to_img_rated = db.Column("path_to_img_rated", db.String(200))
    had_genre = db.Column("had_genre", db.Boolean, default=False)
    sep = db.Column("sep", db.String(200))

app.config["uploads_dataset"] = "/Users/damiangorcak/Desktop/BP-app/static/datasets"
app.config['SECRET_KEY'] = 'anystringthatyoulike'

db.create_all()
global dataset
global sparsity

def save_model(alg, model, database):
    path_model = database.path_to_file.split("datasets/")[0] + "models/" + database.path_to_file.split("datasets/")[1].split(".")[0] + "-" + alg + "_pkl"
    
    with open(path_model, 'wb') as files:
            pickle.dump(model, files)

    return path_model

@app.route("/")
def index():
    return render_template("main.html")

@app.route("/main")
def main():
    x = datasets.query.all()
    return render_template("main.html",x=x)

@app.route("/choose_dataset", methods=["POST"])
def choose_dataset():
    session["df"] = request.form['dataset']
    global dataset

    x = datasets.query.filter_by(name=session["df"]).all()[0]
    with open(x.path_to_dataset, 'rb') as ds:
        dataset = pickle.load(ds)

    flash(u'dataset is chosen', 'ok')
    x = datasets.query.all()
    return render_template("main.html",x=x)

def save_file(file,name,main=False):
    if "csv" in file.content_type or "json" in file.content_type:
        if main:
            file.save(app.config["uploads_dataset"] + "/names_" + name )
            return app.config["uploads_dataset"] + "/names_" + name 
        else:
            file.save(os.path.join(app.config["uploads_dataset"], name))
            return os.path.join(app.config["uploads_dataset"], name)
    else:
        return None

def get_df_and_check_columns(content_type,path_to_file,separator, user_col=None,item_col=None,rating_col=None,names_col=None, name_id=None, genre=None):
    if "csv" in content_type:
        df = pd.read_csv(path_to_file, sep=separator ,encoding='latin-1', on_bad_lines='skip',)
    else:
        df = pd.read_json(path_to_file, lines=True)

    if names_col == None:

        if  not all(x in list(df.columns) for x in [user_col, item_col, rating_col]):
            return None
        else :
            return df
    
    else:

        if user_col == None:
            if genre == None:
                if  not all(x in list(df.columns) for x in [names_col]):
                    return None
                else :
                    return df
            else:
                if  not all(x in list(df.columns) for x in [names_col,genre]):
                    return None
                else :
                    return df
        else:

            if  not all(x in list(df.columns) for x in [user_col, item_col, rating_col,names_col,name_id]):
                return None
            else :
                return df
        


@app.route("/upload", methods=["GET", "POST"])
def upload():
    db.drop_all()
    db.create_all()
    if request.method == "POST":
        if request.files:
            
            file = request.files["file"]
            user_col = request.form.get("user_id")
            item_col = request.form.get("item_id")
            rating_col = request.form.get("rating")
            names_col = request.form.get("item_names")
            name_id_col = request.form.get("item_names_id")
            name = request.form.get("name")
            sep = request.form.get("separator")
            genre = request.form.get("genre")
            genre_separator = request.form.get("genre-separator")

            if request.form.get("genre") is not None and request.form.get("genre-separator") == False:
                genre = None

            if name_id_col == False:
                name_id_col = item_col

            file_names = request.files["file_names"]
            global dataset 
            if file_names is not None and names_col is None:
                flash(u'dataset s menami zadany ale nazov stlpca s menami nieje zadany', 'error')
                return render_template("upload.html")

            if genre != False:
                new = datasets(user_row=user_col, item_row=item_col, rating=rating_col, names_row=names_col, had_genre=True, sep=genre_separator)
            else:
                new = datasets(user_row=user_col, item_row=item_col, rating=rating_col, names_row=names_col)

            db.session.add(new)
      

            file_path = file.filename.split(".")[0] + str(new._id) + "." + file.filename.split(".")[1]


            path_to_file = save_file(file,file_path)
            path_to_file_names = None
            df_names = None

            if file_names is not None:
                path_to_file_names = save_file(file_names,file_path,main=True)

                if path_to_file_names is None:
                    flash(u'data s menami niesu ani v json ani v csv formate', 'error')
                    return render_template("upload.html")
                
                if new.had_genre:
                    df_names = get_df_and_check_columns(file_names.content_type,path_to_file=path_to_file_names,separator=sep,names_col=names_col,name_id=name_id_col, genre=genre)
                else:
                    df_names = get_df_and_check_columns(file_names.content_type,path_to_file=path_to_file_names,separator=sep,names_col=names_col,name_id=name_id_col)

                if df_names is None:
                    flash(u'zadane stlpce (dataset s menami) niesu v zadanom datasete', 'error')
                    return render_template("upload.html")

            if path_to_file is None :
                flash(u'data niesu ani v json ani v csv formate', 'error')
                return render_template("upload.html")
            
            pkl_path_dataset =  path_to_file.split(".")[0] + "_datasets.pkl"
            new.path_to_dataset = pkl_path_dataset
            new.path_to_file = path_to_file

            if df_names is None:
                df = get_df_and_check_columns(file.content_type,path_to_file=path_to_file,separator=sep, user_col=user_col,item_col=item_col,rating_col=rating_col,names_col=names_col)

                dataset = DataSet(dataset=df,user=user_col,item=item_col,rating=rating_col, name=names_col)

                dataset.save(pkl_path_dataset)
            else:
                
                df = get_df_and_check_columns(file.content_type,path_to_file=path_to_file,separator=sep,user_col=user_col,item_col=item_col,rating_col=rating_col)

                if df is None:
                    flash(u'zadane stlpce niesu v zadanom datasete', 'error')
                    return render_template("upload.html")

                dataset = DataSet(dataset=df,user=user_col,item=item_col,rating=rating_col)

                dataset.save(pkl_path_dataset)

            if df is None:
                flash(u'zadane stlpce niesu v zadanom datasete', 'error')
                return render_template("upload.html")


            if df_names is not None:
                
                if new.had_genre:
                    dataset = DataSet(dataset=df,user=user_col,item=item_col,rating=rating_col,name=names_col,dataset_names=df_names, name_item_id_name=name_id_col, genre=genre)
                else:
                    dataset = DataSet(dataset=df,user=user_col,item=item_col,rating=rating_col,name=names_col,dataset_names=df_names, name_item_id_name=name_id_col)

                dataset.save(pkl_path_dataset)
            
            flash(u'data uspesne ulozene', 'ok')

            
            new.name = name + str(new._id)
            db.session.commit() 
            session["df"] = new.name
        
            
    
    return render_template("upload.html")

@app.route("/statistic")
def statistic():

    if session.get("df") is None:
        return render_template("empty.html", 
                text="Nieje zadany ziaden subor s datami, prosim vlozte how v zalozke 'add data'")
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

    if x.path_img_stats_sparsity is None:
        #print("tu1")
        #print(x.path_to_file)
        #print(x.item_row)
        if not "dataset" in globals():
            with open(x.path_to_dataset, 'rb') as ds:
                dataset = pickle.load(ds)

        #print("tu2")
        x.sparsity = '{:.2f}%'.format(dataset.sparsity())
        #print("tu3")
        name = x.path_to_file.split("datasets/")[1].split(".")[0] + ".png"
        #print("tu4")
        x.path_img_stats_all = "static/img/all_" + name
        x.path_img_stats_reduced = "static/img/reduced_" + name
        x.path_img_stats_scatter = "static/img/scatter_" + name
        x.path_img_stats_sparsity = "static/img/sparsity_" + x.path_to_file.split("datasets/")[1].split(".")[0] + ".jpg"
        db.session.commit()
        #print("tu5")
        dataset.sparsity_graph(
            fig_location=x.path_img_stats_sparsity
        )


        dataset.ratings_graph(all=True, all_path=x.path_img_stats_all,
                            scatter=True, scatter_path=x.path_img_stats_scatter,
                            reduced=True, reduced_path=x.path_img_stats_reduced )

        return render_template("stats.html", x=x)

    return render_template("stats.html",x=x)

@app.route("/train")
def train():
    if session.get("df") is None:
        return render_template("empty.html", 
                text="Nieje zadany ziaden subor s datami, prosim vlozte how v zalozke 'add data'")
    else:
        x = datasets.query.filter_by(name=session["df"]).all()[0]
        if x.path_train_graph is not None:
            return render_template("train_stats.html", x=x)

        return render_template("train.html")

@app.route("/find_hyperparams", methods=["GET","POST"])
def find_hyperparams():
    #print("hello")

    global dataset

    x = datasets.query.filter_by(name=session["df"]).all()[0]
    #print(x.path_train_graph)
    if request.method == "POST":
        #print("hello")
        if not "dataset" in globals():
            with open(x.path_to_dataset, 'rb') as ds:
                dataset = pickle.load(ds)

        min = dataset.find_hyperparams(alg=request.form.get("type"))
        #print(min)
        

        path = x.path_to_file.split("datasets/")[1].split(".")[0] + ".png"
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

    if session.get("df") is None:
        return render_template("empty.html", 
                text="Nieje zadany ziaden subor s datami, prosim vlozte how v zalozke 'add data'")

    if request.method == "POST":

        x = datasets.query.filter_by(name=session["df"]).all()[0]
        
        global dataset

        alg = request.form.get("alg")

        path = x.path_to_file.split("datasets/")[0] + "predicted/" + x.path_to_file.split("datasets/")[1].split(".")[0] + "_" + alg + "_predicted.png"
        
        if not "dataset" in globals():
            with open(x.path_to_dataset, 'rb') as ds:
                dataset = pickle.load(ds)

        lr = float(request.form.get("lr"))

        if alg == "svd":
            model = dataset.train_model_svd(lr=lr, steps=int(request.form.get("steps")))

            if model is None:
                return render_template("no_rewiews.html")

        elif alg == "sgd":
            model = dataset.train_model_sgd(lr=lr, steps=int(request.form.get("steps")))

            if model is None:
                return render_template("no_rewiews.html")
        else:
            model = dataset.train_model_als(lr=lr, steps=int(request.form.get("steps")))

            if model is None:
                return render_template("no_rewiews.html")

        dataset.find_predictions(path, model=model, alg=alg.lower())

        results = dataset.get_data_to_render_result(alg=alg.lower())

        # save class because new dict is apended

        dataset.save(path)
        
        if results is None:
            return render_template("empty.html",
                        text="pre tento algoritmus nieje vytrenovany model")
        else:
            return render_template("predictions.html",top_10 = results["top_10"], rated=results["rated"], path=results["path"])

       
    return render_template("train_model.html")



@app.route("/recomend")
def recomend():
    if session.get("df") is None:
        return render_template("empty.html", 
                text="Nieje zadany ziaden subor s datami, prosim vlozte how v zalozke 'add data'")
    else:
        x = datasets.query.filter_by(name=session["df"]).all()[0]
        global dataset
        if not "dataset" in globals():
            with open(x.path_to_dataset, 'rb') as ds:
                dataset = pickle.load(ds)

        if not x.had_genre:
            min, max, data= dataset.get_items()
            return render_template("recomend.html",min=min, max=max, data=data)
        else :
            min, max, data= dataset.get_items_genre()
            return render_template("recomend_genre.html",min=min, max=max, data=data)



@app.route("/make_recomendation",  methods=["POST"])
def make_recomendation():

    if session.get("df") is None:
        return render_template("empty.html")
    else:
        
        x = datasets.query.filter_by(name=session["df"]).all()[0]
        global dataset
        if not "dataset" in globals():
            with open(x.path_to_dataset, 'rb') as ds:
                dataset = pickle.load(ds)

        alg = None
        model = None

        

        if request.form.get("ALS") is not None:
            alg = "ALS"
            #print(dataset.make_dat_with_name(json.loads(request.form.get(alg))))
            model = dataset.train_model_als()
            x.path_to_model_als = save_model(alg.lower(),model,x)

        elif request.form.get("SVD") is not None:
            alg = "SVD"
            #print(dataset.make_dat_with_name(json.loads(request.form.get(alg))))
            model = dataset.train_model_svd()
            x.path_to_model_svd = save_model(alg.lower(),model,x)

        else:
            alg = "SGD"
           
            dataset.make_dat_with_name(json.loads(request.form.get(alg)))
            model = dataset.train_model_sgd()
            x.path_to_model_sgd = save_model(alg.lower(),model,x)
            

        path = x.path_to_file.split("datasets/")[0] + "predicted/" + x.path_to_file.split("datasets/")[1].split(".")[0] + "_" + alg + "_predicted.png"

        #print("data: ")
        if x.had_genre:
            dataset.find_predictions_genre(path, model=model, alg=alg.lower())
        else:
            dataset.find_predictions(path, model=model, alg=alg.lower())

        results = dataset.get_data_to_render_result(alg=alg.lower())

        # save class because new dict is apended

        dataset.save(x.path_to_dataset)
        
        if results is None:
            return render_template("empty.html",
                        text="pre tento algoritmus nieje vytrenovany model")

        db.session.commit()

    if x.had_genre:
        return render_template("predictions_genre.html",top_10 = results["top_10"], rated=results["rated"], 
                            ranges=results["ranges"], values=results["values"])
    else:
        return render_template("predictions_genre.html",top_10 = results["top_10"], rated=results["rated"], 
                            ranges=results["ranges"], values=results["values"])

@app.route("/als_results")
def als_results():
    x = datasets.query.filter_by(name=session["df"]).all()[0]
    global dataset
    if not "dataset" in globals():
        with open(x.path_to_dataset, 'rb') as ds:
            dataset = pickle.load(ds)
    
    results = dataset.get_data_to_render_result(alg="als")
    #print(results)
        
    if results is None:
        return render_template("empty.html",
                        text="pre tento algoritmus nieje vytrenovany model")
    else:
        return render_template("predictions.html",top_10 = results["top_10"], rated=results["rated"], 
                                    ranges=results["ranges"], values=results["values"])

@app.route("/sgd_results")
def sgd_results():
    x = datasets.query.filter_by(name=session["df"]).all()[0]
    global dataset
    if not "dataset" in globals():
        with open(x.path_to_dataset, 'rb') as ds:
            dataset = pickle.load(ds)
    
    
    results = dataset.get_data_to_render_result(alg="sgd")

    if results is None:
        return render_template("empty.html",
                        text="pre tento algoritmus nieje vytrenovany model")
    else:
        return render_template("predictions.html",top_10 = results["top_10"], rated=results["rated"], 
                                    ranges=results["ranges"], values=results["values"])

@app.route("/svd_results")
def svd_results():
    x = datasets.query.filter_by(name=session["df"]).all()[0]
    global dataset
    if not "dataset" in globals():
        with open(x.path_to_dataset, 'rb') as ds:
            dataset = pickle.load(ds)
    
    results = dataset.get_data_to_render_result(alg="svd")
  
    if results is None:
        return render_template("empty.html",
                        text="pre tento algoritmus nieje vytrenovany model")
    else:
        return render_template("predictions.html",top_10 = results["top_10"], rated=results["rated"],
                                ranges=results["ranges"], values=results["values"])
@app.route("/charts")
def charts():
    x = datasets.query.filter_by(name=session["df"]).all()[0]
    global dataset
    if not "dataset" in globals():
        with open(x.path_to_dataset, 'rb') as ds:
            dataset = pickle.load(ds)

    if x.sparsity is None:
        x.sparsity = '{:.2f}%'.format(dataset.sparsity())

    df = dataset.ratings_graph(all=True,scatter=True,reduced=True)
    ratings, counts = dataset.get_counts()
    #print(df)
    return render_template("charts.html",item_reduced=df["item_reduced"], 
                                         user_reduced=df["user_reduced"],
                                         user_all_user=df["user_all"]["user"],
                                         user_all_item=df["user_all"]["item"],
                                         item_all_item=df["item_all"]["item"],
                                         item_all_user=df["item_all"]["user"],
                                         scatter=df["scatter"],
                                         ratings=ratings,
                                         counts=counts,
                                         x=x
                                         )


if __name__ == "__main__":
    app.run(debug=True)
