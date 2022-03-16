from venv import create
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
import scipy.sparse as sparse
import pickle
import gzip
import surprise
from surprise import SVD
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from bayes_opt import BayesianOptimization
import time
from surprise import BaselineOnly
from surprise import Dataset
import math
import json
import itertools

# in this function we can load datased and make some statistic
# dataset is loaded dataset in pandas
# user is name of user column in dataset
# item is name of item column in dataset
# rating is name of rating column in dataset

class DataSet:

    # this variable has interaction matrix it is global because it is used in 2 methods
    # so we initialize it once and second time we dont need to

    __interaction_matrix = None
    __train = None
    __test = None
    __times = None
    __tuning_params = { "k": [1,100],
                        "learning_rate": [0.001, 0.02]
                        }
    __values_svd = None
    __values_als = None
    __values_sgd = None
    __surprise_dataset = None
    __name_and_id_item = None
    __to_train = None
    __user_id = None
    

    def __init__(self, dataset, user, item, rating, name=None, dataset_names=None, name_item_id_name=None, genre=None, sep=None):
        
        self.genre = False

        if dataset_names is not None:
            tmp = dataset_names.rename(columns={name_item_id_name:"item_id", name:"name", genre:"genre"})

            if genre is not None:
                self.dataset = dataset.rename(columns={user:"user_id", item:"item_id", rating:"rating"})
            else:
                self.dataset = dataset.rename(columns={user:"user_id", item:"item_id", rating:"rating"})

            tmp = pd.merge(self.dataset, tmp, how="left", on="item_id")
            
            if "rating_x" in list(tmp.columns):
                tmp = tmp.rename(columns={"rating_x":"rating"})

            if genre is None:
                self.__name_and_id_item = tmp[["user_id","item_id","name","rating"]]
            else:
                self.__name_and_id_item = tmp[["user_id","item_id","name","rating","genre"]]
                self.genre = True

        elif name is not None:
            self.dataset = dataset.rename(columns={user:"user_id", item:"item_id", rating:"rating", name:"name"})
            self.__name_and_id_item = self.dataset[["user_id","item_id","name","rating"]]
        else :
            self.dataset = dataset.rename(columns={user:"user_id", item:"item_id", rating:"rating"})
            self.__name_and_id_item = self.dataset
        
        tmp_dict = {"steps":None, "lr":None, "time":None}
        self.trained_result_dict = {"als":None, "sgd":None, "svd":None}
        self.hyperparams_results = {"als":tmp_dict, "sgd":tmp_dict, "svd":tmp_dict}
        self.sep=sep

        

    # this function returns matrix from dataset in coo format

    def __create_matrix_in_coo(self):
        
        # user and item should be converted to type category because in coo_matrix
        # function it should be integer

        tmp_dat = self.dataset

        tmp_dat.user_id = tmp_dat.user_id.astype("category").copy()
        tmp_dat.item_id = tmp_dat.item_id.astype("category").copy()

        # adding just nonzero values in coo format to matrix

        # [note] method will return float values so data should be float
        # [note] cat.codes after row is because we need to convert category type to int
        
        return sparse.coo_matrix((tmp_dat['rating'].astype(float),
                                                        (tmp_dat["user_id"].cat.codes, tmp_dat["item_id"].cat.codes)))

    # this function will count sparsiness of our dataset

    def sparsity(self, print_sparsity=False):
        
        if self.__interaction_matrix == None:
            self.__interaction_matrix = self.__create_matrix_in_coo()


        sparsity = float(len(self.__interaction_matrix.data)) 
        sparsity /= (self.__interaction_matrix.shape[0] * self.__interaction_matrix.shape[1])

        if print_sparsity == True:
            print('percentage of user-items that have a rating: {:.2f}%'.format(sparsity * 100))

        return sparsity * 100

    # this function will print graph of sparsity matrix

    def sparsity_graph(self, fig_location=None, show_figure=False):
        
        if self.__interaction_matrix == None:
            self.__interaction_matrix = self.__create_matrix_in_coo()

        # making some kind of scale graph will show values same for 350k and also for 3k

        x,y = self.__interaction_matrix.T.shape

        if x > 3000 and y > 3000:
            plt.figure(figsize=(x/500, y/500))
        
    
        plt.spy(self.__interaction_matrix.T,markersize=0.09)
        plt.xlabel("id of users")
        plt.ylabel("id of items")
        
        if fig_location != None:
            plt.savefig(fig_location,  bbox_inches='tight')

    # THIS function set text on slopes when plot bar

    def __set_text_on_slopes(ax, users_graph, item_graph):

        for i in range(0,2):
        
            if i == 0:
                pps = ax[i].bar(users_graph.reset_index().user_id,users_graph.reset_index().item_id, align='center')
                ax[i].set_xlabel("Number of of ratings by user")
                ax[i].set_ylabel("Number of users")
            else:
                pps = ax[i].bar(item_graph.reset_index().item_id,item_graph.reset_index().user_id, align='center')
                ax[i].set_xlabel("Number of ratings on item")
                ax[i].set_ylabel("Number of item")

            counter = 0

            ax[i].tick_params(labelrotation=90)

            # here we want to add percentage above each slope
            for p in pps:
                height = p.get_height()

                if i == 0:
                    who = users_graph.reset_index().percentage[counter]
                else:
                    who = item_graph.reset_index().percentage[counter]

                ax[i].text(x=p.get_x() + p.get_width() / 2, y=height+.10,
                            s="{}%".format(who),
                            ha='center',
                            weight='bold')

                counter += 1

        return ax

    # plot scatter

    def __plot_scatter(self, occurencies_item, path):

        tmp = occurencies_item.reset_index().groupby('user_id').count().reset_index()
        tmp = tmp.rename(columns={"user_id": "number_of_ratings"})

        if path != None:

            ax = tmp.plot.scatter(x='item_id',y="number_of_ratings",c='number_of_ratings',
                            cmap="RdYlGn",logy=True,s=2,sharex=False)

            ax.set_xlabel("number of users")
            ax.set_ylabel("number of ratings")

            plt.savefig(path)

        return {"scatter":json.dumps([[a]+[b] for a,b in zip(tmp.number_of_ratings.to_list() ,tmp.item_id.to_list())])}
    # plot reduced

    def __plot_reduced(self, occurencies_user, occurencies_item, path):

        # added user_id column to user_ratings
        # added book_id column to item_ratings

        user_ratings = occurencies_item.reset_index().groupby('user_id').count().reset_index()
        item_ratings = occurencies_user.reset_index().groupby('item_id').count().reset_index()
        if path != None:
            fig_all, ax_all = plt.subplots(
                        nrows=1, ncols=2, constrained_layout=True, figsize=(30, 10)
                    )

            user_ratings.plot.bar(ax=ax_all[0], x='user_id',y='item_id',logy=True)
            item_ratings.plot.bar(ax=ax_all[1], x='item_id',y='user_id',logy=True)

            plt.savefig(path)

        return {"user_all":{"user":json.dumps(user_ratings.user_id.to_list()),
                            "item":json.dumps(user_ratings.item_id.to_list())},
                "item_all":{"user":json.dumps(item_ratings.user_id.to_list()),
                            "item":json.dumps(item_ratings.item_id.to_list())}}

    # plot reduced

    def __plot_all(self, occurencies_user, occurencies_item, path):

        # for making category of how many users/items rated/was rated
        occurence = ['1','2','3-5','6-9','10-15','15-29','30-49','50-99','100-199','200-499','500-999','>=1000']

        # added user_id column to user_ratings
        # added book_id column to item_ratings

        user_ratings = occurencies_item.reset_index().groupby('user_id').count().reset_index()
        item_ratings = occurencies_user.reset_index().groupby('item_id').count().reset_index()

        # making categories

        user_ratings["user_id"] = pd.cut(user_ratings["user_id"],[0,2,3,6,10,15,30,50,60,100,200,500,1000],labels=occurence)
        item_ratings["item_id"] = pd.cut(item_ratings["item_id"],[0,2,3,6,10,15,30,50,60,100,200,500,1000],labels=occurence)

        user_ratings = user_ratings.fillna('>=1000')
        item_ratings = item_ratings.fillna('>=1000')

        # for percentages

        item_graph = item_ratings.groupby("item_id").sum("item_id")
        users_graph = user_ratings.groupby("user_id").sum("user_id")

        if path != None:

            # count percentage

            users_graph['percentage'] = round(((users_graph.item_id / users_graph.item_id.sum()) * 100), 2)
            item_graph['percentage'] = round(((item_graph.user_id / item_graph.user_id.sum()) * 100), 2)

            fig, ax = plt.subplots(
                        nrows=1, ncols=2, constrained_layout=True, figsize=(30, 10)
                    )

            # here we plot 2 graphs first show how many users rated item
            # in second how many books was rated by user

            ax = DataSet.__set_text_on_slopes(ax, users_graph, item_graph)

            plt.savefig(path)

        return {"item_reduced":json.dumps(item_graph.reset_index().user_id.to_list()),
                "user_reduced":json.dumps(users_graph.reset_index().item_id.to_list())}
    # this function will print graph of statistic ratings
    # will save 5 graphs in jpg format
    # all -> save to rating_graphs/all.jpg and shows how many item rated each user
    # reduced -> save to rating_graphs/reduced.jpg and shows how many item rated each user it is reduced
    # scatter -> save to rating_graphs/rscatter.jpg and scatter how users rated

    def ratings_graph(self, all=False, scatter=False, reduced=False, 
                    all_path=None,
                    reduced_path=None,
                    scatter_path=None):
        
        # find how many users rated each book
        # also how many books was rated by each user

        occurencies_user = self.dataset.groupby('user_id')["item_id"].count().sort_values(ascending=False)
        occurencies_item = self.dataset.groupby('item_id')["user_id"].count().sort_values(ascending=False)

        if scatter and all and reduced:
            tmp_dict = self.__plot_all(occurencies_user, occurencies_item, None)
            tmp_dict.update(self.__plot_scatter(occurencies_item, None))
            tmp_dict.update(self.__plot_reduced(occurencies_user, occurencies_item, None))
            return tmp_dict

        if scatter:
            return self.__plot_scatter(occurencies_item, scatter_path)

        if all:
            return self.__plot_all(occurencies_user, occurencies_item, all_path)
        
        if reduced:
            return self.__plot_reduced(occurencies_user, occurencies_item, reduced_path)

        # this function will find optimal hyperparameters for chosen algorhitm usin library bayes_opt

    # this function sets parameters for alghoritm which train model als

    def __bsl_options_als(self, learning_rate, evaluations):
        
        return {'method': "als",
               'n_epochs': evaluations,
               'learning_rate': learning_rate }
    
    # this function sets parameters for alghoritm which train model sgd

    def __bsl_options_sgd(self, learning_rate, evaluations):
        
        return {'method': "sgd",
               'n_epochs': evaluations,
               'learning_rate': learning_rate }

    # this function using lib bazesian find best suited params for alghoritms

    def optimizer(self, alg="sgd"):

        def BO_func(k, learning_rate):

            k = k.astype(int)

            if alg == "sgd":
                algo = BaselineOnly(bsl_options= self.__bsl_options_sgd(learning_rate,k))
            elif alg == "als":
                algo = BaselineOnly(bsl_options= self.__bsl_options_als(learning_rate,k))
            else:
                algo = SVD(n_epochs=k, lr_all=learning_rate) 

            start = time.time()
            algo.fit(self.__train)
            end = time.time()
            predictions = algo.test(self.__test)
            self.__times.append(end-start)

            return accuracy.rmse(predictions)

        optimizer = BayesianOptimization(
                        f = BO_func,
                        pbounds = self.__tuning_params,
                        verbose = 5,
                        random_state = 5, 
                )

        optimizer.maximize(
                            init_points = 5,
                            n_iter = 5, 
                        )
        
        return optimizer
    
    

    def clear_hyperparams_result(self):
        tmp_dict = {"steps":None, "lr":None, "time":None}
        self.hyperparams_results = {"als":tmp_dict.copy(), "sgd":tmp_dict.copy(), "svd":tmp_dict.copy()}

        
    # this function call function for finding best parameters and take result from them 
    # to the frontend

    def find_hyperparams(self, alg="sgd", rating_from=0.0, rating_to=5.0):

        if self.__surprise_dataset == None:
            self.__create_sp_dtst()

        if self.__train == None or self.__test == None:
            self.__train, self.__test = train_test_split(self.__surprise_dataset, 0.1)

        self.clear_hyperparams_result()

        if alg == "all" or alg == "sgd":
            self.__times = []
            opt = self.optimizer()
    
            self.__values_sgd = [dict(opt.res[i], **{'time':self.__times[i]}) for i in range(len(opt.res))]

            #"item_reduced":json.dumps(item_graph.reset_index().user_id.to_list()
            self.hyperparams_results["sgd"]["steps"] = json.dumps([[self.__values_sgd[i]["target"],self.__values_sgd[i]["params"]["k"]] for i in range(len(self.__values_sgd))])
            self.hyperparams_results["sgd"]["lr"] = json.dumps([[self.__values_sgd[i]["target"],self.__values_sgd[i]["params"]["learning_rate"]] for i in range(len(self.__values_sgd))])
            self.hyperparams_results["sgd"]["time"] = json.dumps([[self.__values_sgd[i]["target"],self.__values_sgd[i]["time"]] for i in range(len(self.__values_sgd))])

        
        if alg == "all" or alg == "als":
            self.__times = []
            opt = self.optimizer(alg="als")
     
            self.__values_als = [dict(opt.res[i], **{'time':self.__times[i]}) for i in range(len(opt.res))]
       
            self.hyperparams_results["als"]["steps"] = json.dumps([[self.__values_als[i]["target"],self.__values_als[i]["params"]["k"]] for i in range(len(self.__values_als))])
            self.hyperparams_results["als"]["lr"] = json.dumps([[self.__values_als[i]["target"],self.__values_als[i]["params"]["learning_rate"]] for i in range(len(self.__values_als))])
            self.hyperparams_results["als"]["time"] = json.dumps([[self.__values_als[i]["target"],self.__values_als[i]["time"]] for i in range(len(self.__values_als))])

        if alg == "all" or alg == "svd":
            self.__times = []
            opt = self.optimizer(alg="svd")
      
            self.__values_svd = [dict(opt.res[i], **{'time':self.__times[i]}) for i in range(len(opt.res))]


            self.hyperparams_results["svd"]["steps"] = json.dumps([[self.__values_svd[i]["target"],self.__values_svd[i]["params"]["k"]] for i in range(len(self.__values_svd))])
            self.hyperparams_results["svd"]["lr"] = json.dumps([[self.__values_svd[i]["target"],self.__values_svd[i]["params"]["learning_rate"]] for i in range(len(self.__values_svd))])
            self.hyperparams_results["svd"]["time"] = json.dumps([[self.__values_svd[i]["target"],self.__values_svd[i]["time"]] for i in range(len(self.__values_svd))])

        return self.hyperparams_results
    
    def __plot_methods(self, to_plot, type="k", ax=None):

        colors = ["green", "blue", "black", "red", "orange", "yellow"]

        for i in range(0, len(to_plot)):

            if type == "time":
                y_axis = [a_dict["time"] for a_dict in to_plot[i]["data"]]
            else: 
                y_axis = [a_dict["params"][type] for a_dict in to_plot[i]["data"]]    

            ax.scatter(    y_axis,
                            [a_dict["target"] for a_dict in to_plot[i]["data"]] ,
                            c=colors[i], 
                            label=to_plot[i]["alg"])

        ax.set_ylabel("rmse")

        if type == "k":
            ax.set_xlabel("number of evaluations")
        elif type == "time":
            ax.set_xlabel("Time in [s]")
        else :
            ax.set_xlabel("learning rate")

        ax.legend()


    def __call_plots_methods(self, to_plot=None, evaluations=False, lr=False, time=False, save_path=None):
        
        what_plot = []
        min = None
        if lr:
            what_plot.append({'type':"learning_rate"})
        if evaluations:

            what_plot.append({'type':"k"})
        if time:
            what_plot.append({'type':"time"})
            
        figure, axes = plt.subplots(
                            nrows=1, ncols=len(what_plot), constrained_layout=True,
                            figsize=(10, 5)
                            )  
          
        for i in range(0, len(what_plot)):
            self.__plot_methods(to_plot=to_plot, type=what_plot[i]["type"], ax=axes[i])

        if save_path != None:
            plt.savefig(save_path)


    def graphs_alghoritms(self, all=False, sgd=False, svd=False, als=False,
                          values_sgd = None, values_als = None, values_svd = None,
                          eval=False, lr=False, time=False, save_path=None, print=False):

        to_plot = []
    
        if (all or als) and values_als == None and self.__values_als == None:
            print("an error occured\n")
            return None
        elif all or als:

            if values_als == None:
                values_als = self.__values_als
        
            to_plot.append({"alg":"als","data":values_als})

            

        if (all or svd) and values_svd == None and self.__values_svd == None:
            print("an error occured\n")
            return None
        elif all or svd:

            if values_svd == None:
                values_svd = self.__values_svd
        
            to_plot.append({"alg":"svd","data":values_svd})

        if (all or sgd) and values_sgd == None and self.__values_sgd == None:
            print("an error occured\n")
            return None
        
        elif all or sgd:

            if values_sgd == None:
                values_sgd = self.__values_sgd
        
            to_plot.append({"alg":"sgd","data":values_sgd})

        self.__call_plots_methods(to_plot=to_plot,evaluations=eval, lr=lr, time=time, save_path=save_path)

    def train_model_svd(self, lr=0.0005, steps=10):

        if self.__to_train is None:
            return None
        
        if self.__surprise_dataset is None:
            self.__create_sp_dtst()

        train = self.__surprise_dataset.build_full_trainset()

        algo = SVD(n_epochs=steps, lr_all=lr)
        return algo.fit(train)

    def train_model_als(self, lr=0.0005, steps=10):

        if self.__to_train is None:
            return None
        
        bsl_options = {'method': 'als',
                        'learning_rate': lr,
                        'n_epochs': steps,
        }
        
        algo = BaselineOnly(bsl_options=bsl_options)

        if self.__surprise_dataset is None:
            self.__create_sp_dtst()

        train = self.__surprise_dataset.build_full_trainset()

        return algo.fit(trainset=train)
    
    def train_model_sgd(self, lr=0.0005, steps=10):

        if self.__to_train is None:
            return None

        bsl_options = {'method': 'sgd',
                        'learning_rate': lr,
                        'n_epochs': steps,
        }
        
        algo = BaselineOnly(bsl_options=bsl_options)

        if self.__surprise_dataset is None:
            self.__create_sp_dtst()

        train = self.__surprise_dataset.build_full_trainset()

        return algo.fit(trainset=train)

    # this function creates from given dataset surprise dataset   
    
    def __create_sp_dtst(self):
        reader = Reader(rating_scale=(self.dataset["rating"].min(),self.dataset["rating"].max()))
        if self.__to_train is None:
            self.__to_train = self.__name_and_id_item

        tmp = self.__to_train[["user_id", "item_id", "rating"]]
        self.__surprise_dataset = Dataset.load_from_df(df=tmp, reader=reader)

    def get_items(self):
        if self.__name_and_id_item is not None:
            x = self.__name_and_id_item.groupby(["item_id","name"]).count()
            x = x.loc[x["rating"] > 5].reset_index()
            x = x.sample(n=100)
            
            return self.dataset["rating"].min(), self.dataset["rating"].max(), x[["item_id", "name"]].values.tolist()

    def get_items_genre(self):
        if self.__name_and_id_item is not None:
            x = self.__name_and_id_item.groupby(["item_id","name","genre"]).count()
            x = x.loc[x["rating"] > 5].reset_index()
            x = x.sample(n=100)
            
            return self.dataset["rating"].min(), self.dataset["rating"].max(), x[["item_id", "name", "genre"]].values.tolist()

    def make_dat_with_name(self, data_to_add):
        if self.__user_id is None:
            self.__user_id = self.__name_and_id_item["user_id"].max() + 1
        if self.__to_train is None:
            self.__to_train = self.__name_and_id_item

        if self.genre:
            dict = {'user_id': [], 'item_id': [], 
                    'name': [], 'rating': [], 'genre':[]}
        else:
            dict = {'user_id': [], 'item_id': [], 
                    'name': [], 'rating': []}
        for key in data_to_add:
            dict['user_id'].append(self.__user_id)
            dict['item_id'].append(key)
            dict['rating'].append(data_to_add[key][1])
            dict['name'].append(data_to_add[key][0])
            if self.genre:
                dict['genre'].append(data_to_add[key][2])
            
        tmp_df = pd.DataFrame(dict)
        self.__to_train = pd.concat([self.__to_train, tmp_df], ignore_index = True, axis = 0)   

    def find_predictions(self,path, model=None, alg=None):
        user_reviews = self.__to_train.loc[self.__to_train["user_id"] == self.__user_id]["item_id"].to_list()

        without_user_reviews = self.__to_train[~self.__to_train["item_id"].isin(user_reviews)][["item_id","name"]].drop_duplicates()

        # if dataset is tooo big it brings just first 1M elements

        without_user_reviews =  without_user_reviews.head(1000000)

        without_user_reviews["predictions"] = without_user_reviews.apply(
                            lambda row: model.predict(uid=self.__user_id,iid=row["item_id"]).est, axis=1)

        without_user_reviews.to_pickle("predictions_pkl")
        top_10 = without_user_reviews.sort_values(by="predictions", ascending=False).head(10)
        rated = self.make_stats_predictions(without_user_reviews,path)

        top_10 = [[a]+[b] for a,b in zip(top_10.name.to_list() ,top_10.predictions.to_list())]

        rated_graph_ranges = json.dumps(rated.reset_index().predictions.astype("str").to_list())

        rated_graph_values = json.dumps(rated.reset_index().name.to_list())

        rated = [[a]+[b] for a,b in zip(rated.reset_index().predictions.astype("str").to_list() ,rated.item_id.to_list())]
        
     
        self.trained_result_dict[alg] = {"top_10":top_10,"rated":rated,"ranges":rated_graph_ranges,"values":rated_graph_values}
        #print(self.trained_result_dict[alg])

    def find_predictions_genre(self,path, model=None, alg=None):
        user_reviews = self.__to_train.loc[self.__to_train["user_id"] == self.__user_id]["item_id"].to_list()

        without_user_reviews = self.__to_train[~self.__to_train["item_id"].isin(user_reviews)][["item_id","name","genre"]].drop_duplicates()

        # if dataset is tooo big it brings just first 1M elements

        without_user_reviews =  without_user_reviews.head(1000000)

        without_user_reviews["predictions"] = without_user_reviews.apply(
                            lambda row: model.predict(uid=self.__user_id,iid=row["item_id"]).est, axis=1)

        
        top_10 = without_user_reviews.sort_values(by="predictions", ascending=False).head(10)
        rated = self.make_stats_predictions(without_user_reviews,path)
        without_user_reviews.to_pickle("predictions_pkl")

        genres_to_hist, values_to_hist = self.get_data_for_genre_hist(without_user_reviews)

        top_10 = [[a]+[c]+[b] for a,b,c in zip(top_10.name.to_list() ,top_10.predictions.to_list(),top_10.genre.to_list())]

        rated_graph_ranges = json.dumps(rated.reset_index().predictions.astype("str").to_list())

        rated_graph_values = json.dumps(rated.reset_index().name.to_list())

        rated = [[a]+[b] for a,b in zip(rated.reset_index().predictions.astype("str").to_list() ,rated.item_id.to_list())]
        
     
        self.trained_result_dict[alg] = {"top_10":top_10,"rated":rated,"ranges":rated_graph_ranges,"values":rated_graph_values,
                                         "genres_hist":genres_to_hist, "values_hist":values_to_hist}

    def make_stats_predictions(self, dataset, filename):
        ranges = []
        fig = plt.figure()
        ax = fig.add_subplot()
        min_range = math.floor(dataset["predictions"].min())
        max_range = math.ceil(dataset["predictions"].max()) + 1

        for i in range(min_range,max_range):

            if i != min_range:
                ranges.append(i - 0.5)

            ranges.append(i)

        dataset["predictions"] = pd.cut(dataset.predictions, bins=ranges, right=False)

        dataset = dataset.groupby("predictions").count()

        #ax = dataset.item_id.plot.bar()

        #ax.set_xlabel("range of prediction")
        #ax.set_ylabel("number of items")
        #plt.tight_layout()
        
        #fig.savefig(filename, figsize=(100, 100))

        return dataset

    def get_data_to_render_result(self, alg=None):
        if self.trained_result_dict[alg] is None:
            return None
        else:
            return self.trained_result_dict[alg]

    def save(self, file_handler):
        with open(file_handler, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def get_counts(self):
        ratings = []
        counts = []
        dc = self.dataset["rating"].value_counts().to_dict()

        print(dc)

        for key in sorted(dc):
            ratings.append(key)
            counts.append(dc[key])

        return ratings, counts

    def get_tmp(self):
        return  self.hyperparams_results

    def get_data_for_genre_hist(self,data) :
        data.loc[data["genre"] == None, 'genre'] = 'No genre'
        data = data.groupby(["predictions","genre"]).count().reset_index()

        ### split !!!!!!!!!!!!!!
        ranges_values  = [list([p,a,c] for a in g.split(self.sep)) for p,g,c in zip(data.reset_index().predictions.astype("str").to_list(),data.genre.astype("str").to_list(),data.item_id.to_list())]

        merged = list(itertools.chain.from_iterable(ranges_values))

        dict = {'range': [], 'genre': [], 
        'val': []}
        for range, genre, value in merged:
            dict["range"].append(range)
            dict["genre"].append(genre)
            dict["val"].append(value)

        tmp_df = pd.DataFrame(dict)
        a = tmp_df.groupby(["range","genre"]).sum().reset_index()
        x = pd.pivot_table(a,columns="genre",values="val",index="range")

        values = [a[1:] for a in x.reset_index().values.tolist()] 
        genres = x.columns.to_list()
        
        return   json.dumps(genres), values
