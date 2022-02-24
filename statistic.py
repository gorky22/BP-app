import pandas as pd
import matplotlib as plt
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

    def __init__(self, dataset, user, item, rating):

        self.dataset = dataset.rename(columns={user:"user_id", item:"item_id", rating:"rating"})


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

        plt.figure(figsize=(x/500, y/500))

        plt.gca().set_aspect('auto', adjustable='box')
        plt.spy(self.__interaction_matrix.T,markersize=0.009,precision = 0.1)
        plt.xlabel("id of users")
        plt.ylabel("id of items")
        
        if fig_location != None:
            plt.savefig(fig_location,  bbox_inches='tight')
    
        if show_figure:
            plt.show()

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

        ax = tmp.plot.scatter(x='item_id',y="number_of_ratings",c='number_of_ratings',
                            cmap="RdYlGn",logy=True,s=2,sharex=False)

        ax.set_xlabel("number of users")
        ax.set_ylabel("number of ratings")

        plt.savefig(path)
    
    # plot reduced

    def __plot_reduced(self, occurencies_user, occurencies_item, path):

        # added user_id column to user_ratings
        # added book_id column to item_ratings

        user_ratings = occurencies_item.reset_index().groupby('user_id').count().reset_index()
        item_ratings = occurencies_user.reset_index().groupby('item_id').count().reset_index()

        fig_all, ax_all = plt.subplots(
                    nrows=1, ncols=2, constrained_layout=True, figsize=(30, 10)
                )

        user_ratings.plot.bar(ax=ax_all[0], x='user_id',y='item_id',logy=True)
        item_ratings.plot.bar(ax=ax_all[1], x='item_id',y='user_id',logy=True)

        plt.savefig(path)

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

    # this function will print graph of statistic ratings
    # will save 5 graphs in jpg format
    # all -> save to rating_graphs/all.jpg and shows how many item rated each user
    # reduced -> save to rating_graphs/reduced.jpg and shows how many item rated each user it is reduced
    # scatter -> save to rating_graphs/rscatter.jpg and scatter how users rated

    def ratings_graph(self, all=False, scatter=False, reduced=False, 
                    all_path="all.jpg",
                    reduced_path="reduced.jpg",
                    scatter_path="scatter.jpg"):
        
        # find how many users rated each book
        # also how many books was rated by each user

        occurencies_user = self.dataset.groupby('user_id')["item_id"].count().sort_values(ascending=False)
        occurencies_item = self.dataset.groupby('item_id')["user_id"].count().sort_values(ascending=False)

        if scatter:
            self.__plot_scatter(occurencies_item, scatter_path)

        if all:
            self.__plot_all(occurencies_user, occurencies_item, all_path)
        
        if reduced:
            self.__plot_reduced(occurencies_user, occurencies_item, reduced_path)

        # this function will find optimal hyperparameters for chosen algorhitm usin library bayes_opt

    def __bsl_options_als(self, learning_rate, evaluations):
        
        return {'method': "als",
               'n_epochs': evaluations,
               'learning_rate': learning_rate }
    
    def __bsl_options_sgd(self, learning_rate, evaluations):
        
        return {'method': "sgd",
               'n_epochs': evaluations,
               'learning_rate': learning_rate }

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
        
    def find_hyperparams(self, alg="sgd", rating_from=0.0, rating_to=5.0):

        reader = Reader(rating_scale=(rating_from,rating_to))

        self.dataset = self.dataset[["user_id", "item_id", "rating"]]

        data = Dataset.load_from_df(df=self.dataset, reader=reader)

        if self.__train == None or self.__test == None:
            self.__train, self.__test = train_test_split(data, 0.1)

        if alg == "all" or alg == "sgd":
            self.__times = []
            opt = self.optimizer()
            self.__values_sgd = [dict(opt.res[i], **{'time':self.__times[i]}) for i in range(len(opt))]
        
        if alg == "all" or alg == "als":
            self.__times = []
            opt = self.optimizer(alg="als")
            self.__values_als = [dict(opt.res[i], **{'time':self.__times[i]}) for i in range(len(opt))]
        
        if alg == "all" or alg == "svd":

            self.__times = []
            opt = self.optimizer(alg="svd")
            self.__values_svd = [dict(opt.res[i], **{'time':self.__times[i]}) for i in range(len(opt))]
    
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


    def __call_plots_methods(self, to_plot=None, evaluations=False, lr=False, time=False):
        
        what_plot = []

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


    def graphs_alghoritms(self, all=False, sgd=False, svd=False, als=False,
                          values_sgd = None, values_als = None, values_svd = None,
                          eval=False, lr=False, time=False):

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

        self.__call_plots_methods(to_plot=to_plot,evaluations=eval, lr=lr, time=time)

        
        

        


