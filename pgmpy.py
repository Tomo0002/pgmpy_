import pandas as pd
from pgmpy.estimators import ParameterEstimator
# from pgmpy.inference import VariableElimination
# import numpy as np
# from pgmpy.estimators import K2Score, HillClimbSearch, ExhaustiveSearch
# from pgmpy.sampling import BayesianModelSampling
# from networkx.generators import random_clustered
# import os
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_boston
# import networkx as nx

from pgmpy.models import BayesianModel
# from pgmpy.estimators import ConstraintBasedEstimator
# from pgmpy.estimators import BicScore

# from pyvis.network import Network


# 本モデル
model = BayesianModel([('Stress','Transport'), ('Tiredness','Transport'),('Accustomed_to_driving', 'Transport'),('Not_accustomed_to_driving', 'Transport')])
data_2 = pd.DataFrame(data={'Stress': ['0.8', '0.9', '0.7','0.2','0.1','0.4','0.3','1','0.5','0.5','0.7','0.8'],
                          'Tiredness': ['0.2', '0.1', '0.3','0.8','0.9','0.6','0.7','0','0,5','0.5','0.3','0.2'],
                          'Accustomed_to_driving':['0.2', '0.3', '0.5' ,'0.6','0.6','0.7','0.7','0.0','0.6','0.4','0.7','0.3'],
                          'Not_accustomed_to_driving':['0.8', '0.7', '0.5' ,'0.4','0.4','0.3','0.3','1','0.4','0.6','0.3','0.7'],
                          'Transport': ['Go', 'Go','Go','Not','Not','Not','Not','Go','think','think','Go','Go']})

model.fit(data_2)
estimator = ParameterEstimator(model, data_2)
print(estimator.state_counts('Accustomed_to_driving'))
print(estimator.state_counts('Not_accustomed_to_driving'))
print(estimator.state_counts('Stress'))
print(estimator.state_counts('Transport'))

# edges = pd.DataFrame({'source': [0, 1, 2],
#                'target': [2, 2, 3],
#                'weight':[1,1,1]})


# G = nx.from_pandas_edgelist(edges,edge_attr=True,create_using=nx.DiGraph())
# G = nx.from_pandas_edgelist(edges, edge_attr=True)
# eigen_centrality = nx.eigenvector_centrality_numpy(G)
# bet_centrality = nx.betweenness_centrality(G)
# degree_centrality = nx.degree_centrality(G)
# degree_coefficient = nx.clustering(G)

# pyvis_G = Network()
# pyvis_G.from_nx(G)
# pyvis_G.toggle_physics(True)  #html上でレイアウト動かしたくない場合false
# pyvis_G.show("mygraph.html")