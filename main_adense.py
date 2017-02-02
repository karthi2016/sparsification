import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import os.path
import pickle
import sys
import operator
import pqueue
import random

def plotADense(G, takenEdges, sets):
    plt.figure()
    pos = nx.spring_layout(G)

    c = 0
    for i in sets.values():
        elist = []
        colors = "bgrcmyk"   
        for i1 in i:
            for i2 in i:
                if G.has_edge(i1,i2):
                    elist.append((i1,i2))
        nx.draw_networkx_edges(G, pos, edgelist = elist, edge_color=colors[np.mod(c,len(colors))], width='7', alpha = 0.2)
        c += 1
    
    nx.draw_networkx_edges(G, pos, edgelist = takenEdges, width='2')
    nx.draw_networkx_nodes(G, pos)
    plt.show()


def plotInit(G, initSets, name):
    plt.figure()
    pos=nx.spring_layout(setsG)

    c = 0
    for i in initSets.values():
        elist = []
        colors = "bgrcmyk"   
        for i1 in i:
            for i2 in i:
                elist.append((i1,i2))
        nx.draw_networkx_edges(setsG, pos, edgelist=elist, edge_color=colors[np.mod(c,len(colors))], width='7', alpha = 0.2)
        c += 1
    #plt.savefig(name + '.pdf')

def getDataGraph(sets):
    G = nx.Graph()
    Gteams = {}
    for k,v in sets.iteritems():
        g = nx.Graph()
        for i in v:
            for j in v:
                if i != j:
                    G.add_edge(i,j)
                    g.add_edge(i,j)
        Gteams[k] = g
        
    G = nx.Graph(G)
    G.remove_edges_from(G.selfloop_edges())
    return G, Gteams

    
def calcMaxEdges(sets):
    maxEdges = {}
    for team, edges in sets.iteritems():
        nodes = set()
        for i in edges:
            nodes.add(i[0])
            nodes.add(i[1])
        n = len(nodes)
        maxEdges[team] = n*(n-1)/2.0       
    return maxEdges
    
def updateEdges(edges, e, teams, Gteams):
    del edges[e]
    for team in teams:
        for e in edges:
            edges[e].remove(team)
    return edges
    
def getGraph(out):
    G = nx.Graph()
    for k,v in out.iteritems():
        for j in v:
            G.add_edge(*j)
            
    G = nx.Graph(G)
    G.remove_edges_from(G.selfloop_edges())
    return G

def getEdgeTeamsDict(sets):
    teams = {}
    for i, edges in sets.iteritems():
        for e in edges:
            if e not in teams.keys():
                teams[e] = set()
            teams[e].add(i)
    return teams
    
def getTeamEdgesDict(teams):
    edges = {}
    for edge, ts in teams.iteritems():
        for t in ts:            
            if t not in edges.keys():
                edges[t] = set()            
            edges[t].add(edge)
    return edges
    
def greedy_runFromGiven(alpha, edges, teams, maxEdges, output, takenEdges):
    
    #output = {i: set() for i in edges.keys()}
    coveredSets = set()
    scores = {}
    
    for t in edges.keys():        
        if maxEdges[t]*alpha <= len(output[t]):
            coveredSets.add(t)                
    
    for e, ts in teams.iteritems():
        scores[e] = -len(ts - coveredSets)
    orderByScores = pqueue.priority_dict(scores)
    
    tic = time.time()
    while orderByScores:
        if len(coveredSets) == len(edges):
            break
        edge, score = orderByScores.pop_smallest()
        
        if score == 0:
            #print 'could not cover'
            break
        takenEdges.add(edge)
        
        for t in (teams[edge] - coveredSets):
            output[t].add(edge)
            if maxEdges[t]*alpha <= len(output[t]):
                coveredSets.add(t)                
                for item in edges[t] - takenEdges:
                    orderByScores[item] = -len(teams[item] - coveredSets)
        
    return takenEdges, output 

def trivialRandom(epsilon, edges, teams, numEdges):
    
    output = {i: set() for i in edges.keys()}
    coveredSets = set()
    takenEdges = set()

    tic = time.time()
    edges_list = set()
    for i in edges.values():
        edges_list |= set(i)
    edges_list = list(edges_list)
    random.shuffle(edges_list)
    while len(coveredSets) != len(edges):
        edge = edges_list.pop()
        
        takenEdges.add(edge)
        for t in (teams[edge] - coveredSets):
            output[t].add(edge)
            if np.floor(numEdges[t]*epsilon) <= len(output[t]): 
                coveredSets.add(t)
    
    return takenEdges, output

def greedy(epsilon, edges, teams, numEdges, similarity = {}):
    
    output = {i: set() for i in edges.keys()}
    coveredSets = set()
    takenEdges = set()
    
    tic = time.time()
    scores = {}
    for e, ts in teams.iteritems():
        if similarity:
            scores[e] = -np.divide(len(ts),(1.0 - similarity[e]))
        else:
            scores[e] = -len(ts)
    orderByScores = pqueue.priority_dict(scores)
    
    tic = time.time()
    while orderByScores:
        if len(coveredSets) == len(edges):
            break
        edge, score = orderByScores.pop_smallest()
        
        if score == 0:
            break
        takenEdges.add(edge)
        
        for t in (teams[edge] - coveredSets):
            output[t].add(edge)
            if np.floor(numEdges[t]*epsilon) <= len(output[t]): 
                coveredSets.add(t)
                for item in edges[t] - takenEdges:
                    orderByScores[item] = -len(teams[item] - coveredSets)
    return takenEdges, output 


if __name__ == '__main__':

    #name = sys.argv[1]
    #epsilon = float(sys.argv[2])
    dataset = 'lastFM_tags'
    epsilon = 0.5
    
    G, sets = pickle.load(open(os.path.join('UnderlyingNetwork', dataset +'.pkl'),'rb'))
            
    sets_edges = {}
    numEdges = {}
    nodes = set()
    similarity = {}
    
    Gind = nx.Graph()
       
    for i, j in sets.iteritems():        
        t = G.subgraph(j)
        Gind.add_edges_from(t.edges())
        nodes.update(set(j))
        if t.number_of_edges() > 0:
            numEdges[i] = t.number_of_edges()
            sets_edges[i] = set()
        for e in t.edges():
            if e[0] < e[1]:
                edge = e                
            else:
                edge = (e[1], e[0])                
            sets_edges[i].add(edge)
            
                    
    Gind = nx.Graph(Gind)
    Gind.remove_edges_from(Gind.selfloop_edges())    
    
    teams = getEdgeTeamsDict(sets_edges)
    takenEdges, output = greedy(epsilon, sets_edges, teams, numEdges)
    takenEdgesTriv, outputTriv = trivialRandom(epsilon, sets_edges, teams, numEdges)
    
    G_out = nx.Graph()
    G_out.add_nodes_from(G.nodes())
    G_out.add_edges_from(list(takenEdges))
    
    G_out_sim = nx.Graph()
    
    G_triv = nx.Graph()
    G_triv.add_nodes_from(G.nodes())
    G_triv.add_edges_from(list(takenEdgesTriv))    
    
    print 'our method:'
    print 'fraction of induced edges', 1.0*G_out.number_of_edges()/Gind.number_of_edges()
    print 'fraction of all edges', 1.0*G_out.number_of_edges()/G.number_of_edges()
    print
    print 'baseline method:'
    print 'fraction of induced edges', 1.0*G_triv.number_of_edges()/Gind.number_of_edges()
    print 'fraction of all edges',1.0*G_triv.number_of_edges()/G.number_of_edges()
   
