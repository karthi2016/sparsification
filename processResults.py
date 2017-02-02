import os.path
import pickle
import copy
import networkx as nx
import matplotlib.pyplot as plt
#import syntheticTest as synth
import sys
import numpy as np
import scipy.stats
import main_dscores_pqueue as star

def readGraph(file):  
    G = nx.Graph()
    with open(file, "r") as f:
        for line in f:
            if line[0] != '#':
                t = str.split(line.strip(),'\t')
                G.add_edge(int(t[0]),int(t[1])) 
    G = nx.Graph(G)
    G.remove_edges_from(G.selfloop_edges())
    return G
    
# def readGraphDirected(file):  
    # G = nx.DiGraph()
    # with open(file, "r") as f:
        # for line in f:
            # if line[0] != '#':
                # t = str.split(line.strip(),'\t')
                # G.add_edge(int(t[0]),int(t[1])) 
    # G.remove_edges_from(G.selfloop_edges())
    # return G

def getGraph(out, sets):
    G = nx.Graph()
    for k,v in out.iteritems():
        for j in v:
            G.add_edge(k,j)
            
    nodes = set()
    for k,v in sets.iteritems():
        nodes|=set(v)
        
    for i in nodes:
        if not G.has_node(i):
            G.add_node(i)
            
    G = nx.Graph(G)
    G.remove_edges_from(G.selfloop_edges())
    return G
    
def getGraphDense(out, sets):
    G = nx.Graph()
    for k,v in out.iteritems():
        for i in v:
            G.add_edge(*i)
            
    nodes = set()
    for k,v in sets.iteritems():
        for i in v:
            nodes.add(i[0])
            nodes.add(i[1])
        
    for i in nodes:
        if not G.has_node(i):
            G.add_node(i)
            
    G = nx.Graph(G)
    G.remove_edges_from(G.selfloop_edges())
    return G
    
# def getGraphToCompare(Gstar, G, sets):
    # Gprime = copy.deepcopy(G)
    # for k,v in sets.iteritems():
        # for i in v:
            # for j in v:
                # if Gprime.has_edge(i,j):
                    # Gprime.remove_edge(i,j)
    # #print Gstar.edges()
    # Gprime.add_edges_from(Gstar.edges())
    # #Gprime = nx.union(Gprime, Gstar)                
    
    # Gprime = nx.Graph(Gprime)
    # Gprime.remove_edges_from(Gprime.selfloop_edges())
    # return Gprime
   
def getGraphToCompareStars(G, sets):
    nodes = set()
    for k,v in sets.iteritems():
        nodes|=set(v)
        
    Gprime = G.subgraph(nodes)
    return Gprime
    
def getGraphToCompareDense(G, sets):
    nodes = set()
    for k,v in sets.iteritems():
        for i in v:
            nodes.add(i[0])
            nodes.add(i[1])
        
    Gprime = G.subgraph(nodes)
    return Gprime
    
def getGraphDirected(out):
    G = nx.DiGraph()
    for k,v in out.iteritems():
        for j in v:
            G.add_edge(k,j)
            
    G.remove_edges_from(G.selfloop_edges())
    return G
    
def getDiam(G):
    d = 0.0
    for i in nx.connected_component_subgraphs(G):
        if i.number_of_nodes() > 1:
            d = max(d, nx.diameter(i))
    return d
    
def getMappedMaxPath(G, GTG):
    d = []
    # CC = list(nx.connected_component_subgraphs(G))
    # Gcc=sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
    # G0=Gcc[0]
    
    diam, daimn1, daimn2 = 0.0, 0, 0
    for i in nx.connected_component_subgraphs(G):
        t = nx.all_pairs_shortest_path_length(i)
        nodes = i.nodes()
        for i1 in xrange(len(nodes)):
            for i2 in xrange(i1+1, len(nodes)):
                n1, n2 = nodes[i1], nodes[i2]
                if t[n1][n2] > diam:
                    diam = t[n1][n2]
                    daimn1, daimn2 = n1, n2
    mappedDiam = nx.shortest_path_length(GTG, n1, n2)
    return diam, mappedDiam
    
def getMeanMedMaxAvgPath(G):
    d = 0.0
    c = 0.0
    l = []
    for i in nx.connected_component_subgraphs(G):
        if i.number_of_nodes() > 1:
            l.append(nx.average_shortest_path_length(i))
    #print l
    return np.mean(l), np.max(l), np.median(l)
    
def connComp(G):
    c = 0.0
    for i in nx.connected_component_subgraphs(G):
        if i.number_of_nodes() > 1:
            c += 1
    return c
    
def getMedPath(G):
    d = []
    for i in nx.connected_component_subgraphs(G):
        t = nx.all_pairs_shortest_path_length(i)
        for v in t.values():
            d += v.values()
    return np.median(d)
    
def sortByDegree(G):
    n,d = [],[]
    for k,v in G.degree_iter():
        n.append(k)
        d.append(v)    
    idx = np.argsort(d)[::-1]    
    return [n[i] for i in idx]
    
def sortByPR(G):
    n,d = [],[]
    #for k,v in nx.pagerank(G).iteritems():
    #for k,v in nx.pagerank_numpy(G).iteritems():
    for k,v in nx.pagerank_scipy(G).iteritems():
        n.append(k)
        d.append(v)    
    idx = np.argsort(d)[::-1]    
    return [n[i] for i in idx]

def accuracy(Gprime, Gstar):
    inter = 0.0
    for e in Gstar.edges_iter():
        if Gprime.has_edge(*e):
            inter += 1.0
    prec = inter/Gstar.number_of_edges()
    recall = inter/Gprime.number_of_edges()
    F1 = 2.0/(1.0/prec + 1.0/recall)
    return prec, recall, F1
    
def getAvgPath(G):
    d = 0.0
    c = 0.0
    for i in nx.connected_component_subgraphs(G):
        if i.number_of_nodes() > 1:            
            d += nx.average_shortest_path_length(i)
            c += 1
    d /= c
    return d
    
# def effDiam(G):
    # p = []
    # for i in nx.connected_component_subgraphs(G):
        # nodes = i.nodes()
        # lengths = []
        # t = nx.all_pairs_shortest_path_length(i)
        # for i1 in xrange(len(nodes)):
            # for i2 in xrange(i1+1,len(nodes)):
                # n1,n2 = nodes[i1],nodes[i2]
                # lengths.append(t[n1][n2])
        # if lengths:
            # p.append(np.percentile(lengths, 90))
    # return max(p)
    
def paths(G):
    p = []
    lengths = []
    for i in nx.connected_component_subgraphs(G):      
        for n in i.nodes_iter():
            lengths += nx.shortest_path_length(i,n).values()           
    return np.percentile(lengths,90), np.median(lengths)
    
def getEdgesInSets(G, sets):
    nodes = set()
    edges = set()
    density = []
    for i, j in sets.iteritems():        
        t = G.subgraph(j)
        nodes.update(set(j))
        density.append(nx.density(t))
        #alpha[i] = nx.density(t)*epsilon
        
        for e in t.edges():
            if e[0] < e[1]:
                edge = e
                #sets_edges[i].add(e)
            else:
                edge = (e[1], e[0])
                #sets_edges[i].add((e[1], e[0]))
            edges.add(edge)
    G = G.subgraph(nodes)
    return G, len(edges), np.mean(density)
    
if __name__ == '__main__': 
    
    #dataset = 'bookmarks_tags'
    dataset = 'BM_tags'
    print dataset 
    
    if 'DBLP' in dataset:
        if 'KDD' in dataset:    
            file_network = os.path.join('dblp', 'author-author-wpapers.KDD_all.txt')    
            file_labels = os.path.join('dblp', 'author-label-count.KDD_all.txt')    
        else:
            file_network = os.path.join('dblp', 'author-author-wpapers.ICDM_all.txt')    
            file_labels = os.path.join('dblp', 'author-label-count.ICDM_all.txt')    
        G = read_dblp_network(file_network)
        sets = read_dblp_sets(file_labels)
        
    elif 'FB' in dataset:
        file_network = os.path.join('FB', 'facebook_combined.txt')
        G = get_ego_network(file_network)
        if 'circles' in dataset:
            sets = get_ego_circles('FB')
        else:
            sets, _ = get_ego_features('FB')
            
    elif 'lastFM' in dataset:
        G = get_lastfm_network()        
        sets_tags, sets_artists = get_lastfm_tags()        
        if 'tag' in dataset:
            sets = sets_tags
        else:
            sets = sets_artists
    
    elif 'BM' in dataset:
        G = get_bookmarks_network()        
        sets_tags, sets_bookmarks = get_bookmarks_tags()        
        if 'tag' in dataset:
            sets = sets_tags
        else:
            sets = sets_bookmarks
    sets = removeStandalones(sets)
    sets = removeSameTeams(sets)
    
    dumpname = dataset + '.pkl'
    pickle.dump((G, sets), open(dumpname, 'wb'))
    exit()
    
    
    
    
    G, num_inset_edges, avg_setdensity = getEdgesInSets(G, sets)
    
    print G.number_of_nodes(), num_inset_edges, G.number_of_edges(), nx.density(G)
    print np.mean(G.degree().values()), max(G.degree().values()), nx.number_connected_components(G)
    print paths(G), avg_setdensity 
    
    print len(sets), min([len(i) for i in sets.values()]), np.mean([len(i) for i in sets.values()])
    items = star.getItemIndex(sets)
    print max([len(i) for i in items.values()]), np.mean([len(i) for i in items.values()])
    exit()
    print G.number_of_edges()
    numNodes, numEdges, diam, effectDiam, avgPath, medianPath, rankDegree, rankPR, clCoef = [],[],[],[],[],[],[],[],[]
    connC, meanAvgPath, medAvgPath, maxAvgPath, diamReal, diamMapped = [], [], [], [], [], []
    connCBL, meanAvgPathBL, medAvgPathBL, maxAvgPathBL,diamRealBL, diamMappedBL = [], [], [], [], [], []
    connCG, meanAvgPathG, medAvgPathG, maxAvgPathG = [], [], [], []
    numNodesBL, numEdgesBL, diamBL, effectDiamBL, avgPathBL, medianPathBL, rankDegreeBL, rankPRBL, clCoefBL = [],[],[],[],[],[],[],[],[]
    numNodesG, numEdgesG, diamG, effectDiamG, avgPathG, medianPathG, clCoefG = [],[],[],[],[],[],[]
    prec, recall, Fm = [],[],[]
    precBL, recallBL, FmBL = [],[],[]
    alphaAvg = []
    GStatsK = []
    for k in K:
        print k,d
        name = problem +'_' + str(k) + '_' + d + '.pkl'
        
        if mode == 'stars':            
            output, outputBL, sets = pickle.load(open(os.path.join('resIndSet', name), 'rb'))
            Gprime = getGraphToCompareStars(G, sets)
            starGBL = getGraph(outputBL, sets)
            starG = getGraph(output, sets)
        else:
            output, outputBL, sets, alphas = pickle.load(open(os.path.join('resIndSet', name), 'rb'))
            Gprime = getGraphToCompareDense(G, sets)
            alphaAvg.append(1.0*sum(alphas.values())/len(alphas))
            starGBL = getGraphDense(outputBL, sets)
            starG = getGraphDense(output, sets) 

            
        if stat == 'edges':                
            numEdges.append(starG.number_of_edges())
            numEdgesBL.append(starGBL.number_of_edges())
            numEdgesG.append(Gprime.number_of_edges())                
        
        elif stat == 'diam':
            diam.append(getDiam(starG))
            diamBL.append(getDiam(starGBL))
            diamG.append(getDiam(Gprime))
            
        elif stat == 'avgpath':
            avgPath.append(getAvgPath(starG))
            avgPathBL.append(getAvgPath(starGBL))
            avgPathG.append(getAvgPath(Gprime))
            
        elif stat == 'medpath':
            medianPath.append(getMedPath(starG))
            medianPathBL.append(getMedPath(starGBL))
            medianPathG.append(getMedPath(Gprime))
         
        elif stat == 'degRank':
            sd = sortByDegree(starG)
            sdBL = sortByDegree(starGBL)
            sdG = sortByDegree(Gprime)
            #print starG.number_of_nodes(), starGBL.number_of_nodes(), Gprime.number_of_nodes()
            #print len(sd), len(sdBL), len(sdG)
            rankDegree.append(scipy.stats.kendalltau(sd, sdG)[0])
            rankDegreeBL.append(scipy.stats.kendalltau(sdBL, sdG)[0])
            
        elif stat == 'prRank':
            sd = sortByPR(starG)
            sdBL = sortByPR(starGBL)                
            sdG = sortByPR(Gprime)
            rankPR.append(scipy.stats.kendalltau(sd, sdG)[0])
            rankPRBL.append(scipy.stats.kendalltau(sdBL, sdG)[0])
            
        elif stat == 'badprRank':
            try:
                sd = sortByPR(starG)                                   
                sdG = sortByPR(Gprime)
                rankPR.append(scipy.stats.kendalltau(sd, sdG)[0])
            except:
                rankPR.append(-2)
                
            try:
                sdBL = sortByPR(starGBL)                                   
                sdG = sortByPR(Gprime)
                rankPRBL.append(scipy.stats.kendalltau(sdBL, sdG)[0])
            except:
                rankPRBL.append(-2)
            
        elif stat == 'clCoef':
            clCoef.append(nx.average_clustering(starG))
            clCoefBL.append(nx.average_clustering(starGBL))
            clCoefG.append(nx.average_clustering(Gprime))
            
        elif stat == 'connComp':
            connC.append(connComp(starG))
            connCBL.append(connComp(starGBL))
            connCG.append(connComp(Gprime))
            
        elif stat == 'MMMpath':
            t1, t2, t3 = getMeanMedMaxAvgPath(starG)
            meanAvgPath.append(t1)
            medAvgPath.append(t2) 
            maxAvgPath.append(t3)
            t1, t2, t3 = getMeanMedMaxAvgPath(starGBL)
            meanAvgPathBL.append(t1)
            medAvgPathBL.append(t2)
            maxAvgPathBL.append(t3)
            t1, t2, t3 = getMeanMedMaxAvgPath(Gprime)
            meanAvgPathG.append(t1)
            medAvgPathG.append(t2)
            maxAvgPathG.append(t3)

        elif stat == 'mapDiam':
            t1, t2 = getMappedMaxPath(starG, Gprime) 
            #print t1,t2
            diamReal.append(t1)
            diamMapped.append(t2)
            t1, t2 = getMappedMaxPath(starGBL, Gprime)
            #print t1,t2
            diamRealBL.append(t1)
            diamMappedBL.append(t2)
        
        elif stat == 'acc':
            t1,t2,t3 = accuracy(Gprime, starG)
            prec.append(t1)
            recall.append(t2)
            Fm.append(t3)
            t1,t2,t3 = accuracy(Gprime, starGBL)
            precBL.append(t1)
            recallBL.append(t2)
            FmBL.append(t3)
            
            
        #print rankPR, rankPRBL
        
        GStatsK.append(k*GNumNodes/100.0)
    OUT[d] = {'E': numEdges,'D': diam, 'Apath': avgPath, 'Mpath': medianPath, 'degRank': rankDegree, 'prRank': rankPR, 'badprRank': rankPR,'clCoef': clCoef}
    OUTBL[d] = {'E': numEdgesBL,'D': diamBL, 'Apath': avgPathBL,'Mpath': medianPathBL, 'degRank': rankDegreeBL, 'prRank': rankPRBL, 'badprRank': rankPRBL,'clCoef': clCoefBL}
    GStats[d] = {'k': GStatsK, 'E': numEdgesG, 'D': diamG, 'Apath': avgPathG, 'Mpath': medianPathG, 'clCoef': clCoefG, 'avgAlpha': alphaAvg}
    OUT[d]['CC'], OUTBL[d]['CC'], GStats[d]['CC'] = connC, connCBL, connCG
    OUT[d]['meanAvgPath'], OUTBL[d]['meanAvgPath'], GStats[d]['meanAvgPath'] = meanAvgPath,meanAvgPathBL,meanAvgPathG
    OUT[d]['medAvgPath'], OUTBL[d]['medAvgPath'], GStats[d]['medAvgPath'] = medAvgPath, medAvgPathBL, medAvgPathG
    OUT[d]['maxAvgPath'], OUTBL[d]['maxAvgPath'], GStats[d]['maxAvgPath'] = maxAvgPath, maxAvgPathBL, maxAvgPathG
    OUT[d]['prec'], OUTBL[d]['prec'] = prec, precBL
    OUT[d]['recall'], OUTBL[d]['recall'] = recall, recallBL
    OUT[d]['Fm'], OUTBL[d]['Fm'] = Fm, FmBL
    
    OUT[d]['diamReal'], OUTBL[d]['diamReal'] = diamReal, diamMapped
    OUT[d]['diamMapped'], OUTBL[d]['diamMapped'] = diamRealBL, diamMappedBL
    #print connC, connCBL, connCG
    dumpname = mode + '_' +stat + '.pkl'
    pickle.dump((OUT, OUTBL, GStats), open(dumpname, 'wb'))
    for d in data:
        plt.plot(K, GStats[d]['avgAlpha'], linewidth=2.0)
    plt.legend(data, loc = 4, )
    plt.xlabel('K (%)',fontsize=27)
    plt.ylabel('density of the ego-networks',fontsize=27)
    plt.show()

    # for d in data:
        # plt.plot(GStats[d]['k'], OUT[d]['D'])  
        # plt.plot(GStats[d]['k'], OUTBL[d]['D'])
    # plt.show()
        

