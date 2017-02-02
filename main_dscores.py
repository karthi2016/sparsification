import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import copy
import itertools
import time
import pickle
import os.path
import pqueue
import random
import sys

   
def pickleLoad(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
        
def check(sets, items, out, starOrder):
    coverage = {}
    for t, i in sets.iteritems():
        for item, coveredItems in out.iteritems():
            star = set(coveredItems)
            star.add(item)
            if set(i) <= star and item in set(i):
                coverage[t] = starOrder[item]
    return coverage

def plotting(setsG, starG, initSets, starOrder, name):
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
        
    nx.draw(starG, pos, width='2')
    nx.draw_networkx(starG, pos, nodelist=starOrder.keys(), node_color='g',labels = starOrder, width='1',with_labels=True, font_color='w', font_size = 15, node_size=500)
    plt.savefig(name + '.pdf')
    plt.show()

def readInput(file):
    sets = {}
    id = 0
    with open(file, "r") as f:
        for line in f:
            sets[id] = sets.get(id,[])+ (str.split(line.strip(),';'))
            id += 1
    return sets
    
def readGIT(file):
    sets = {}
    with open(file, "r") as f:
        for line in f:
            line = str.split(line.strip(),';')
            id = line[3]
            dev = line[1]
            t = sets.get(id, set())
            t.add(dev)
            sets[id] = t
    return sets
    
def readGITtoy(file):
    devs = ['oremj', 'jasonthomas', 'willkg', 'onhung', 'glogiotatidis', 'ngokevin', 'lonnen', 'brutal-chaos', 'murrown', 'jlongster', 'johngian', 'edunham', 'ppapadeas', 'cvan', 'craigcook', 'Osmose', 'sgarrity', 'lauraxt', 'Sancus', 'rhelmer', 'selenamarie', 'akatsoulas', 'MostAwesomeDude', 'spasovski', 'washort', 'ednapiranha', 'Pike', 'potch', 'peterbe', 'uberj', 'jpetto', 'bhearsum', 'wraithan', 'pmclanahan', 'fwenzel', 'andymckay', 'rlr', 'AdrianGaudebert', 'ossreleasefeed', 'mythmon', 'robhudson', 'clouserw', 'mattbasta', 'comzeradd', 'ebal', 'kumar303', 'rehandalal']
    sets = {}
    with open(file, "r") as f:
        for line in f:
            line = str.split(line.strip(),';')
            id = line[3]
            dev = line[1]
            if dev in devs:
                t = sets.get(id, set())
                t.add(dev)
                #print t 
                sets[id] = t
                #print sets[id]
    return sets
    
def readGITbigtoy(file):
    devs = ['yrashk', 'megalithic', 'christianh92', 'monfresh', 'PaulKinlan', 'fnagel', 'Wilto', 'raggi', 'millermedeiros', '21x9', 'dhilipsiva', 'rajatchopra', 'fabiokung', 'brittballard', 'jctod', 'weilu', 'Raynes', 'ADmad', 'deld', 'imathis', 'svenfuchs', 'mlebarron', 'jcaudle', 'hubot', 'rstrong', 'harlow', 'flightblog', 'tmcw', 'kshimabukuro', 'mirar', 'joneslee85', 'alrra', 'peol', 'ajashton', 'neerajdotname', 'knewter', 'ajones446', 'muffe', 'leehuffman', 'dmethvin', 'Joe23', 'dwax', 'markstory', 'gphat', 'dpetrov', 'miyagawa', 'rwldrn', 'jyurek', 'Phatestroke', 'secondsun', 'wagenet', 'jnjosh', 'vojtajina', 'adamlogic', 'vertigoclinic', 'Machiavelli86', 'johnbender', 'ryanmaxwell', 'neilhunt1', 'mrkn', 'kswedberg', 'nurse', 'bostonaholic', 'dfoulger', 'aureliosr', 'jsmecham', 'pwightman', 'drogus', 'richo', 'yogthos', 'davatron5000', 'arg0s', 'carlosantoniodasilva', 'mike-burns', 'clintjhill', 'hjdivad', 'agrieve', 'grayghostvisuals', 'incanus', 'SlexAxton', 'kevinrenskers', 'pkozlowski-opensource', 'skronhar', 'einars', 'amalloy', 'agassiyzh', 'kthakore', 'rkh', 'rafl', 'skrasser', 'msv', 'jrgifford', 'jferris', 'eastridge', 'JangoSteve', 'mikedeboer', 'lorenzo', 'ceeram', 'asynchronism', 'bluebrindle', 'tjvantoll', 'slemke', 'arunagw', 'SandeepDhull1990', 'ninjudd', 'msenges', 'jzaefferer', 'amatsuda', 'NoiseEee', 'duggieawesome', 'yakko', 'mmccroskey', 'alesj', 'mtibben', 'derrickko', 'mostafaeweda', 'winston', 'darhodesjr', 'Simm033', 'tomlane', 'renansaddam', 'SebastienThiebaud', 'dvizzacc', 'sarahlensing', 'DaveStein', 'BenWur', 'jacobjennings', 'sillylogger', 'sebastienblanc', 'zporter', 'jaydee3', 'nschum', 'Kalle05', 'levenste', 'bradlygreen', 'nbeloglazov', 'Sweepy86', 'mwichary', 'Xavan', 'tdawson', 'allending', 'mhevery', 'LBRapid', 'RedWolves', 'johndavid400', 'ajoslin', 'alloy', 'jsgarvin', 'cc-chris-cc', 'PragTob', 'wibblymat', 'royfochtman', 'necolas', 'radar', 'gabrielrinaldi', 'anselmbradford', 'Rlind', 'greendog', 'mbostock', 'schneems', 'qrush', 'lulinqing', 'lexrus', 'arielkennan', 'bcardarella', 'Smithes', 'steveyken', 'criscristina', 'technomancy', 'jimweirich', 'danmcclain', 'agcolom', 'janpadrta', 'rwz', 'georgebrock', 'thradec', 'pczajor', 'franckverrot', 'DanielHeath', 'Dignifiedquire', 'ruafozy', 'angelolloqui', 'blackgold9', 'nacin', 'davidfestal', 'elektrojunge', 'FroMage', 'jfirebaugh', 'mdz', 'jmertic', 'gonecoding', 'jawa', 'alexandernst', 'KuraFire', 'pixeltrix', 'petejkim', 'LarryEitel', 'gseguin', 'samvermette', 'fusion94', 'dnalot', 'thomasfedb', 'abhgupta', 'Krinkle', 'percysnoodle', 'stohlern', 'thesteve0', 'gf3', 'jahnkej', 'uGoMobi', 'scottgonzalez', 'sperris', 'mjankowski', 'nimbupani', 'andreaskri', 'oalders', 'trptcolin', 'kborchers', 'ProLoser', 'jbirdjavi', 'l4u', 'sire-sam', 'kenhys', 'abhayashenoy', 'louh', 'vladikoff', 'rmillr', 'dthompson', 'ilikepi', 'Pascal2142', 'explicitcall', 'bbenezech', 'ihatanaka', 'ZJDennis', '0xced', 'joshuaclayton', 'romaonthego', 'drbrain', 'ranguard', 'qmx', 'kevinjalbert', 'pashields', 'gylaz', 'neilmiddleton', 'gilltots', 'lassebm', 'xsawyerx', 'czgarrett', 'vijaydev', 'tagawa', 'gabebw', 'daveray', 'zhigang1992', 'btford', 'monken', 'josevalim', 'kou', 'satazor', 'katielewis', 'haines', 'matejonnet', 'marcelombc', 'Kjuly', 'BanzaiMan', 'mjsmard', 'larsacus', 'jasperblues', 'schof', 'nahi', 'ashbb', 'dcutting', 'abstractj', 'adarshpandit', 'chochos', 'petebacondarwin', 'lucasmedeirosleite', 'ahmet', 'pengwynn', 'headius', 'ebryn', 'frodsan', 'ikasiuk', 'graetzer', 'thepumpkin1979', 'jouve', 'muqusun', 'rohit', 'awagener', 'mxie', 'lslang', 'tenebrousedge', 'mmocny', 'rxaviers', 'Cocoanetics', 'parkr', 'aharren', 'fxn', 'sleeper', 'nuarhu', 'thisandagain', 'mikesherov', 'norman', 'tristen', 'ehmorris', 'ajuckel', 'davorb', 'jdutil', 'judearasu', 'marzapower', 'kaishin', 'garthex', 'GeekOnCoffee', 'lnader', 'atomkirk', 'ahhrrr', 'MaxNe', 'doy', 'spagalloco', 'spenrose', 'yaakaito', 'youknowone', 'MattRogish', 'paulirish', 'derSdot', 'danheberden', 'jksk', 'murdockn', 'ypitomets', 'joshk', 'nicholassm', 'alekseyn', 'lucascaton', 'dpree', 'AliSoftware', 'rcackerman', 'smarterclayton', 'rrausch', 'Aetherpoint', 'petersendidit', 'chrisjeane', 'irrationalfab', 'herhausa', 'gtraxx', 'lauritzthamsen', 'jdiago', 'AD7six', 'roidrage', 'maxamillion', 'wiistriker', 'waltz', 'jwilling', 'riceguitar', 'gfontenot', 'gabrielschulhof', 'andyhull', 'dewind', 'lippytak', 'djoaquin', 'rksm', 'gorsuch', 'zenspider', 'crx', 'xuanxu', 'rclosner', 'lucasmazza', 'jasonmay', 'davetoxa', 'lancepantz', 'addyosmani', 'drublic', 'jaubourg', 'timonv', 'jerolimov', 'MSch', 'spastorino', 'erikj', 'FremyCompany', 'kevinchampion', 'djcp', 'icco', 'MarcCfA', 'laurameixell', 'yas375', 'lepouchard', 'adamyonk', 'danmcp', 'AlexDenisov', 'jasondavies', 'athomschke', 'JoelQ', 'niketdesai', 'richaagarwal', 'genehack', 'prakashk', 'conradz', 'mRs-', 'benjaminoakes', 'sergi', 'levjj', 'kselden', 'adamgamble', 'koos', 'dbussink', 'wycats', 'dqminh', 'allspiritseve', 'spara', 'MattStopa', 'soberstadt', 'yoshihara', 'kommen', 'alexbaldwin', 'brianlittmann', 'mangini', 'parndt', 'piroor', 'orta', 'mattt', 'ajpiano', 'githubtrainer', 'justinxreese', 'kares', 'lukemelia', 'alanjosephwilliams', 'jaredmoody', 'tony', 'drapergeek', 'preinheimer', 'shukydvir', 'arctouch-kevinlim', 'Ovanderb', 'daijiro', 'tombentley', 'bricooke', 'rpwll', 'loicfrering', 'Mik-die', 'imrehg', 'guilleiguaran', 'twinge', 'KL-7', 'lholmquist', 'gsnowman', 'basti1253', 'TommyCreenan', 'madrobby', 'dogmatic69', 'aoridate', 'cypher', 'gingerir', 'lennartcl', 'rdavies', 'okkez', 'brixen', 'donnahawkins', 'mattdbridges', 'uwabami', 'krainboltgreene', 'samanpwbb', 'marvindpunongbayan', 'marcandre', 'jessieay', 'mklabs', 'hyunsook', 'drockwell', 'shama', 'Brodingo', 'mojombo', 'Steveboo', 'hone', 'cnoss', 'javruben', 'pat', 'dhh', 'sliu-gemvara', 'steveklabnik', 'unity', 'Candid', 'zzak', 'bcaccinolo', 'thagomizer', 'tombenner', 'IgorMinar', 'gjtorikian', 'SoniaMane', 'stefanpenner', 'dereuromark', 'dmathieu', 'marknutter', 'springmeyer', 'openshift-bot', 'egrim', 'kpdecker', 'barism', 'rspier', 'aporat', 'blakewatters', 'brynbellomy', 'vipulnsward', 'alexefish', 'ndbroadbent', 'theocalmes', 'smyrgl', 'edgurgel', 'mhoran', 'lest', 'insanehunter', 'janjongboom', 'technoweenie', 'zakkain', 'rexeisen', 'cries', 'sleepYdrone', 'xy00z', 'mbarr-snap', 'simi', 'stevan', 'enebo', 'mpapis', 'jstory-gemvara', 'ileitch', 'eric-horacek', 'Reprazent', 'mneorr', 'nbleisch', 'rchavik', 'watsonian', 'loraenia', 'sikachu', 'joeyw', 'evanphx', 'plapier', 'alexgibson', 'sbellity', 'nfarina', 'jcoleman', 'tenderlove', 'arunthampi', 'arschmitz', 'enriquez', 'matzew', 'jnewland', 'ldiqual', 'MugunthKumar', 'aliHafizji', 'acacheung', 'Jippi', 'ofiesh', 'stmontgomery', 'sindresorhus', 'tomdale', 'rdworth', 'gugmaster', 'garann', 'indirect', 'SheilaLDugan', 'joshkurz', 'jeremy', 'kennethjohnbalgos', 'AshFurrow', 'mfine', 'malandrina', 'tchak', 'jeffarena', 'jcollas', 'ondrae', 'yliu80016', 'ansis', 'keiththomps', 'gavinking', 'jonathantneal', 'BDQ', 'jameswatts', 'calmez', 'hyPiRion', 'bkardell', 'xeqi', 'cowboy', 'marcoscaceres', 'githubstudent', 'cmar', 'Cr4ckX', 'mbinna', 'nightwing', 'shugo', 'marcooliveira', 'timmywil', 'tigrang', 'michaelklishin', 'croaky', 'awwaiid', 'cvasilak', 'yhahn', 'pwinkler', 'jaustinhughey', 'rafaelfranca', 'gibson042', 'azcostanzo', 'nlogax', 'sstephenson', 'nashby', 'lox', 'mohangk', 'laptobbe', 'goArchy', 'kraman', 'cssboy', 'ycombinator', 'stoffeastrom', 'quintesse', 'jrubyci', 'magadred', 'rwstauner', 'dr-nafanya', 'pravisankar', 'bsideup', 'mislav', 'azimmer4', 'toddparker', 'tapi', 'jlembeck', 'ugisozols', 'goshakkk', 'Gigfel', 'artemp', 'uvtc', 'siuying', 'mmunhall', 'mmickle', 'danbev', 'ndang', 'jonleighton', 'weddingcakes', 'ekdevdes', 'aFarkas', 'clayallsopp', 'danavery', 'wesbos', 'smartinez87', 'sferik', 'tbaba', 'filipediasferreira', 'mwoghiren', 'jsamuelson024', 'jhudson8', 'acerbetti', 'kreeger', 'nhr', 'joliss', 'danparsons', 'evan', 'daguar', 'maroessler', 'vtourraine', 'xcambar', 'sangregoriopaolo', 'ocean90', 'Usernamenigel', 'gnarf37', 'pjaspers', 'mlester', 'jsteiner', 'rclements', 'juth', 'tkellen', 'henrikhodne', 'westonplatter', 'Keithbsmiley', 'anoldguy', 'timraymond']
    sets = {}
    with open(file, "r") as f:
        for line in f:
            line = str.split(line.strip(),';')
            id = line[3]
            dev = line[1]
            if dev in devs:
                t = sets.get(id, set())
                t.add(dev)
                sets[id] = t
    return sets
  
    
def getItemIndex(sets):
    items = {}
    for team, i in sets.iteritems():
        for j in i:
            items[j] = items.get(j, []) + [team]        
    return items

def removeFound(sets, items, coveredItems, coveredSets):
    for i in coveredSets:
        del sets[i]
    
    items = getItemIndex(sets)
    return sets, items
    
def getGraph(out):
    G = nx.Graph()
    for k,v in out.iteritems():
        for j in v:
            G.add_edge(k,j)
            
    G = nx.Graph(G)
    G.remove_edges_from(G.selfloop_edges())
    return G
    
def getDataGraph(sets):
    G = nx.Graph()
    for k,v in sets.iteritems():
        for i in v:
            for j in v:
                G.add_edge(i,j)
    plt.figure()
    nx.draw(G) 
    plt.savefig('input.pdf')
    G = nx.Graph(G)
    G.remove_edges_from(G.selfloop_edges())

    return G
    

def removeStandalones(sets):
    sets = {k: v for k,v in sets.iteritems() if len(v) > 1}
    return sets

def removeSameTeams(sets):
    out = {}
    for k1,v1 in sets.iteritems():
        if v1 not in out.values():
            out[k1] = v1
    return out
    
    
def getBestHyperedge(sets, items):
    bestSet, bestTeams, bestScore, bestItem = set(), set(), -1, []  
    for item, teams in items.iteritems():
        coveredItems, coveredSets = {}, []
        for s in teams:
            for i in sets[s]:
                coveredItems[i] = coveredItems.get(i,-1) + 1.0                
            coveredSets.append(s)
        score = sum(coveredItems.values()) - max(coveredItems.values())

        if score > bestScore:
            bestSet, bestTeams, bestScore = set(coveredItems), coveredSets, score
            bestItem = item
    return bestSet, bestTeams, bestScore, bestItem    
    
def getScores(sets, items):
    itemsScore = {}
    for item, teams in items.iteritems():
        coveredItems, coveredSets = {}, []
        for s in teams:
            for i in sets[s]:
                coveredItems[i] = coveredItems.get(i,-1) + 1.0         
            coveredSets.append(s)
        score = sum(coveredItems.values()) - max(coveredItems.values())
        itemsScore[item] = -score
    return itemsScore
    
def greedy(sets, items):
    coveredItems, coveredSets = set(), set()
    out = {}
    starOrder = {}
    itemsScore = getScores(sets, items)
    
    orderByScores = pqueue.priority_dict(itemsScore)
    c = 0
    while orderByScores:
        item, score = orderByScores.pop_smallest()
        if len(coveredSets) == len(sets):
            break
        starOrder[item] = c
        c += 1
        out[item] = set()       
        
        for team in set(items[item]) - coveredSets:
            out[item] |= set(sets[team])
        coveredSets |= set(items[item])
        
        for item in out[item] - set(out.keys()):            
            localCoveredItems = {}  
            for t in set(items[item]) - coveredSets:
                for i in sets[t]:
                    localCoveredItems[i] = localCoveredItems.get(i,-1) + 1.0
            
            if localCoveredItems:
                score = -(sum(localCoveredItems.values()) - max(localCoveredItems.values()))
            else:
                score = 1            
            orderByScores[item] = score
    return out, starOrder

def greedyBL(sets, items):
    coveredItems, coveredSets = set(), set()
    out = {}
    starOrder = {}
    
    itemsScore = {}
    for item, teams in items.iteritems():
        score = len(teams)
        itemsScore[item] = -score
        
    orderByScores = pqueue.priority_dict(itemsScore)
    print 'q is done'
    c = 0
    while orderByScores:
        item, score = orderByScores.pop_smallest()
        if len(coveredSets) == len(sets):
            break
        starOrder[item] = c
        c += 1
        out[item] = set()       
        
        for team in set(items[item]) - coveredSets:
            out[item] |= set(sets[team])
        coveredSets |= set(items[item])
        
        for item in out[item] - set(out.keys()):
            t = set(items[item]) - coveredSets
            if t:
                score = len(t)
            else:
                score = -1
            orderByScores[item] = -score        
    return out, starOrder

def trivial(sets):    
    out = {}
    starOrder = {}
    c = 0
    for id, t in sets.iteritems():
        team = list(t)
        center = random.choice(team)
        out[center] = out.get(center,[]) + team
        starOrder[center] = c
        c += 1
    return out, starOrder
 
if __name__ == '__main__':

    #dataset = sys.argv[1]
    dataset = 'lastFM_tags'
                     

    G, sets = pickle.load(open(os.path.join('UnderlyingNetwork', dataset +'.pkl'),'rb'))
    
    tic = time.time()
    items = getItemIndex(sets)    
    preproc = time.time() - tic
    
    tic = time.time()
    outGreedy, starOrderGreedy = greedy(sets, items)
    timeGr = time.time() - tic
    
    tic = time.time()
    outTrivial, starOrderTrivial = trivial(sets)
    timeTriv = time.time() - tic
    
    starGGreedy = getGraph(outGreedy)
    starGTriv = getGraph(outTrivial)
    
    print 'our method:'
    print 'fraction of edges:', 1.0*starGGreedy.number_of_edges()/G.number_of_edges()
    
    print
    print 'baseline method:'    
    print 'fraction of edges:', 1.0*starGTriv.number_of_edges()/G.number_of_edges()
   
