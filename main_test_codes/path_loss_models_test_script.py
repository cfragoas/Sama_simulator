import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Cursor
from collections import Counter
#import tensorflow as tf

def calculate_los_prob_uma(d2d, hut, multipath=False):
    """
    Calcula a probabilidade de um percurso ser Line of Sight (LOS) ou No Line of Sight (NLOS). Baseado na TR 338.901 v 17.1.0.\n
    d2d - Distância no eixo horizontal entre a BS e a UE (m) / no cenário outdoor-outdoor apenas!\n
    hut - Altura do UE (m)
    """
    #for i,d in enumerate(d2d):
    if hut > 23:
        raise Exception("Altura do UE deve ser menor que 23 m")
    if d2d <= 18:
        problos = 1
    if d2d > 18:
        if hut <= 13:
            c = 0
        else:
            c = np.power((hut - 13) / 10, 1.5)
        problos = ((18 / d2d) + np.exp(-d2d / 63) * (1 - 18 / d2d)) * (
                    1 + c * 5 / 4 * np.power(d2d / 100, 3) * np.exp(-d2d / 150))

    return problos

#Create the UMA 3GPP Path Loss based on TR 38.901
def generate_uma_path_loss(d2d, d3d, hut, hbs, fc, multipath=False):
    """
    Calcula um percurso, considerado ele ser Line of Sight (LOS) ou No Line of Sight (NLOS). Baseado na TR 38.901 v 17.1.0.\n
    fc - frequência em GHz \n
    hut -  Altura da UE (m) \n
    hbs - Altura da BS (m) \n
    d2d e d3d são as distâncias em (m)!
    """
    c = 3*10**8 # Velocidade da luz (m/s)
    dbp = 4*(hbs-1)*(hut-1)*fc*10**9/c
    Ploss = np.empty_like(d2d) # Cria array de vazio para preencher com os PLOS

    with np.nditer([d2d, d3d, Ploss], op_flags=[['readonly'], ['readonly'], ['writeonly']]) as iter:
        for a,b,pl in iter:
           # if a > 5000 or a < 10:
           #     raise Exception("Distância entre BS e UE deve estar entre 10 m e 5 km")

            problos = calculate_los_prob_uma(a,hut,multipath)

            prop = random.choices(["LOS","NLOS"],weights = [problos,1-problos]) #Simula se a propagação será LOS ou NLOS
            #prop = "LOS"
            #prop = "NLOS"

            #Calcula o PL1 e o PL2 (usado tanto em casos de LOS como NLOS
            if a < dbp:
                PLOS = 28+22*np.log10(b)+20*np.log10(fc)+shadow_fading(0,4) #PL1
            else:
                PLOS = 28+40*np.log10(b)+20*np.log10(fc) \
                       -9*np.log10(np.power(dbp,2)+np.power(hbs-hut,2))+shadow_fading(0,4) #PL2
            if prop[0] == "LOS":
                Pathloss = PLOS
            else:
                PNLOS = 13.54+39.08*np.log10(b)+20*np.log10(fc)-0.6*(hut-1.5)+shadow_fading(0,6)
                #PNLOS = 32.4 + 20*np.log10(fc)+30*np.log10(b)
                Pathloss = np.max((PLOS,PNLOS))
            pl[...] = Pathloss
    return Ploss

def calculate_los_prob_win2(d2d):
    """
    Calcula a probabilidade de um percurso ser Line of Sight (LOS) ou No Line of Sight (NLOS). Baseado no projeto winner 2.\n
    d2d - Distância no eixo horizontal entre a BS e a UE (m) / no cenário outdoor-outdoor apenas!\n
    """
    problos = np.minimum(18/d2d,1)*(1-np.exp(-d2d/63))+np.exp(-d2d/63)

    return problos

#Create the WIN2  Path Loss based on WINNER II Projeect
def generate_win2_path_loss(d2d, d3d, hut, hbs, fc, multipath=False):
    """
    Calcula um percurso, considerado ele ser Line of Sight (LOS) ou No Line of Sight (NLOS). Baseado na TR 38.901 v 17.1.0.\n
    fc - frequência em GHz \n
    hut -  Altura da UE (m) \n
    hbs - Altura da BS (m) \n
    d2d e d3d são as distâncias em (m)!
    """
    c = 3*10**8 # Velocidade da luz (m/s)
    dbp = 4*(hbs-1)*(hut-1)*fc*10**9/c
    Ploss = np.empty_like(d2d) # Cria array de vazio para preencher com os PLOS
   
    with np.nditer([d2d, d3d, Ploss], op_flags=[['readonly'], ['readonly'], ['writeonly']]) as iter:
        for a,b,pl in iter:
           # if a > 5000 or a < 10:
           #     raise Exception("Distância entre BS e UE deve estar entre 10 m e 5 km")

            problos = calculate_los_prob_win2(a)

            prop = random.choices(["LOS","NLOS"],weights = [problos,1-problos]) #Simula se a propagação será LOS ou NLOS
            #prop = "LOS"
            #prop = "NLOS"

            #Calcula o PL1 e o PL2 
            if a < dbp:
                PLOS = 39+26*np.log10(b)+20*np.log10(fc/5.0)+shadow_fading(0,4) #PL1
            else:
                PLOS = 13.47+40*np.log10(b)+6*np.log10(fc/5.0) \
                       -14.0*np.log10(hbs-1)- 14.0*np.log10(hut-1)+shadow_fading(0,6) #PL2
            if prop[0] == "LOS":
                Pathloss = PLOS
            else:
                PNLOS = (44.9-6.55*np.log10(hbs))*np.log10(b)+31.46 \
                        + 5.83*np.log10(hbs)+23*np.log10(fc/5.0)+shadow_fading(0,8)
                #PNLOS = 32.4 + 20*np.log10(fc)+30*np.log10(b)
                Pathloss = PNLOS
            pl[...] = Pathloss   
    return Ploss

    
def generate_distance_map(eucli_dist_map, cell_size, htx, hrx, plot=False):
    # returns a centroids x lines x columns matrix with per pixel distance information from a centroid with htx height -
    # - to a coordinate with hrx height
    ds = []
    for eucli_dist_map in eucli_dist_map:
    # converting the euclidean distance to actual distance between Tx and Rx
        k =np.sqrt(np.power(eucli_dist_map* cell_size,2) + np.power(htx - hrx,2)) 
        ds.append(k)
    return np.where(ds == 0, 1, ds)



def fs_path_loss(d, f, var=6):  # simple free space path loss function with lognormal component 
    if var:
        log_n = np.random.lognormal(mean=0, sigma=np.sqrt(var), size=d.shape)  # lognormal variance from the mediam path loss
    else:
        log_n = 0

    pl = 40 * np.log10(d) + 20 * np.log10(f) + 92.45 #+ log_n  # f in GHz and d in km
    return pl

# Sombreamento:
def shadow_fading(m,std):
    """
    Função para calcular o efeito de perda por sombreamento em, baseada numa distribuição lognormal.Sendo:
    m: Média
    std: Desvio padrão
    """
    sf = np.random.lognormal(mean=m, sigma=std)
    return sf

# Funções para efeito doppler:
def get_velocity(scenario = None):
#Gera uma velocidade aleatória baseada nos cenários de cada modelo de perda de propagação:
    if "3GPP UMA": #3GPP UMA 
        vel = random.uniform(0,3)
    elif "WINNER2 C2": #Cenário C2 WINNER: Urban Macrocell
        vel = random.uniform(0,120)
    elif "WINNER2 D2": #Cenário D2 WINNER: Moving Networks
        vel = random.uniform(0,350)
    else:
        print("Cenario nao definido.Desconsiderando o efeito de deslocamento doppler")
        vel = 0
    return vel
def doppler_shift(v,f):
    """
    Calcula o desvio doppler de frequência. Considera-se que a chance de ocorrer um desvio para cima é a mesma que para baixo.\n
    f: frequência em Mhz\n
    c: velocidade em km/h 
    """
    c = 3*10**8
    fd = random.choices([f*(1-v/(c*3.6)),f*(1+v/(c*3.6))],weights = [0.5,0.5]) #Velocidade convertida para m/s
    return fd[0]

#Funções para penetração O2I

def indoor_loss(los_model = None):
    """Gera uma penetração O2I baseado no item 7.4.3 da TR 38.901. V.17.1.0"""
    d2in = random.uniform(0,25) #Distância indoor. Deve ser entre 0-25 m
    hl = #Cenário de alta perda de penetração
    ll = #Cenário de baixa perda de penetração
    d2in = random.uniform(0,25)
    if None:

    return 0 



# Código teste abaixo:


htx = 25
hrx = 1.5
csize = 1
fc = 3.6

em = np.random.uniform(10,1000,(1,1000))


dm = generate_distance_map(em,csize,htx,hrx,False)


uma = generate_uma_path_loss(em,dm,hrx,htx,fc,multipath = False)

fs = fs_path_loss(dm/100,fc)

win2 = generate_win2_path_loss(em,dm,hrx,htx,fc,multipath = False)

print(f"Resultados UMA:\n min: {np.min(uma)}\n max: {np.max(uma)}\n mean: {np.mean(uma)}\n std: {np.std(uma)}")

print(f"Resultados FS:\n min: {np.min(fs)}\n max: {np.max(fs)}\n mean: {np.mean(fs)}\n std: {np.std(fs)}")

print(f"Resultados WIN2:\n min: {np.min(win2)}\n max: {np.max(win2)}\n mean: {np.mean(win2)}\n std: {np.std(win2)}")

print(f"Breakpooint distance: { 4*(htx-1)*(hrx-1)*fc*10**9/(3*10**8)} m.")
print(f"fs at {dm[0,500]}m {fs[0,500]}")
print(f"uma at {dm[0,500]}m {uma[0,500]}")
print(f"win2 at {dm[0,500]}m {win2[0,500]}")

fig,ax = plt.subplots(figsize = (10,6))
ax.plot(np.sort(dm[0,:]),np.sort(uma[0,:]),'r',np.sort(dm[0,:]),np.sort(fs[0,:]),'b',np.sort(dm[0,:]),np.sort(win2[0,:]),'g')
plt.title("Path Loss for different approaches")
plt.xlabel("Distance (m)")
plt.ylabel("Path Loss (db)")
plt.legend(['UMA','FS','WIN2'])
plt.grid()
cursor = Cursor(ax,horizOn= True,vertOn=True)
plt.show()

dbp = 4*(htx-1)*(hrx-1)*fc*10**9/(3*10**8)

PL1 = 39+26*np.log10(dm)+20*np.log10(fc/5.0) #PL1 win2

PL2 = 13.47+40*np.log10(dm)+6*np.log10(fc/5.0)-14.0*np.log10(htx-1)- 14.0*np.log10(hrx-1) #PL2 win2

PL3 = 28+22*np.log10(dm)+20*np.log10(fc) #PL1 UMA

PL4 = 28+40*np.log10(dm)+20*np.log10(fc) -9*np.log10(np.power(dbp,2)+np.power(htx-hrx,2)) #PL2 UMA

#fig,ax = plt.subplots(figsize = (10,6))
#ax.plot(np.sort(dm[0,:]),np.sort(PL1[0,:]),'r',np.sort(dm[0,:]),np.sort(PL2[0,:]),'b',np.sort(dm[0,:]),np.sort(PL3[0,:]),'g',np.sort(dm[0,:]),np.sort(PL4[0,:]),'y')
#plt.title("WIN2 & UMA PL1 X PL2")
#plt.xlabel("Distance (m)")
#plt.ylabel("Path Loss (db)")
#plt.legend(['PL1 WIN2','PL2 WIN2','PL1 UMA','PL2 UMA'])
#plt.grid()
#cursor = Cursor(ax,horizOn= True,vertOn=True)
#plt.show()

# Teste :  Obtendo quantos percussos foram LOS e quantos foram NLOS

uma = []

win = []

i = em[0,:]

for x in i:
    u = calculate_los_prob_uma(x,hrx)
    w = calculate_los_prob_win2(x)
    uma.append(u)
    win.append(w)    

uma_type = []
win_type = []

for u,w in zip(uma,win):
    uma_type.append(random.choices(["LOS","NLOS"],weights = [u,1-u]))
    win_type.append(random.choices(["LOS","NLOS"],weights = [w,1-w]))


uma_los = [i for j,i in enumerate(uma_type) if i[0]=='LOS'] 
uma_nlos = [i for j,i in enumerate(uma_type) if i[0]=='NLOS']  

win_los = [i for j,i in enumerate(win_type) if i[0]=='LOS'] 
win_nlos = [i for j,i in enumerate(win_type) if i[0]=='NLOS']  



propag = ['UMA LOS', 'UMA NLOS', 'WIN2 LOS', 'WIN2 NLOS']
counts = [len(uma_los),len(uma_nlos),len(win_los), len(win_nlos)]
bar_labels = ['UMA LOS', 'UMA NLOS', 'WIN2 LOS', 'WIN2 NLOS']
bar_colors = ['r', 'b', 'g', 'k']

#fig, ax = plt.subplots()
#ax.bar(propag, counts, label=bar_labels,color = bar_colors)
#for index,value in enumerate(counts):
#    plt.text(x=index-0.1 , y =value+4 , s=f"{value}",color = 'black',fontweight='bold')
#ax.set_ylabel('Occurences')
#ax.set_title('Ocurrence of LOS and NLOS propagation for different models tested')
#ax.legend(title='Propagation')
#plt.show()


def doppler_shift(v,f):
    """
    Calcula o desvio doppler de frequência. Considera-se que a chance de ocorrer um desvio para cima é a mesma que para baixo.\n
    f: frequência em Mhz\n
    c: velocidade em km/h 
    """
    c = 3*10**8
    fd = random.choices([f*(1-v/(c*3.6)),f*(1+v/(c*3.6))],weights = [0.5,0.5]) #Velocidade convertida para m/s
    return fd[0]

#def get_ue_velocity(prop_model):