import math as m 
import numpy as np
from scipy.integrate import quad

###########UNITES###########
#pression : Pa
#volume : m^3
#debit : m^3/s
#longueur : m 
#temps : s
############################

###########VARIABLES########

#Résistance des tissus respiratoires (Pa.m^-3.s)
R=0.028*1e+06
#Résistance à l'écoulement de l'air dans les voies supérieures
R_ext = 1.2e7   #Pa.(m^3.s^-1)^-1.68
r = 1.68
#Choix de la pression motrice initiale
Pm=24000
#Choix de la constante de temps de relaxation des muscles expiratoires
tau=0.2
#Volume résiduel (volume d'air min dans les poumons)
RV=1.5e-3
#Capacité vitale (Volume d'air max dans les poumons)
CV=5.5e-3
# Diamètre de la trachée et dimension fractale
Dmax_0 = 2e-2
h_list = [1]+[2**(-1/3) for k in range(20)] +[0.7,0.75,0.8]+[0.85 for k in range(20)] #
h_th = 2**(-1/3)
# Compliance pulmonaire à P de rétraction élastique nulle:
C_0 = 5.7e-6 #m^3.Pa-1
# Volumes pulmonaires min et max, = à 106 et 20% de la CPT (Capacité Pulmonaire Totale  = CV+RV)
V_max = 7.5e-3 #m^3
V_min = 1.4e-3 #m^3

Compliance_param  = [ [0,0.882,0.0108e-2,1,10],
                    [1,0.882,0.0294e-2,1,10],
                    [2,0.686,0.050e-2,1,10],
                    [3,0.546,0.078e-2,1,10],
                    [4,0.428,0.098e-2,1,10],
                    [5,0.337,0.123e-2,1,10],
                    [6,0.265,0.139e-2,1,10],
                    [7,0.208,0.156e-2,1,10],
                    [8,0.164,0.171e-2,1,10],
                    [9,0.129,0.180e-2,1,10],
                    [10,0.102,0.190e-2,1,10],
                    [11,0.080,0.202e-2,1,9],
                    [12,0.063,0.214e-2,1,8],
                    [13,0.049,0.221e-2,1,8],
                    [14,0.039,0.228e-2,1,8],
                    [15,0.031,0.234e-2,1,7] ]

eta=1.8e-05          #Viscosité de l'air (Pa.s)
rho=1.14             #Masse volumique de l'air (kg/m^3)
P_atm = 0.0# 1013e2       #Atmospheric pressure
##################################################################################

import time  
class Timer(object):  
    def start(self):  
        if hasattr(self, 'interval'):  
            del self.interval  
        self.start_time = time.time()  
  
    def stop(self):  
        if hasattr(self, 'start_time'):  
            self.interval = time.time() - self.start_time  
            del self.start_time # Force timer reinit


def mat_links(n):
    '''A matrix that gives the ending node (lower gen) and starting node (higher gen) of a dichotomous and 
    symmetric tree of n generations.

    General form of the Matrix : (for any, non dichotomous or non symmetric system)
    Each line's index is the index of the link, and each line gives the indexes of the two related nodes
    '''

    M = [['atm',0],[0,1]]
    for i in range(2,2**n):
        if i%2 == 0 :
            M.append([i//2,i])
        else : 
            M.append([(i-1)//2,i])
    return M

def mat_nodes(n):
    '''A matrix that gives the links connected to a node in a dichotomous and symmetric tree for n gen

    General form of the Matrix : (for any non dich. or non sym. system)
    Each line's index is the index of the node. Gives the list of the index of the connected links. 
    '''

    M = [[0,1]]
    for i in range(1,2**(n-1)):
        M.append([i,2*i,2*i+1])
    return M

##################################################################################


def gen_count(mat_link, mat_node):
    '''builds a vector where the index is the same as the links and the value is the generation of the link
    i.e. : the i-th value is the generation of the link n°i
    '''
    N = len(mat_link)
    gen = [-1]+[0 for k in range(N-1)]
    for i in range(2,N): #the first two links are the one depicting the trachea and the first, only branch, we set them at gen = -1 and gen = 0
        j = i
        c = 0
        while j>1: #we stop if we reach the gen 0, i.e. the link n° 1
            a = mat_link[j][0] #a = index of node of lower gen than j
            j = min(mat_node[a]) #we refresh j by taking the index of the link that leads to j (of lower gen)
            c+=1 #we count how many gen we go up
            if c >= 40:
                break
        gen[i] = c

    return(gen)



def P_alv(t,VL_t,Phi):
    """Computes the initial pleural pression for a last-gen node (to determine Initial Conditions)
    """
    P_al = Pm*(1-np.exp(-t/tau))*((VL_t-RV)/CV)-(R*Phi)
    return P_al

def P_Pl(t,VL_t,Phi):
    '''Computes the pleural pressure, considered uniform for the lung system at time t:
    '''
    Pst = (V_max - V_min)*(1/C_0)*np.log((V_max-V_min)/(V_max-VL_t))
    return P_alv(t,VL_t,Phi) - Pst

def DMAX(g):
    h=1
    for i in range(g+1):
        h = h*h_list[i]
    return(Dmax_0*h)
def D(P,g):
    '''Compliance of a branch (link)
    returns the diameters D of the branch in function of the local pressure and generation
    '''
    h=1
    for i in range(g+1):
        h = h*h_list[i]
    Dmax = Dmax_0*h
    a_0,n_1,n_2 = Compliance_param[g][1],Compliance_param[g][3],Compliance_param[g][4]
    P_1 = (Compliance_param[g][1]*Compliance_param[g][3])/Compliance_param[g][2]
    P_2 = -((1-Compliance_param[g][1])*Compliance_param[g][4])/Compliance_param[g][2]

    if P<0:
        return Dmax*np.sqrt(a_0*((1-P/P_1)**-n_1))
    else : 
        return Dmax*np.sqrt(1-((1-a_0)*((1-P/P_2)**-n_2)))

def D4(P,g):
    return D(P,g)**4

def int_D4(Po,Pi,g):
    '''
    Calcule l'intégrale de Pi à Po de la fonction D(P)**4 pour la génération g
    '''
    #On charge les paramètres de compliance
    h=1
    for i in range(g+1):
        h = h*h_list[i]
    Dmax = Dmax_0*h
    a_0,n_1,n_2 = Compliance_param[g][1],Compliance_param[g][3],Compliance_param[g][4]
    P_1 = (Compliance_param[g][1]*Compliance_param[g][3])/Compliance_param[g][2]
    P_2 = -((1-Compliance_param[g][1])*Compliance_param[g][4])/Compliance_param[g][2]

    #On explicite l'intégrale sur les parties positives et négatives
    def neg(P):
        return (Dmax**4)*(a_0**2)*(P_1/(2*n_1-1))*((1-P/P_1)**(-2*n_1+1))

    def pos(P):
        return (Dmax**4)*( P - 2*(1-a_0)*P_2*(1/(n_2-1))*((1-P/P_2)**(-n_2+1)) + ((1-a_0)**2)*P_2*(1/(2*n_2-1))*((1-P/P_2)**(-2*n_2+1)) )

    if Po<0:
        if Pi<0:
            return neg(Po) - neg(Pi)
        if Pi>=0:
            return neg(Po) - neg(0) + pos(0) - pos(Pi)
    if Po>=0:
        if Pi>=0:
            return pos(Po) - pos(Pi)
        if Pi<0:
            return neg(0) - neg(Pi) + pos(Po) - pos(0)

def dDdP(P,g):
    '''Returns the derivative of D on P
    '''
    h=1
    for i in range(g+1):
        h = h*h_list[i]
    Dmax = Dmax_0*h
    a_0,n_1,n_2 = Compliance_param[g][1],Compliance_param[g][3],Compliance_param[g][4]
    P_1 = (Compliance_param[g][1]*Compliance_param[g][3])/Compliance_param[g][2]
    P_2 = -((1-Compliance_param[g][1])*Compliance_param[g][4])/Compliance_param[g][2]

    if P<0:
        return Dmax*np.sqrt(a_0)*(n_1/P_1)*((1-(P/P_1))**(-n_1-1))*(1/(2*np.sqrt((1-(P/P_1))**-n_1)))
    else : 
        return -Dmax*(1-a_0)*(n_2/P_2)*((1-(P/P_2))**(-n_2-1))*(1/(2*np.sqrt(1-(1-a_0)*((1-(P/P_2))**-n_2))))


def f(Po,Pi,g,Q,P_plr,eps=0.0035):
    """Function to solve = 0 for each node to find the local Pi, Po and Q of the link between
    """
    h=1
    for i in range(g+1):
        h = h*h_list[i]
    L = 3*Dmax_0*h
    Re = 4*rho*Q/(eta*np.pi*(D(Po-P_plr,g)+D(Pi-P_plr,g))/2)
    return np.array( int_D4(Po-P_plr,Pi-P_plr,g)-32*((rho*Q*(1/np.pi))**2)*np.log(D(Po-P_plr,g)/D(Pi-P_plr,g))+1*(128*eta*L*Q*(1/np.pi))*(1.5+eps*Re) )

def debug_f(Po,Pi,g,Q,P_plr,eps=0.0035):
    """Function to solve = 0 for each node to find the local Pi, Po and Q of the link between
    """
    h=1
    for i in range(g+1):
        h = h*h_list[i]
    L = 3*Dmax_0*h
    Re = 4*rho*Q/(eta*np.pi*(D(Po-P_plr,g)+D(Pi-P_plr,g))/2)
    a=int_D4(Po-P_plr,Pi-P_plr,g)
    b=-32*((rho*Q*(1/np.pi))**2)*np.log(D(Po-P_plr,g)/D(Pi-P_plr,g))
    c=(128*eta*L*Q*(1/np.pi))*(1.5+eps*Re) 
    print('int D4 :', a,'\n log funtion : ',b,'\n loss term: ',c)
    print('assert that residue is negligeable in linear case : ',(abs(a)-abs(c))/(abs(a)+abs(c)))
    return

def dfdQ(Po,Pi,g,Q,P_plr,eps=0.0035):
    '''returns df/dQ
    '''
    h=1
    for i in range(g+1):
        h = h*h_list[i]
    L = 3*Dmax_0*h
    Re = 4*Q*rho/(eta*np.pi*(D(Po-P_plr,g)+D(Pi-P_plr,g))/2)
    return np.array(  -64*Q*((rho*(1/np.pi))**2)*np.log(D(Po-P_plr,g)/D(Pi-P_plr,g))+1*(128*eta*L*(1/np.pi))*(1.5+2*eps*Re) )

def dfdPo(Po,Pi,g,Q,P_plr):
    '''returns df/dPo
    '''
    return np.array( D4(Po-P_plr,g)-32*((rho*Q*(1/np.pi))**2)*dDdP(Po-P_plr,g)/D(Po-P_plr,g) )

def dfdPi(Po,Pi,g,Q,P_plr):
    '''returns df/dPi
    '''
    return np.array( -D4(Pi-P_plr,g)+32*((rho*Q*(1/np.pi))**2)*dDdP(Pi-P_plr,g)/D(Pi-P_plr,g) )


def sat_fun(x,sat):
    x=np.array(x)
    if np.linalg.norm(x)!=0: 
        return (x/np.linalg.norm(x))*np.linalg.norm(sat)*np.log(1+ (np.linalg.norm(x)/np.linalg.norm(sat)))
    else : 
        return 0

def R_lin_eq(g_in,g_max=15):
    'Compute the equivalent resistance of gmax-g_in generations of linear'
    assert g_in <= g_max, 'GENERATION NB ERROR'
    R0 = (128*eta*3*Dmax_0) / (np.pi * Dmax_0**4) #Poiseuille hydraulic resistance
    return (g_max - g_in)*(2**(g_in))*R0


def refresh_system(mat_link,mat_node,P_ini,Q_ini,t=0,DeltaP=100000,stop_crit=1e-4,epsilon=[1e-2,1e-6],it_lim=1e3,V=CV,DeltaT=0.1,grad='naive'):  
    """Computes the state of the system for 1 temporal iteration 

    P_ini, Qini = set of value for P and Q 

    epsilon : the precision we want on F=0, first digit for the kirshoff equation and second for the flow one

    it_lim : max amount of iterations

    DeltaP : Linearises the D(P) :  if DeltaP if very high, the behaviour of D(P) is asymptotical and D(P) = Dmax, dDdP(P) = 0.
    If it is set to 0, we have full non linear behavior. Lim of divergence aroud P_Pl

    V = Initial lung volume at the beginning of the iteration

    Delta_t = duration of the iteration

    """

    print('---\nSTART ITERATION\n---')

    Phi = Q_ini[0]
    
    n = len(mat_link) #N° of nodes and intersections
    P = P_ini.copy() #will contain the pressures at time t
    Q = Q_ini.copy() #will contain the debit at time t
    gen = gen_count(mat_link,mat_node)
    N_gen = max(gen) #N° of gen


    N_f = gen.count(max(gen)) # N° of end node/link
    N_Q = len(Q)-1-N_f #Nb of debit equations 
    N_Qv = len(Q)-1 #Nb of debit variables
    N_P = len(P)-1-N_f #Nb of pressure equation NOT COUNTING THE EXTRA
    N_Pv = len(P)-1-N_f #Nb of pressure variables

    
    assert len(P) == n+1+N_f , "wrong amount of Pressures eq"
    assert len(Q) == n , "wrong amount of Debit eq"


    F=[0 for k in range(N_Qv+N_Pv)]
    dF = np.zeros((N_Qv+N_Pv,N_Qv+N_Pv))
    X = np.array( Q[1:]+P[1:len(P)-N_f] )

    P_pl0 = P_Pl(t, V, Phi) 
    if DeltaP >= 0:
        P_pl = P_pl0 - DeltaP
    if DeltaP <0:
        P_pl = P_pl0

    if DEBUG >= 1:
        print('P_pl:',P_pl+DeltaP, '\nDeltaP:',DeltaP)

    assert len(X) == N_Pv+N_Qv, "Var X len issue"
        
    
    steps = 0
    start = 1


    ########################################INITIALIZE F AND DF##############################################
    P = P[1:] #BC the index 0 is the pressure BEFORE the tracheat (above gen 0), and in the P list, index 0 = P_atm, after the trachea
    F = F[:-N_f] #We cut of the end DeltaP to append them afterward

    for i in range(N_Q):  #N of nodes equation

        node = mat_node[i+1] #No node equation to solve for the node 0 between link 0 and trachea 
        for xn in node : 
            if xn == min(node):
                F[i]+=Q[xn]
            else :
                F[i]-=Q[xn]
        
        dF[i][min(node)-1] = 1

        for j in node:
            if j != min(node):
                dF[i][j-1] = -1

    for i in range(N_Q,N_Q+N_P): # N of link equation

        if i == N_Q : #Different equation for the link of the upper trachea
        #     print(i)
        #     print(P[0])
            F[i] = P[0] - P_atm - R_ext*abs(Q[1])**r
            dF[i][0] = -r*R_ext*abs(Q[1])**(r-1)
            dF[i][N_P-1] = 1
            dF[i+1][N_P-1] = dfdPo(P[0], P[1], gen[1], Q[0], P_pl )

        else :
         

            link = mat_link[i-N_Q] #CARE +1..?


            F[i] = f( P[link[0]], P[link[1]], gen[link[1]], Q[link[1]], P_pl )
            

            dF[i][i-N_Q-1] = dfdQ( P[link[0]], P[link[1]], gen[link[1]], Q[link[1]], P_pl )

            for xl in link :

                if gen[xl] != -1 and gen[xl] != N_gen:

                    if xl == link[0]:

                        if DEBUG == 4: ######### Debugging
                            print('xl-------------',xl)
                            print(i,N_P+xl-1)   
                            print(dF[i][N_P+xl-1],xl,'C0')

                        dF[i][N_Qv+xl]=dfdPo(P[link[0]], P[link[1]], gen[link[1]], Q[link[1]], P_pl ) #CARE -1..?
                        

                    else : 

                        if DEBUG == 4: ######### Debugging
                            print('xl-------------',xl)
                            print(i,xl,N_P+xl-1)
                            print(dF[i][N_P+xl-1],xl,'C1')

                        dF[i][N_Qv+xl]=dfdPi(P[link[0]], P[link[1]], gen[link[1]], Q[link[1]], P_pl ) #CARE -1..?
                
                if gen[xl] == N_gen :
                    
                    dF[i][N_P+xl-1]=dfdPi(P[link[0]], P[link[1]], gen[link[1]], Q[link[1]], P_pl ) #eq for the inside of the link

                    dF[N_Qv+xl][N_Qv+xl] = -1

                    dF[N_Qv+xl][xl-1] = -R_lin_eq(N_gen)

                    F.append(P[xl+N_f]-P[xl]- R_lin_eq(N_gen)*Q[xl])


        
    ##################################################################################

    if DEBUG >= 1:
        print('\n first X:',X,'\n first F: ',F,'\n first dF: ',dF)
    
    def norm_corr(F,treshold,a=epsilon[0],b=epsilon[1]): #Norme personnalisée
        R=0
        for x in F[:treshold[0]]:
            R+=a*x*x
        for x in F[treshold[0]:treshold[1]]:
            R+=b*x*x
        for x in F[treshold[1]:]:
            R+=1*x*x
        return np.sqrt(R)

    N=0
    ##################################################################################
    ##################################################################################
    while norm_corr(F,[N_Q,N_Qv+N_Pv-N_f])>=stop_crit or start == 1:

        start = 0

        #update X with Newton Raphson scheme

       # dX = np.dot(np.linalg.inv(dF),F)
        dX = np.linalg.solve(dF,F)
        N+=np.linalg.norm(dX)
        if grad == 'adapt' :
            X -= dX/np.log(1+N**(2))
        if grad == 'naive' :
            X -= dX
        if grad == 'sat':
            dX_sat = [sat_fun(dx,0.0001)for dx in dX[:N_Qv]]+[sat_fun(dx,5)for dx in dX[N_Qv:]]
            X-= dX_sat
        steps += 1

        if DEBUG >= 0 :print('Step N° : \n',steps,'------------------------------------------------------')
        if DEBUG >= 2:
            print('  dX:',dX,'\n  dX Norm :',norm_corr(X,N_P),'\n \n  X:',X,'\n  X Norm :',norm_corr(X,N_P),'\n \n  F: ',F,'\n dF.dX:',np.dot(dF,dX),'\n  dF: ',dF)
            print('P_pleural',P_pl+DeltaP)

        #Update P and Q for next iteration
        P = [P_atm + P_pl]+list(X[len(Q[1:]):])+[P_alv(t,V,Phi)for k in range(N_f)]
        Q = [X[0]]+list(X[:len(Q[1:])])
        Q = [q for q in Q] #TO CORRECT HIGHER Q DUE TO LOW GEN 0.002?

        P=P[1:]
        F = F[:-N_f]
        #Update F and dF

        for i in range(N_Q):  #N of nodes equation

            node = mat_node[i+1] #No node equation to solve for the node 0 between link 0 and trachea 
            for xn in node : 
                if xn == min(node):
                    F[i]+=Q[xn]
                else :
                    F[i]-=Q[xn]
            
            dF[i][min(node)-1] = 1

            for j in node:
                if j != min(node):
                    dF[i][j-1] = -1


        for i in range(N_Q,N_Q+N_P): # N of link equation

            if i == N_Q : #Different equation for the link of the upper trachea
                #     print(i)
                #     print(P[0])
                F[i] = P[0] - P_atm - R_ext*(abs(Q[1])**r)
                # print('CORRECT',Q[1])
                dF[i][0] = -r*R_ext*abs(Q[1])**(r-1)
                dF[i][N_P-1] = 1
                dF[i+1][N_P-1] = dfdPo(P[0], P[1], gen[1], Q[0], P_pl )

            else :
         

                link = mat_link[i-N_Q] #CARE +1..?


                F[i] = f( P[link[0]], P[link[1]], gen[link[1]], Q[link[1]], P_pl )


                dF[i][i-N_Q-1] = dfdQ( P[link[0]], P[link[1]], gen[link[1]], Q[link[1]], P_pl )

                for xl in link :

                    if gen[xl] != -1 and gen[xl] != N_gen:

                        if xl == link[0]:

                            if DEBUG == 4: ######### Debugging
                                print('xl-------------',xl)
                                print(i,N_P+xl-1)   
                                print(dF[i][N_P+xl-1],xl,'C0')

                            dF[i][N_Qv+xl]=dfdPo(P[link[0]], P[link[1]], gen[link[1]], Q[link[1]], P_pl ) #CARE -1..?
                            

                        else : 

                            if DEBUG == 4: ######### Debugging
                                print('xl-------------',xl)
                                print(i,xl,N_P+xl-1)
                                print(dF[i][N_P+xl-1],xl,'C1')

                            dF[i][N_Qv+xl]=dfdPi(P[link[0]], P[link[1]], gen[link[1]], Q[link[1]], P_pl ) #CARE -1..?
                    
                    if gen[xl] == N_gen :
                        
                        dF[i][N_P+xl-1]=dfdPi(P[link[0]], P[link[1]], gen[link[1]], Q[link[1]], P_pl ) #eq for the inside of the link

                        dF[N_Qv+xl][N_Qv+xl] = -1

                        dF[N_Qv+xl][xl-1] = -R_lin_eq(N_gen)

                        F.append(P[xl+N_f]-P[xl]- R_lin_eq(N_gen)*Q[xl])
                        #print('DEBUG----',P[xl+N_f],P[xl],R_lin_eq(N_gen),Q[xl],'----')
                
                if DeltaP < 0: ##
                    P_pl = P_pl0


        if DEBUG == 3:
            if steps%10 == 0 or steps == 1:
                print('dF-1:',np.linalg.inv(dF))

        if DEBUG >= 0 :print('\n-----------------------------------------------------\n')


        if steps >= it_lim:
            break
    
    if DEBUG >= 1:
        print(' last dX:',dX,'\n last X:',X,'\n last F(X): ',F,'\n last dF(X): ',dF)
        #print('dX.dF:',np.dot(dX,dF))

    print('Number of total steps : \n',steps)
    print('Pleural Pressure : \n',P_pl)
    print('---\nEND ITERATION\n---')

    return X,[P_atm]+P,Q,P_pl,Q[0]*DeltaT

    
def expiration(links,nodes,T_series,DeltaP=100000,V_ini=CV,grad='naive',stop_crit=1e-4):

    DeltaT = T_series[0]
    V=V_ini
    c=0
    dV = 1

    gen = gen_count(links,nodes)
    gen_unique = [k for k in range(max(gen)+1)]
    id_gen = [gen.index(k) for k in gen_unique]
    P_over_time = [[] for k in gen_unique]
    D_over_time = [[] for k in gen_unique]

    NL=len(links)
    N_f = gen.count(max(gen))
    P=[P_atm for k in range(NL+1)]+[P_alv(T_series[0],V_ini,0)for k in range(N_f)] #At t = 0, we consider the lungs full, with no flow thus phi = 0 also
    Q=[0 for k in range(NL)]

    Debits=[]
    Volumes=[]
    lung_vol=[]

    print('EXPIRATION MODELISATION START','\nInitial Lung Volume:',V_ini,'\nMin Lung Volume:',V_min,'\nDelta P:',DeltaP)

    while V >= V_min and dV >=0.000001:
        t = T_series[c]
        c+=1
        iter = refresh_system(links,nodes,P,Q,t,it_lim=1000,DeltaP=DeltaP,stop_crit=stop_crit,V=V,DeltaT=DeltaT,grad=grad)
        DeltaT = T_series[c] - t

        

        if CHECK >= 2 : 
            print("DeltaT",DeltaT)

        dV = iter[-1] #volume expired during DeltaT
    
        if CHECK>=1:
            print('------------------------------------------------------','\n iteration n°',c,'\n time:',t,'\n Initial Pressure :',P,'\n Pressure :',iter[1],'\n Initial Debit :',Q,'\n Debit :',iter[2],'\n Volume before step',V ,'\n expired volume during iteration:', dV)
        if dV <= 0 :
            print('ERROR : dV negative :', dV)
            break

        for i,id in enumerate(id_gen):
            P_over_time[i].append(iter[1][id]-iter[3])
            D_over_time[i].append(D(iter[1][id]-iter[3],i)/DMAX(i))
            # print(f'P added for gen{i}' , iter[1][id])
            # print(f'D added for gen{i}' , D(iter[1][id],i))

        V-=dV #Volume remaining in lungs
        print(' Volume after step :',V)
        Expired_Volume = V_ini-V #Volume expired (total)

        lung_vol.append(V) #Lung volume at time t
        Debits.append(iter[2][0]) #Debit during the last iteration
        Volumes.append(Expired_Volume) #Total expired volume at time t

        t = T_series[c]
        P_end=P_alv(t,V,iter[2][0])
        alpha=P_end/iter[1][-1]
        P=[alpha*p for p in iter[1]]
        Q=iter[2]

        if c>=1000:
            break
        
    print('time to empty lungs :',T_series[c])

    return Debits, Volumes, lung_vol, P_over_time, D_over_time



#####################################################################################
#########################################TEST########################################
#####################################################################################

DEBUG = -1
CHECK = -1
    
#NOTES : On observe une divergence des solution pour le D non linéarisé. La divergence a lieu lorsque DeltaP, le delta de linearisation, s'approche (a 10Pa près) de P_pleural

#testing for low n in a symetric tree : #############################################
#####################################################################################

# Delta_t = 0.1
# timer=Timer()
# timer.start()

n=7
links, nodes = mat_links(n), mat_nodes(n)

# NL=len(links)

# gen = gen_count(links,nodes)
# N_gen = max(gen) 
# N_f = gen.count(max(gen))

# P=[P_atm for k in range(NL+1)]+[P_alv(Delta_t,CV,0)for k in range(N_f)] #At t = 0, we consider the lungs full, with no flow thus phi = 0 also

# Q=[0 for k in range(NL)]

# print('links:',links,'\nnodes:',nodes)
# print('links generations:',gen)
# print('initial Pressure:',P,'\ninitial Debit:',Q)

# Rez = refresh_system(links,nodes,P,Q,t=Delta_t,DeltaP=0,it_lim=1000,grad='adapt')
# print('New X---------------')
# print(Rez[0])
# # print('\n individual values :', list(set(Rez[0])),'\n')
# print('New P---------------')
# print(Rez[1])
# print('\n Individual values :', list(set(Rez[1])),'\n')
# print('New Q---------------')
# print(Rez[2])
# # print('\n individual values :', list(set(Rez[2])),'\n')
# timer.stop()
# print('Duration of execution : ', timer.interval)

# print((Rez[1][-1]-Rez[1][-17])/Rez[2][-1])
# print((128*eta*3*Dmax_0) / (np.pi * Dmax_0**4))


#testing for low n and an asymetric tree : ##########################################
#####################################################################################

# Delta_t = 0.02
# n=5
# # A la main
# links = [['atm',0],[0,1],[1,2],[1,3],[2,4],[3,5],[3,6],[4,7],[4,8],[5,9],[6,10]]
# nodes = [[0,1],[1,2,3],[2,4],[3,5,6],[4,7,8],[5,9],[6,10]]
# NL=len(links)

# gen = gen_count(links,nodes)
# N_gen = max(gen) 
# N_f = gen.count(max(gen))

# P=[P_atm for k in range(NL+1-N_f)]+[P_alv(Delta_t,CV,0)for k in range(N_f)] #At t = 0, we consider the lungs full, with no flow thus phi = 0 also

# Q=[0 for k in range(NL)]



# Rez = refresh_system(links, nodes,P,Q,Delta_t)
# print('New X---------------')
# print(Rez[0])
# print('New P---------------')
# print(Rez[1])
# print('New Q---------------')
# print(Rez[2])


# print('------------------------------')

# print(links, nodes)
# print (gen)
# print(P,Q)

#####################################################################################
################################     PLOT    ########################################
#####################################################################################

V_ini = CV
import matplotlib.pyplot as plt

################## plot for different epsilon on different graphs ###################
################## w\ pressures and diameters for each generation ###################
#####################################################################################



timer = Timer()
timer.start()

eps=1e-6
fig = plt.figure(figsize =(14, 9))
T_2 = [0.01*k for k in range(1,10)]+[0.1+0.02*k for k in range(200)]
T_opti = [0.01*k for k in range(1,4)]+[0.03 + 0.05*k for k in range(1,20) ]
T_min = [0.005*k for k in range(1,10000)]

T_l = [T_2 for k in range(4)]

# ---------------
T = T_2

Y,X,Z,PoT,DoT = expiration(links,nodes,T,V_ini=V_ini,DeltaP=0,grad='adapt',stop_crit=eps)
times = T[:len(Y)]
#We put everything in liters
Debit = [0]+[y*1000 for y in Y]
LungVol = [V_ini*1000]+[z*1000 for z in Z]
ExpVol = [0] + [x*1000 for x in X]
import matplotlib.pyplot as plt

ax = fig.add_subplot(2,3,1) #SPIROGRAPHY
ax.plot(LungVol,Debit)
ax.invert_xaxis()
ax.set_title(f'Spirography Simulation of Forced Expiration\n stopping criteria = {eps}')
ax.set_xlabel('Volume (L)')
ax.set_ylabel('Debit (L/s)')
ax.set_xlim(6,1)

ax = fig.add_subplot(2,3,2) #PRESSURE FOR EACH GEN
for j,P_list in enumerate(PoT):
    ax.plot(times,P_list,label =f'gen {j}')
ax.set_title('Pressure over time \nfor the different generations')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Pressure (Pa)')

plt.legend()

ax = fig.add_subplot(2,3,3) #DIAMETER FOR EACH GEN
for j,D_list in enumerate(DoT):
    ax.plot(times,D_list,label =f'gen {j}')
ax.set_title('D/Dmax over time \nfor the different generations')
ax.set_xlabel('Time (s)')
ax.set_ylabel('D/Dmax')

plt.legend()

# ax = fig.add_subplot(2,3,4) #Pleural pressure over time
# P_pl_list = [P_Pl(times[k],Z[k],Y[k]) for k in range(len(times))]
# ax.plot(times,P_pl_list,label = 'Pleural pressure')
# ax.set_title('Pleural pressure over time')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('P_pl(Pa)')

# plt.legend()

ax = fig.add_subplot(2,3,4) #Debit over time
ax.plot(times,Debit[:-1],label = 'Debit')
ax.set_title('Debit over time')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Debit (L.s-1)')

plt.legend()

ax = fig.add_subplot(2,3,5) #Alveolar pressure over time
P_alv_list = [P_alv(times[k],Z[k],Y[k]) for k in range(len(times))]
ax.plot(times,P_alv_list,label = 'Alveolar pressure')
ax.set_title('Alveolar pressure over time')
ax.set_xlabel('Time (s)')
ax.set_ylabel('P_al(Pa)')

plt.legend()

# ax = fig.add_subplot(2,3,5) #Pulmonar Volume over time
# ax.plot(times,LungVol[:-1],label = 'Lung Volume')
# ax.set_title('Lung Volume over time')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Lung Volume (L)')

# plt.legend()

ax = fig.add_subplot(2,3,6) #Total Resistance (Palv/Q) pressure over time
P_alv_list = [P_alv(times[k],Z[k],Y[k])/Y[k] for k in range(len(times))]
ax.plot(times,P_alv_list,label = 'Alvelolar pressure / Debit')
ax.set_title('Alveolar pressure / Debit over time _n (= total Resistance)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Total R(Pa.s/m-3)')

plt.legend()



timer.stop()
print('total expiration computing time: ', timer.interval)

plt.subplots_adjust(wspace=0.40, hspace=0.20)

plt.show()



# ###to mesure the resistance
# L = [2*k for k in range(15)]

# a = [(PoT[3][l]-PoT[4][l])/Y[l] for l in L]
# b = [(PoT[2][l]-PoT[3][l])/Y[l] for l in L]
# c = [(PoT[1][l]-PoT[2][l])/Y[l] for l in L]

# print('Linear res mesured at \n gen 3-4',a,'\n gen 2-3',b,'\n gen 1-2',c)
# ###


#################plot for different epsilon on the same graph #######################
#####################################################################################


# T = [0.01*k for k in range(1,10)]+[0.1+0.02*k for k in range(200)]
# T=T2
# Y1,X1,Z1,A,B = expiration(links,nodes,T,V_ini=V_ini,DeltaP=0,grad='adapt')
# tau = 0.1
# Y2,X2,Z2,A,B = expiration(links,nodes,T,V_ini=V_ini,DeltaP=0,grad='adapt')
# tau = 0.3
# Y3,X3,Z3,A,B = expiration(links,nodes,T,V_ini=V_ini,DeltaP=0,grad='adapt')

# D1 = [0]+[y*1000 for y in Y1]
# L1 = [V_ini*1000]+[z*1000 for z in Z1]
# D2 = [0]+[y*1000 for y in Y2]
# L2 = [V_ini*1000]+[z*1000 for z in Z2]
# D3 = [0]+[y*1000 for y in Y3]
# L3 = [V_ini*1000]+[z*1000 for z in Z3]
# fig, (ax) = plt.subplots(1, 1)
# ax.plot(L2,D2,c='g',label = 'tau=0.1');plt.legend()
# ax.plot(L3,D3,c='b',label = 'tau=0.3');plt.legend()
# ax.plot(L1,D1,c='r',label = 'tau=0.2');plt.legend()
# ax.invert_xaxis()
# ax.set_title('Simulated Forced Expiration Spirography \n for tau = 0.2, 0.1 and 0.05')
# ax.set_xlabel('Volume (L)')
# ax.set_ylabel('Debit (L/s)')
# plt.show()


# # print('------------------------------')

# import matplotlib.pyplot as plt
# X = np.linspace(-1000,1000,10000)
# Y = [sat_fun(x,10)for x in X]
# plt.plot(X,Y)
# plt.show()