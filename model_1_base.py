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
h = 2**(-1/3)
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

def D(P,g):
    '''Compliance of a branch (link)
    returns the diameters D of the branch in function of the local pressure and generation
    '''
  
    Dmax = Dmax_0 * h**g
    a_0,n_1,n_2 = Compliance_param[g][1],Compliance_param[g][3],Compliance_param[g][4]
    P_1 = Compliance_param[g][1]*Compliance_param[g][3]/Compliance_param[g][2]
    P_2 = -(1-Compliance_param[g][1])*Compliance_param[g][4]/Compliance_param[g][2]

    if P<0:
        return Dmax*np.sqrt(a_0*(1-P/P_1)**-n_1)
    else : 
        return Dmax*np.sqrt(1-(1-a_0)*(1-P/P_2)**-n_2)
    # return(Dmax)

def D4(P,g):
    return D(P,g)**4

def dDdP(P,g):
    '''Returns the derivative of D on P
    '''
   
    Dmax = Dmax_0 * h**g
    a_0,n_1,n_2 = Compliance_param[g][1],Compliance_param[g][3],Compliance_param[g][4]
    P_1 = Compliance_param[g][1]*Compliance_param[g][3]/Compliance_param[g][2]
    P_2 = -(1-Compliance_param[g][1])*Compliance_param[g][4]/Compliance_param[g][2]

    if P<0:
        return Dmax*np.sqrt(a_0)*(n_1/P_1)*((1-P/P_1)**(-n_1-1))*(1/(2*np.sqrt((1-P/P_1)**-n_1)))
    else : 
        return -Dmax*(1-a_0)*(n_2/P_2)*((1-P/P_2)**(-n_2-1))*(1/(2*np.sqrt(1-(1-a_0)*((1-P/P_2)**-n_2))))
    # return(0)


def f(Po,Pi,g,Q,P_plr,eps=0):
    """Function to solve = 0 for each node to find the local Pi, Po and Q of the link between
    """
    L = 3*Dmax_0*h**g
    Re = 4*rho*Q/(eta*np.pi*(D(Po-P_plr,g)+D(Pi-P_plr,g))/2)
    return np.array(  quad(D4,Pi-P_plr,Po-P_plr,args=g)[0]-32*((rho*Q*(1/np.pi))**2)*np.log(D(Po-P_plr,g)/D(Pi-P_plr,g))+(128*eta*L*Q*(1/np.pi))*(1.5+eps*Re) )

def debug_f(Po,Pi,g,Q,P_plr,eps=0):
    """Function to solve = 0 for each node to find the local Pi, Po and Q of the link between
    """
    L = 3*Dmax_0*h**g
    Re = 4*rho*Q/(eta*np.pi*(D(Po-P_plr,g)+D(Pi-P_plr,g))/2)
    a=quad(D4,Pi-P_plr,Po-P_plr,args=g)[0]
    b=-32*((rho*Q*(1/np.pi))**2)*np.log(D(Po-P_plr,g)/D(Pi-P_plr,g))
    c=(128*eta*L*Q*(1/np.pi))*(1.5+eps*Re) 
    print('int D4 :', a,'\n log funtion : ',b,'\n loss term: ',c)
    print('assert that residue is negligeable in linear case : ',(abs(a)-abs(c))/(abs(a)+abs(c)))
    return

def dfdQ(Po,Pi,g,Q,P_plr,eps=0):
    '''returns df/dQ
    '''
    L = 3*Dmax_0*h**g
    Re = 4*Q*rho/(eta*np.pi*(D(Po-P_plr,g)+D(Pi-P_plr,g))/2)
    return np.array(  -64*Q*((rho*(1/np.pi))**2)*np.log(D(Po-P_plr,g)/D(Pi-P_plr,g))+(128*eta*L*(1/np.pi))*(1.5+2*eps*Re) )

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

def refresh_system(mat_link,mat_node,P_ini,Q_ini,t=0,DeltaP=100000,SAT=False,epsilon=[1e-2,1e-6],it_lim=1e3):  
    """Computes the state of the system for 1 temporal iteration 

    P_ini, Qini = set of value for P and Q 

    epsilon : the precision we want on F=0, first digit for the kirshoff equation and second for the flow one

    it_lim : max amount of iterations

    DeltaP : Linearises the D(P) :  if DeltaP if very high, the behaviour of D(P) is asymptotical and D(P) = Dmax, dDdP(P) = 0.
    If it is set to 0, we have full non linear behavior. Lim of divergence aroud P_Pl
    """
    Phi = Q_ini[0]
    
    n = len(mat_link) #N° of nodes and intersections
    P = P_ini.copy() #will contain the pressures at time t
    Q = Q_ini.copy() #will contain the debit at time t
    gen = gen_count(mat_link,mat_node)
    N_gen = max(gen) #N° of gen

    N_f = gen.count(max(gen)) # N° of end node/link
    N_Q = len(Q)-1-N_f #Nb of debit equations = Nb of pressure variables:
    N_P = len(P)-2 #Nb of pressure equations = Nb of debit variables

    assert len(P) == n+1 , "wrong amount of Pressures"
    assert len(Q) == n , "wrong amount of Debit"


    F=[0 for k in range(N_Q+N_P)]
    dF = np.zeros((N_Q+N_P,N_Q+N_P))
    X = np.array( Q[1:]+P[2:len(P)-N_f] )

    sat = np.array([0.1 for k in range(len(Q[1:]))] + [10 for k in range(2,len(P)-N_f)])

    P_pl = P_Pl(t, CV, Phi) - DeltaP

    if DEBUG >= 1:
        print('P_pl:',P_pl+DeltaP, '\nDeltaP:',DeltaP)

    assert len(X) == N_P+N_Q, "Var X len issue"
        
    
    steps = 0
    start = 1


    ########################################INITIALIZE F AND DF##############################################
    P = P[1:] #BC the index 0 is the pressure BEFORE the tracheat (above gen 0), and in the P list, index 0 = P_atm, after the trachea

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

        # if i == N_Q : #Different equation for the link of the upper trachea
        # #     print(i)
        # #     print(P[0])
        #     F[i] = P[0] - P_atm - R_ext*abs(Q[1])**r
        #     dF[i][0] = -r*R_ext*abs(Q[1])**(r-1)
        #     dF[i][N_P-1] = 1

        link = mat_link[i+1-N_Q] #CARE +1..?

        F[i] = f( P[link[0]], P[link[1]], gen[link[1]], Q[link[1]], P_pl )
        

        dF[i][i-N_Q] = dfdQ( P[link[0]], P[link[1]], gen[link[1]], Q[link[1]], P_pl )

        for xl in link :

            if gen[xl] != -1 and gen[xl] != N_gen:

                if xl == link[0]:

                    if DEBUG == 4: ######### Debugging
                        print('xl-------------',xl)
                        print(i,N_P+xl-1)   
                        print(dF[i][N_P+xl-1],xl,'C0')

                    dF[i][N_P+xl-1]=dfdPo(P[link[0]], P[link[1]], gen[link[1]], Q[link[1]], P_pl ) #CARE -1..?
                    

                else : 

                    if DEBUG == 4: ######### Debugging
                        print('xl-------------',xl)
                        print(i,xl,N_P+xl-1)
                        print(dF[i][N_P+xl-1],xl,'C1')

                    dF[i][N_P+xl-1]=dfdPi(P[link[0]], P[link[1]], gen[link[1]], Q[link[1]], P_pl ) #CARE -1..?
    
    ##################################################################################

    if DEBUG >= 1:
        print('\n first X:',X,'\n first F: ',F,'\n first dF: ',dF)
    
    def norm_corr(F,treshold,a=epsilon[0],b=epsilon[1]): #Norme personnalisée
        R=0
        for x in F[:treshold]:
            R+=a*x*x
        for x in F[treshold:]:
            R+=b*x*x
        return np.sqrt(R)

    ##################################################################################
    ##################################################################################
    while norm_corr(F,N_Q+1)>=1e-4 or start == 1:

        start = 0

        #update X with Newton Raphson scheme
        dX = np.dot(np.linalg.inv(dF),F)
        X -= dX
        steps += 1

        print('Step N° : \n',steps,'------------------------------------------------------')
        if DEBUG >= 2:
            print('  dX:',dX,'\n  X:',X,'\n  F: ',F,'\n  dF: ',dF)
            print('dF.dX:',np.dot(dF,dX))

        #Update P and Q for next iteration
        P = [P_atm + P_pl, P_atm - R_ext*Phi**r]+list(X[len(Q[1:]):])+[P_alv(t,CV,Phi)for k in range(N_f)] #CARE need to update VL_t
        Q = [X[0]]+list(X[:len(Q[1:])])
        V_t = Delta_t*X[0] #Volume exhaled during this step

        P=P[1:]

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

            # if i == N_Q : #Different equation for the link of the upper trachea
            # #     print(i)
            # #     print(P[0])
            #     F[i] = P[0] - P_atm - R_ext*abs(Q[1])**r
            #     dF[i][0] = -r*R_ext*abs(Q[1])**(r-1)
            #     dF[i][N_P-1] = 1

            link = mat_link[i+1-N_Q] #CARE +1..?

            if DEBUG == 5 :
                debug_f(P[link[0]], P[link[1]], gen[link[1]], Q[link[1]], P_pl )
                


            F[i] = f( P[link[0]], P[link[1]], gen[link[1]], Q[link[1]], P_pl )
         
            dF[i][i-N_Q] = dfdQ( P[link[0]], P[link[1]], gen[link[1]], Q[link[1]], P_pl )


            for xl in link :
                if gen[xl] != -1 and gen[xl] != N_gen:
                    if xl == link[0]:
                        if DEBUG == 4: ######### Debugging
                            print('xl-------------',xl)
                            print(i,N_P+xl-1)   
                            print(dF[i][N_P+xl-1],xl,'C0')
                        dF[i][N_P+xl-1]=dfdPo(P[link[0]], P[link[1]], gen[link[1]], Q[link[1]], P_pl ) #CARE -1..?
                    else : 
                        if DEBUG == 4: ######### Debugging
                            print('xl-------------',xl)
                            print(i,xl,N_P+xl-1)
                            print(dF[i][N_P+xl-1],xl,'C1')
                        dF[i][N_P+xl-1]=dfdPi(P[link[0]], P[link[1]], gen[link[1]], Q[link[1]], P_pl ) #CARE -1..?



        if DEBUG == 3:
            if steps%10 == 0 or steps == 1:
                print('dF-1:',np.linalg.inv(dF))

        print('------------------------------------------------------')


        if steps >= it_lim:
            break
    
    if DEBUG >= 1:
        print('Number of total steps : \n',steps )
        print(' last dX:',dX,'\n last X:',X,'\n last F(X): ',F,'\n last dF(X): ',dF)
        #print('dX.dF:',np.dot(dX,dF))


    return X,[P_atm]+P,Q,V_t

    


#####################################################################################
#########################################TEST########################################
DEBUG = 5
    
#NOTES : On observe une divergence des solution pour le D non linéarisé. La divergence a lieu lorsque DeltaP, le delta de linearisation, s'approche (a 10Pa près) de P_pleural

#testing for low n in a symetric tree : ###########################################

Delta_t = 0.1
# n=2
# links, nodes = mat_links(n), mat_nodes(n)
# NL=len(links)

# gen = gen_count(links,nodes)
# N_gen = max(gen) 
# N_f = gen.count(max(gen))

# P=[P_atm for k in range(NL+1-N_f)]+[P_alv(Delta_t,CV,0)for k in range(N_f)] #At t = 0, we consider the lungs full, with no flow thus phi = 0 also

# Q=[0 for k in range(NL)]

# print('links:',links,'\nnodes:',nodes)
# print('links generations:',gen)
# print('initial Pressure:',P,'\ninitial Debit:',Q)

# Rez = refresh_system(links,nodes,P,Q,Delta_t)
# print('New X---------------')
# print(Rez[0])
# print('New P---------------')
# print(Rez[1])
# print('New Q---------------')
# print(Rez[2])


print('------------------------------')


#Testint for low n and an asymetric tree : ###########################################

Delta_t = 0.1
n=5
# A la main
links = [['atm',0],[0,1],[1,2],[1,3],[2,4],[3,5],[3,6],[4,7],[4,8],[5,9],[6,10]]
nodes = [[0,1],[1,2,3],[2,4],[3,5,6],[4,7,8],[5,9],[6,10]]
NL=len(links)

gen = gen_count(links,nodes)
N_gen = max(gen) 
N_f = gen.count(max(gen))

P=[P_atm for k in range(NL+1-N_f)]+[P_alv(Delta_t,CV,0)for k in range(N_f)] #At t = 0, we consider the lungs full, with no flow thus phi = 0 also

Q=[0 for k in range(NL)]



Rez = refresh_system(links, nodes,P,Q,Delta_t)
print('New X---------------')
print(Rez[0])
print('New P---------------')
print(Rez[1])
print('New Q---------------')
print(Rez[2])


# print('------------------------------')

# print(links, nodes)
# print (gen)
# print(P,Q)