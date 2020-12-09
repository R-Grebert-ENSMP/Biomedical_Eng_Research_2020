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

Compliance_param  = [ [0,0.882,0.0108,1,10],
                    [1,0.882,0.0294,1,10],
                    [2,0.686,0.050,1,10],
                    [3,0.546,0.078,1,10],
                    [4,0.428,0.098,1,10],
                    [5,0.337,0.123,1,10],
                    [6,0.265,0.139,1,10],
                    [7,0.208,0.156,1,10],
                    [8,0.164,0.171,1,10],
                    [9,0.129,0.180,1,10],
                    [10,0.102,0.190,1,10],
                    [11,0.080,0.202,1,9],
                    [12,0.063,0.214,1,8],
                    [13,0.049,0.221,1,8],
                    [14,0.039,0.228,1,8],
                    [15,0.031,0.234,1,7] ]

eta=1.8e-05          #Viscosité de l'air (Pa.s)
rho=1.14             #Masse volumique de l'air (kg/m^3)
P_atm = 1013e2       #Atmospheric pressure
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



def P_alv(t,VL,Phi):
    """Computes the initial pleural pression for a last-gen node (to determine Initial Conditions)
    """
    P_al = Pm*(1-np.exp(-t/tau))*((VL-RV)/CV)-(R*Phi)
    return P_al

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

def D4(P,g):
    return D(P,g)**4


def dD(P,g):
    '''Returns the derivative of D on P
    '''
   
    Dmax = Dmax_0 * h**g
    a_0,n_1,n_2 = Compliance_param[g][1],Compliance_param[g][3],Compliance_param[g][4]
    P_1 = Compliance_param[g][1]*Compliance_param[g][3]/Compliance_param[g][2]
    P_2 = -(1-Compliance_param[g][1])*Compliance_param[g][4]/Compliance_param[g][2]

    if P<0:
        return Dmax*np.sqrt(a_0)*(n_1/P_1)*(1-P/P_1)**(-n_1-1)*(1/(2*np.sqrt((1-P/P_1)**-n_1)))
    else : 
        return Dmax*(1-a_0)*(n_2/P_2)*(1-P/P_2)**(-n_2-1)*(1/(2*np.sqrt((1-P/P_2)**-n_2)))
    # return(Dmax)


def f(Po,Pi,g,Q):
    """Function to solve = 0 for each node to find the local Pi, Po and Q of the link between
    """
    L = 3*Dmax_0*h**g
    Re = 4*rho*Q/(eta*np.pi*(D(Po,g)+D(Pi,g))/2)
    return np.array(  quad(D4,Pi,Po,args=g)[0]-32*((rho*Q*(1/np.pi))**2)*np.log(D(Po,g)/D(Pi,g))+(128*eta*L*Q*(1/np.pi))*(1.5+0.0035*Re) )


def dfQ(Po,Pi,g,Q):
    '''returns df/dQ
    '''
    L = 3*Dmax_0*h**g
    Re = 4*rho*Q/(eta*np.pi*(D(Po,g)+D(Pi,g))/2)
    return np.array(  -32*2*Q*((rho*(1/np.pi))**2)*np.log(D(Po,g)/D(Pi,g))+(128*eta*L*(1/np.pi))*(1.5+0.0035*Re) )

def dfPo(Po,Pi,g,Q):
    '''returns df/dPo
    '''
    return np.array( D4(Po,g)-32*((rho*Q*(1/np.pi))**2)*dD(Po,g)/D(Po,g) )

def dfPi(Po,Pi,g,Q):
    '''returns df/dPi
    '''
    return np.array( -D4(Pi,g)+32*((rho*Q*(1/np.pi))**2)*dD(Pi,g)/D(Pi,g) )


def refresh_system(mat_link,mat_node,P_ini,Q_ini,t=0,Phi = 0,epsilon=1e-3):
    """Computes the state of the system for 1 temporal iteration 

    P_ini, Qini = set of value for P and Q 
    """
   
    
    n = len(mat_link) #N° of nodes and intersections
    P = P_ini.copy() #will contain the pressures at time t
    Q = Q_ini.copy() #will contain the debit at time t
    gen = gen_count(mat_link,mat_node)
    N_gen = max(gen) #N° of gen
    # print(N_gen)

    N_f = gen.count(max(gen)) # N° of end node/link
    N_Q = len(Q)-1-N_f #Nb of debit equations = Nb of pressure variables:
    N_P = len(P)-2 #Nb of pressure equations = Nb of debit variables

    assert len(P) == n+1 , "wrong amount of Pressures"
    assert len(Q) == n , "wrong amount of Debit"


    F=[0 for k in range(N_Q+N_P)]
    dF = np.zeros((N_Q+N_P,N_Q+N_P))
    X = np.array( Q[1:]+P[2:len(P)-N_f] )
    X0=X.copy()

    assert len(X) == N_P+N_Q, "Var X len issue"
        
    
    steps = 0

    while np.linalg.norm(X)>=epsilon:

    #First we update F and dF
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
            
            link = mat_link[i+1-N_Q] #CARE +1..?
            F[i] = f( P[link[0]], P[link[1]], gen[link[1]], Q[link[1]] )
            # print(link)
            # print(P)
            # print(P[link[0]],P[link[1]],gen[link[1]])
            # print(f(P[link[0]],P[link[1]],gen[link[1]],Q[link[1]]))
            # print(F[i])

            dF[i][i-N_Q] = dfQ( P[link[0]], P[link[1]], gen[link[1]], Q[link[1]] )

            for xl in link :
                if gen[xl] != -1 and gen[xl] != N_gen:
                    if xl == link[0]:
                        dF[i][N_P+xl-1]=dfPo(P[link[0]], P[link[1]], gen[link[1]], Q[link[1]] ) #CARE +1..?
                        # print(dF[i][N_P+xl-1],xl,'C0')
                        # print(i,N_P+xl-1)
                    else : 
                        dF[i][N_P+xl-1]=dfPi(P[link[0]], P[link[1]], gen[link[1]], Q[link[1]] ) #CARE +1..?
                        # print(dF[i][N_P+xl-1],xl,'C1')
                        # print(i,N_P+xl-1)



        #print(F,dF)

        #update X with Newton Raphson scheme
        dX = np.dot(np.linalg.inv(dF),F)
        X-= dX
        steps+=1

        #Update P and Q for next iteration
        P = [P_atm,P_atm - R_ext*Phi**r]+list(X[len(Q[1:]):])+[P_alv(t,CV,Phi)for k in range(N_f)] #CARE need to update VL(t)
        Q = [X[0]]+list(X[:len(Q[1:])])

        # if steps%1000==0 or steps == 1:
        #     print(steps,'\n dX:',X,'\n X:',dX,'\n F: ',F,'\n dF: ',dF)

        if steps >= 10e2:
            break
    
    
    return X,X0,P,Q

    



    


#testing for low n in a symetric tree
Delta_t = 0.1
n=5
links, nodes = mat_links(n), mat_nodes(n)
NL=len(links)

gen = gen_count(links,nodes)
N_gen = max(gen) 
N_f = gen.count(max(gen))

P=[P_atm for k in range(NL+1-N_f)]+[P_alv(Delta_t,CV,0)for k in range(N_f)] #At t = 0, we consider the lungs full, with no flow thus phi = 0 also

Q=[0 for k in range(NL)]


Rez = refresh_system(links, nodes,P,Q,Delta_t)
print('New X---------------')
print(Rez[0])
print('X before descent----')
print(Rez[1])
print('New P---------------')
print(Rez[2])
print('New Q---------------')
print(Rez[3])


print('------------------------------')

print(links, nodes)
print (gen)
print(P,Q)

#Testint for low n and an asymetric tree : 