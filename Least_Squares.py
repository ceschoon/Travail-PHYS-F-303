import numba,pandas
from numba import jit
import numpy as np

@jit
def moindres_carres(x,y,A_test,X0_test,C_test):
    """ Finds the best fitting parabola for the set of points (x,y)
   
    Parameters:
    ----------
    x : x-coordinates of the points to fit
    y : y-coordinates of the points to fit
    A_test  : test coefficent in y = A*(x-X0)**2+C
    X0_test : test coefficent in y = A*(x-X0)**2+C
    C_test  : test coefficent in y = A*(x-X0)**2+C
    
    Returns:
    -------
    A  : coefficent of the best fitting parabola
    X0 : coefficent of the best fitting parabola
    C  : coefficent of the best fitting parabola
    """
    
    if len(x)!=len(y):
        print("x and y must have the same size")
        return
    
    N = len(x)
    
    # on calcule la sommes des erreurs au carré pour chaque parabole test
    S = np.zeros((len(A_test),len(X0_test),len(C_test)))
    for i in range(len(A_test)):
        for j in range(len(X0_test)):
            for k in range(len(C_test)):
                for l in range(N):
                    ytest = A_test[i]*(x[l]-X0_test[j])**2+C_test[k]
                    S[i,j,k] += (y[l] - ytest)**2
                
    # on cherche la parabole minimisant cette somme
    [i,j,k] = np.where(S==np.min(S))
    A  =  A_test[i[0]]
    X0 = X0_test[j[0]]
    C  =  C_test[k[0]]
    
    # on renvoie une erreur si la parabole minimisante est une parabole limite (intervalle considéré trop petit),
    # sauf dans le cas ou le nombre de paramètres a tester est 1, on considère que c'est dans ce cas intentionnel
    if ((i==0)|(i==len(A_test)-1))&(len(A_test)!=1):
        print('The best fitting parabola is not one of the tested parabola, \
        try a wider range of coefficents')
    if ((j==0)|(j==len(X0_test)-1))&(len(X0_test)!=1):
        print('The best fitting parabola is not one of the tested parabola, \
        try a wider range of coefficents')
    if ((k==0)|(k==len(C_test)-1))&(len(C_test)!=1):
        print('The best fitting parabola is not one of the tested parabola, \
        try a wider range of coefficents')
    
    return [A,X0,C]

@jit
def moindres_carres_avec_erreur(filename,A_test,X0_test,C_test):
    """ Finds the alpha that maximises a parabolic fit of the top of the heat capacity curve stored in the 
        file at "filename", as well as the error associated
   
    Parameters:
    ----------
    filename : name of the file where the data is stored
    A_test  : test coefficent in y = A*(x-X0)**2+C
    X0_test : test coefficent in y = A*(x-X0)**2+C
    C_test  : test coefficent in y = A*(x-X0)**2+C
    
    Returns:
    -------
    alpha_max  : alpha value that maximises the fit
    err_alpha_max  : the error on alpha_max (including both the statistical error related to the variance of the data 
                     and the numerical error induced by the method used here)
    """
    
    # D'abord, on lit les données du fichier donné
    
    dataIsing = pandas.read_csv(filename)
    [alphas, Ms, Es, VarEs] = np.transpose(dataIsing.as_matrix(['Couplage', '<Magnetisation>', \
                                                            '<Energie>', 'Var(Energie)']))

    indices = np.argsort(alphas)
    alphas = alphas[indices]
    Ms = Ms[indices]
    Es = Es[indices]
    VarEs = VarEs[indices]
    Cvs = VarEs
    
    # Ensuite, on applique la méthode des moindres carrés implémentée plus haut pour fitter le sommet de la courbe
    # de capacité calorifique avec une parabole
    
    indices = np.where((alphas>0.41)&(alphas<0.46))

    [A_fit,X0_fit,C_fit] = moindres_carres(alphas[indices],Cvs[indices],A_test,X0_test,C_test)
    print("Best fitting parameters: {:.4f} {:.4f} {:.4f}".format(A_fit,X0_fit,C_fit))

    alpha_max = X0_fit
    err_num = abs(X0_test[1]-X0_test[0])
    
    # Ensuite, on calcule l'erreur sur ce résultat en réitérant la procédure de fit sur les données bruitées,
    # avec un bruit d'une amplitude semblable à la variance des données.
    
    repet = 50
    alpha_max_vec = np.zeros((repet))

    A_fit_original  = A_fit
    X0_fit_original = X0_fit
    C_fit_original  = C_fit

    for i in range(repet):
    
        #print("loop {:d} of {:d}".format(i,repet))
    
        Cvs_bruit = Cvs + np.random.uniform(-1,1,len(VarEs))*3e-4 # amplitude plausible -> vérifiée systématiquement sur les graphiques !

        indices = np.where((alphas>0.41)&(alphas<0.46))

        A_test  = np.array([A_fit_original])
        #X0_test = np.linspace(0.430,0.440,51)  # variation utile que pour le paramètre le plus pertinent (domine)
        C_test  = np.array([C_fit_original])

        [A_fit,X0_fit,C_fit] = moindres_carres(alphas[indices],Cvs_bruit[indices],A_test,X0_test,C_test)
        #print(A_fit,X0_fit,C_fit)

        alpha_max_vec[i] = X0_fit
    
    err_stat  = np.std(alpha_max_vec)
    err_alpha_max = np.sqrt(err_stat**2 + err_num**2)
    
    print("Le maximum de la parabole de fit se trouve en alpha = {:.5f} +- {:.5f}".format(alpha_max, err_alpha_max))
    print("Erreur numérique: {:.5f} - Erreur Statistique: {:.5f}".format(err_num,err_stat))
    print("--------------------------------------------------------------------")
    
    return [alpha_max, err_alpha_max]