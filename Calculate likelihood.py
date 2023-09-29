import numpy as np
import mpmath
from numpy import exp, cos, sin, sqrt, mean, sum, log, mean


def Series(A=[0,9,.0005], T=[0,18,.0005], n=2000, r=1e-50, dps=50):
    
    """
    Series(a=[0,9,.0005], t=[0,18,.0005], n_terms=2000, dps=50)
    
    Calculates the series used in the likelihood function of the CDM.
    
    Parameters
    ----------
    A: list
        A list of real numbers as [lower bound of the range, higher bound of the range, step size] indicating the range of the decision criterion 
        at which the series needs to be calculated 

    T: list
        A list of real numbers as [lower bound of the range, higher bound of the range, step size] indicating the range of the response time
        at which the series needs to be calculated.
        
    n: int
        Maximum number of terms that the series will be calculated.

    r: float
        Criterion for relative value of the terms which terminates the calculation process.
    
    dps: int
        The precision of the calculations as number of decimal places.
        

    Returns
    -------
    out : numpy array
        A 2D array of calculated values of the series for the range of decision criterion and response time values.


    Notes
    -----
    The higher precision is needed only in the calculation pf the series. So at the end, the calculated values could be transformed 
    to the default floating point precision of the Python.
    """

    mpmath.mp.dps = dps
    
    j0 = np.empty(n, dtype=mpmath.mpf)
    J1 = np.empty(n, dtype=mpmath.mpf)
    
    for i in range(n):
        j0[i] = mpmath.besseljzero(mpmath.mpf(0),i+1)
        J1[i] = mpmath.besselj(mpmath.mpf(1),j0[i])

    a = np.arange(A[0],A[1],A[2])
    t = np.arange(T[0],T[1],T[2])
    P = np.empty((len(a),len(t)))
    for i in range(len(a)):
        for j in range(len(t)):
            a_ = mpmath.mpf(a[i])
            t_ = mpmath.mpf(t[j])
            a2 = a_**2
            series = 0
            for k in range(n):
                term = j0[k]/J1[k]*mpmath.exp(-j0[k]**2*t_/2/a2)
                series += term
                if abs(term/series)<r:
                    break
            series = series/a2/2/mpmath.pi
            if series<0:
                series = 0
            P[i,j] = series
            
    return P




def CDM(params, data, A, T, P):
    
    """
    CDM(params, data)
    
    Calculate the log-likelihood value of the data.
    
    Parameters
    ----------
    params : list
        A list of real numbers containing the parameter values as [decision criterion, multiplier for decision criterion range, 
        drift length, drift angle, multiplier for the standard deviation of radial component of the drift, multiplier for the 
        standard deviation of tangental component of the drift, non-decision time, range of the non-decision time variability]

    data: numpy array
        An array of shape (2,n) containing the choice angle and responce time pairs for all trials.

    A: list
        A list of real numbers as [lower bound of the range, higher bound of the range, step size] indicating the range of the decision criterion 
        at which the series is calculated 

    T: list
        A list of real numbers as [lower bound of the range, higher bound of the range, step size] indicating the range of the response time
        at which the series is calculated.

    P: numpy array
        A 2D array of calculated values of the series for the range of decision criterion and response time values.

    Returns
    -------
    out : float
        logarithm of the likelihood value.


    Notes
    -----
    'dt' and 'da' are the step size in which the decision criterion and response time is chosen in calculation of 'P'.
    They are used as the step size in the calculation of the integrals.
    """

    
    a = params[0]
    sa = params[1]*a
    v = params[2]
    bias = params[3]
    eta1 = params[4]*v
    eta2 = params[5]*v
    t0 = params[6]
    st = params[7]
    CA,RT = data
    a_ = np.arange(A[0],A[1],A[2])
    t_ = np.arange(T[0],T[1],T[2])
    da = A[2]
    dt = T[2]
    if int((min(RT)-t0+st/2)/dt)<=0:
        return -np.inf
    ll = 0
    for i in range(len(CA)):
        A_ = a_[int((a-sa/2)/da):int((a+sa/2)/da)+1]
        if RT[i]>=t0+st/2:
            T_ = t_[int((RT[i]-t0-st/2)/dt):int((RT[i]-t0+st/2)/dt)+1]
            P_ = P[int((a-sa/2)/da):int((a+sa/2)/da)+1, int((RT[i]-t0-st/2)/dt):int((RT[i]-t0+st/2)/dt)+1]
            mT = 1
        else:
            T_ = t_[0:int((RT[i]-t0+st/2)/dt)+1]
            P_ = P[int((a-sa/2)/da):int((a+sa/2)/da)+1, 0:int((RT[i]-t0+st/2)/dt)+1]
            mT = (int((RT[i]-t0+st/2)/dt)+1) / (int((RT[i]-t0+st/2)/dt) - int((RT[i]-t0-st/2)/dt)+1)
        T_ = T_.reshape((1,len(T_)))
        A_ = A_.reshape((len(A_),1))
        Z = exp((-v**2*T_+A_**2*cos(CA[i]-bias)**2*eta1**2+2*v*A_*cos(CA[i]-bias))/(eta1**2*T_+1)/2 + A_**2*sin(CA[i]-bias)**2*eta2**2/(eta2**2*T_+1)/2)/sqrt(eta1**2*T_+1)/sqrt(eta2**2*T_+1)
        Z[P_==0] = 0
        ll += log(mean(Z*P_)*mT)
    return ll


