import numpy as np

def SimCDM(params, dt=.001, n=100):
    
    """
    SimCDM(params, dt=.001, n=100)
    
    Simulate a dataset from the Circular Diffusion Model.
    
    Parameters
    ----------
    params : list
        A list of real numbers containing the parameter values as [decision criterion, multiplier for decision criterion range, 
        drift length, drift angle, multiplier for the radial component of the drift, multiplier for the tangental component of the drift,
        non-decision time, range of the non-decision time variability]

    dt: float
        The time step.

    n: int
        The number of trials.
        

    Returns
    -------
    out : numpy array
        An array of shape (2,n) containing the choice angle and responce time pairs for all trials.


    Notes
    -----
    The variability on the radial and tangental components of the drift vector is considered to follow the Normal distribution. The 
    variability on the decision criterion and non-decision time is considered to follow the Uniform distribution.

    Set variability parameter values to zero to simulate data from the simple CDM.
    """
    
    a = params[0] 
    sa = a*params[1]
    v = params[2] 
    bias = params[3]
    eta1 = v*params[4] 
    eta2 = v*params[5] 
    t0 = params[6] 
    st = params[7]
    hit = False
    x = [0,0]
    t = 0
    RT = []
    CA = []
    for i in range(n):
        A = np.random.uniform(a-sa/2, a+sa/2)
        A2 = A**2
        drift = np.random.normal([v,0], [eta1,eta2])
        V = np.array([drift[0]*np.cos(bias)-drift[1]*np.sin(bias), drift[0]*np.sin(bias)+drift[1]*np.cos(bias)])
        T = np.random.uniform(t0-st/2, t0+st/2)
        while not hit:
            x += np.random.normal(V*dt, [np.sqrt(dt),np.sqrt(dt)])
            t += dt
            hit = x[0]**2 + x[1]**2 >= A2
        else:
            RT.append(t+T)
            CA.append(np.arctan2(x[1],x[0]))
            x = [0,0]
            t = 0
            hit = False
    return np.array([CA, RT])
