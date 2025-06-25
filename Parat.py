import math
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mpm
mpm.mp.dps = 20
#import csv
#from operator import add    
#import ROOT
#from calcrate2 import CalcRate
#from matplotlib.colors import LogNorm
#import concurrent.futures
#import multiprocessing

def heaviside(x):
    return 1.0 * (x >= 0)

def g_nu_mu(x, r):
    return (3 - 2 * r) / (9 * (1 - r)**2) * (9 * x**2 - 6 * mpm.log(x) - 4 * x**3 - 5)

def h1_nu_mu(r):
    return (3 - 2 * r) / (9 * (1 - r)**2) * (9 * r**2 - 6 * mpm.log(r) - 4 * r**3 - 5)

def h2_nu_mu(x, r):
    return (1 + 2 * r) * (r - x) / (9 * r**2) * (9 * (r + x) - 4 * (r**2 + r * x + x**2))

def f_nu_mu(x):
    r = 0.573 # constant value of r = 1 - lambda = (m_mu/m_pi)^2
    term1 = g_nu_mu(x, r) * heaviside(x - r)
    term2 = (h1_nu_mu(r) + h2_nu_mu(x, r)) * heaviside(r - x)
    return term1 + term2

def g_nu_e(x, r):
    return 2 / (3 * (1 - r)**2) * ((1-x)*(6 * (1 - x)**2 + r * (5 + 5 * x - 4 * x**2)) + 6 * r * mpm.log(x))

def h1_nu_e(r):
    return (2 / (3 * (1 - r)**2)) * ((1 - r) * (6 - 7 * r + 11 * r**2 - 4 * r**3) + 6 * r * mpm.log(r))

def h2_nu_e(x, r):
    return (2 * (r - x)) / (3 * r**2) * (7 * r**2 - 4 * r**3 + 7 * x * r - 4 * x * r**2 - 2 * x**2 - 4 * x**2 * r)

def f_nu_e(x):
    r = 0.573 # constant value of r = 1 - lambda = (m_mu/m_pi)^2
    term1 = g_nu_e(x, r) * heaviside(x - r)
    term2 = (h1_nu_e(r) + h2_nu_e(x, r)) * heaviside(r - x)
    return term1 + term2

def gaussian(x, A, sigma):
    return A * np.exp(-0.5 * ((x) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)

def integrand_e(x):
    return x * f_nu_e(x)
def integrand_mu(x):
    return x * f_nu_mu(x)

def rejection_sampling_e():
    accepted = -1
    while accepted <= 0:
        x = np.abs(float(np.random.normal(0, 0.37)))  # Sampling from a normal distribution
        if 0 <= x <= 1:
            y = float(np.random.uniform(0, float(gaussian(x,2.28,0.37))))
            if y <= f_nu_e(x):
                accepted = x
    return accepted

def rejection_sampling_mu():
    accepted = -1
    while accepted <= 0:
        x = np.abs(float(np.random.normal(0, 0.37)))  # Sampling from a normal distribution
        if 0 <= x <= 1:
            y = float(np.random.uniform(0, float(gaussian(x,2.28,0.37))))
            if y <= f_nu_mu(x):
                accepted = x
    return accepted


class Parameters:
    def __init__(self, kineticEnergy, ejectaMass, Dstar, delta, epsCR, s, epsB, Rstar,Rw=1e20):
        self.kineticEnergy = kineticEnergy
        self.ejectaMass = ejectaMass
        self.w = 2
        self._Dstar =Dstar
        self.Rw=Rw
        self.delta = delta
        self.epsCR = epsCR
        self.s = s
        self.epsB = epsB
        self.Rstar = Rstar

    @property
    def Dstar(self):
        return self._Dstar
    
    @Dstar.setter
    def Dstar(self, value):
        self._Dstar = value
    @property
    def D(self):
        return (5*(10**16)) * self._Dstar #g cm^-1
    @property
    def X(self):
        return (((3-self.w)*(4-self.w))**(1/(self.delta-self.w)))*((10*(self.delta-5))**((self.delta-3)/(2*(self.delta-self.w))))*((4*mpm.pi*(self.delta-4)*(self.delta-3)*self.delta)**((-1)/(self.delta-self.w)))*((3*(self.delta-3))**(-(self.delta-5)/(2*(self.delta-self.w))))
    
    def shockRadius(self, t):
        return self.X*self.D**(-1/(self.delta-self.w))*self.kineticEnergy**((self.delta-3)/(2*(self.delta-self.w)))*(2e33*self.ejectaMass)**(-(self.delta-5)/(2*(self.delta-self.w)))*t**((self.delta-3)/(self.delta-self.w))
    
    def csmDensityProfile(self, t):        
        if self.shockRadius(t)>self.Rw and self.Dstar>1.34e-4:
            self.Dstar=1.34e-4
            #print("yeah")
            return self.D*self.shockRadius(t)**-self.w
        else:
            return self.D*self.shockRadius(t)**-self.w

    def shockVelocity(self, t):
        return self.X*self.D**(-1/(self.delta-self.w))*self.kineticEnergy**((self.delta-3)/(2*(self.delta-self.w)))*(2e33*self.ejectaMass)**(-(self.delta-5)/(2*(self.delta-self.w)))*((self.delta-3)/(self.delta-self.w))*t**(-1 +(self.delta-3)/(self.delta-self.w))
        
    def kineticLuminosity(self, t):
        return 2*mpm.pi*self.D*self.shockVelocity(t)**3

    def CRLuminosity(self, t):
        return self.epsCR*self.kineticLuminosity(t)
    
    @property
    def onset(self):
        once=mpm.re(mpm.findroot(lambda t: self.Rstar - self.shockRadius(t), 7e4,epsabs=1e-3))
        tbo=6e3*self.Dstar/0.01
        return np.maximum(float(once+1),tbo)

    def Hdens(self, T):
        return  self.csmDensityProfile(T)/1.67353284e-24

    
    def optical_depth(self,t,kpp=0.5,crossection=3e-26,c=3e10):
        fpp = kpp*crossection*self.Hdens(t)*self.shockRadius(t)*c/self.shockVelocity(t)
        return fpp
    
    def Emp(self,t):
        Emp = self.epsB**0.5*self.Dstar**(0.5)*(self.shockVelocity(t))**2 *12*1.6*6.24*1e-2*1e-12 #in TeV (-Emp**(2 - self.s)*mpm.expint(-1 + self.s, 1)+ mpm.expint(-1 + self.s, 1 / Emp)) 
        return Emp
 
    def Ucr(self,t):        
        initial_guess = 100.0  # need to adjust the initial guess based on the SN
        Ucr = lambda t: self.epsCR*self.csmDensityProfile(t)/(2)*self.shockVelocity(t)**2 #in erg cm-3
        A = mpm.findroot(lambda A: mpm.quad(lambda E: A*E**-1*mpm.exp(-E/self.Emp(t)),[2.797e-4,self.Emp(t)])-Ucr(t)/(1.60217663) , initial_guess) #in TeV cm^-3
        return float(A)

    def shockedVolume(self,t):
        V= 4/3*mpm.pi*self.shockRadius(t)**3*(1.208**3-0.978**3)
        return V

class Spectrum:
    def __init__(self, s, A,Emp, b=1): #0.07229626908
        self.Emp = Emp #in TeV
        self.A = A
        self.s = s
        self.b = b
    

    def CR(self,Ep):
        espectra = self.A * Ep**-self.s * mpm.exp(-(Ep/self.Emp)**self.b)
        return espectra

    def muonic_nu(self,Ep,x):
        L= mpm.log(Ep) #Ep in TeV
        B = 1.75 + 0.204*L + 0.010*L**2
        y = x/0.427
        beta_= 1/(1.67 + 0.111*L + 0.0038*L**2)
        k_ = 1.07 - 0.086*L + 0.002*L**2
        F1 = B*mpm.log(y)/y*((1-y**beta_)/(1+k_*y**beta_*(1-y**beta_)))**4*(1/mpm.log(y) - ((4*beta_*y**beta_)/(1-y**beta_)) - ((4*k_*beta_*y**beta_*(1-2*y**beta_))/(1+k_*y**beta_*(1-y**beta_))))
        Fvu = F1
        return Fvu
    
    def electric_nu(self,Ep,x):
        L= mpm.log(Ep) #Ep in TeV
        Be = 1/(69.5 + 2.65*L + 0.3*L**2)
        beta_e = 1/((0.201+0.062*L+0.00042*L**2)**(1/2))**(1/2)
        k_e = (0.279 +0.141*L+0.0172*L**2)/(0.3+(2.3+L)**2)
        F2 = Be*(1+k_e*mpm.log(x)**2)**3/(x*(1+0.3/(x**beta_e)))*(-mpm.log(x))**5
        Fve = F2
        return Fve

    def crossection(self,Ep):
        ott = (30.7-0.96*mpm.log(Ep/2.797e-4)+0.18*mpm.log(Ep/2.797e-4)**2)*(1-(2.797e-4/Ep)**1.9)**3*1e-27 #in cm2
        return ott
    

    def Rate_munu(self,Evu):
        #in TeV np.random.uniform(1e-9,0.427)
        g = lambda x: self.muonic_nu(Evu/x,x)*self.CR(Evu/x)*self.crossection(Evu/x)/x
        f = lambda x: self.electric_nu(Evu/x,x)*self.CR(Evu/x)*self.crossection(Evu/x)/x
        rate = 3e10*(mpm.re(mpm.quad(g,[1e-3, 0.4269]))+mpm.re(mpm.quad(f,[1e-3, 1])))
        return rate

    def Rate_elenu(self, Evu):
        #in TeV
        g = lambda x: self.electric_nu(Evu/x,x)*self.CR(Evu/x)*self.crossection(Evu/x)/x
        rate = 3e10*mpm.re(mpm.quad(g,[1e-3, 1]))
        return rate
    
    def total_rate(self, Evu):
        total = self.Rate_elenu(Evu)+self.Rate_munu(Evu)
        return total
    
    def proto(self,Evu,x):
        #g = lambda x: self.muonic_nu(Evu/x,x)*self.CR(Evu/x)*self.crossection(Evu/x)/x
        f = self.electric_nu(Evu/x,x)*self.CR(Evu/x)*self.crossection(Evu/x)/x
        #par = 3e10*(mpm.re(mpm.quad(g,[0.03, 0.05])))
        
        return mpm.re(f)
    
    def partial(self,Evu):
        g = lambda x: self.muonic_nu(Evu/x,x)*self.CR(Evu/x)*self.crossection(Evu/x)/x
        f = lambda x: self.electric_nu(Evu/x,x)*self.CR(Evu/x)*self.crossection(Evu/x)/x
        partial = 3e10*(mpm.re(mpm.quad(g,[0.03, 0.05]))/fnumu+2*mpm.re(mpm.quad(f,[0.03, 0.05]))/fnue)
        return partial

def lightcurve(t,SN_params):
    lumie = SN_params.epsCR*np.minimum(1,SN_params.optical_depth(t))*SN_params.kineticLuminosity(t)/1e2
    return lumie*1.60218 #inr erg-1 s-1

def flux(SN,dd,t):
    d=dd*3.08567758e21
    A=SN.Ucr(t)
    Emp=SN.Emp(t)
    ll=Spectrum(SN.s,A,Emp)
    return lambda E: 3*(ll.Rate_munu(E)+ll.Rate_elenu(E))/(4*mpm.pi*d**2)
    

def fluences(t,SN,Ev,dd=10):
    ll=Spectrum(SN.s,SN.Ucr(t),SN.Emp(t))
    d=dd*3.08567758e21
    #flue= 1/(4*d**2)*np.minimum(1,SN.optical_depth(t))*SN.epsCR/10*SN.D*SN.shockVelocity(t)**2*SN.shockRadius(t)*(Ev/(4))**(2-SN.s)
    flue= 1/(8*mpm.pi*d**2)*np.minimum(1,SN.optical_depth(t))*SN.kineticLuminosity(t)*SN.epsCR/10*(SN.Hdens(t)*SN.shockedVolume(t)*(ll.total_rate(Ev))*(Ev/4e-4)**2)#
    return flue

def partialfluences(t,SN,Ev,dd=10):
    ll=Spectrum(SN.s,SN.Ucr(t),SN.Emp(t))
    d=dd*3.08567758e21
    #flue= 1/(4*d**2)*np.minimum(1,SN.optical_depth(t))*SN.epsCR/10*SN.D*SN.shockVelocity(t)**2*SN.shockRadius(t)*(Ev/(4))**(2-SN.s)
    flue= 1/(8*mpm.pi*d**2)*np.minimum(1,SN.optical_depth(t))*SN.kineticLuminosity(t)*SN.epsCR/10*(SN.Hdens(t)*SN.shockedVolume(t)*(ll.partial(Ev))*(Ev/4e-4)**2)#
    return flue

def t_values(onset):
    return np.logspace(np.log10(float(onset)), 8, 1000)



def Nv(SN,distance,t):
    Nv=lambda E: 0.66*(E*1e3)**0.443*flux(SN,distance,t)(E)*10 #input TeV output GeV^-1 cm^-2 s^2^-1
    nev=mpm.quad(Nv,[1e-3,2e4])
    return nev
def Nh(SN,tar,tstop):
    ds=SN.Dstar
    tr = lambda t: SN.Hdens(t)*SN.shockedVolume(t)*np.minimum(1,SN.optical_depth(t))
    nh= [tr(ti) for ti in tar]
    limit_index = np.argmin(np.abs(tar - tstop))
    integral_value_trapz = np.trapz(nh[:limit_index], tar[:limit_index])
    SN.Dstar=ds
    return integral_value_trapz

def back():
    nv=lambda E:  0.66*(E*1e3)**0.443*(1.59*(E*1e3)**-3.7+12e-9*E**-2*mpm.exp(-E/1e4))*1e-4 #in TeV
    return mpm.quad(nv,[1e-3,2e4])

def get_sigback(SN, CR, distance, declination, tstar, tstop, cone=2, logE_threshold=0):

    tar=np.logspace(np.log10(tstar),np.log10(tstop),200)
    tarr=Nh(SN,tar,tstop)
    CR.set_flux(flux(SN,distance,tstar,tarr))
    x = CR.rates_table(livetime=(tstop-tstar), declination=declination, cone=cone, logE_threshold=logE_threshold).as_dataframe()
    bg = x.iloc[:, 3].tolist()[5]
    signal = x.iloc[:, 2].tolist()[5]

    return signal*2, bg*2


def p_val( s, mu ) :
    probability = 0
    for x in range(0,s):
        kans = (np.exp(-mu) * mu**x)/(math.factorial(x))
        probability += kans
    return 1 - probability

def n_needed_for_discovery(b, p_value_needed = 2.7e-3 ):
    for n in range( 100 ) :
        p = p_val( n, b )
        if p < p_value_needed : 
            return n
    return -1

def probability_of_discovery(s,b,p_value_needed = 2.7e-3) :
    n = n_needed_for_discovery( p_value_needed, b )
    p = p_val( n, mu = s+b) 
    return p

'''def compute_values(distance, dec,tsa,tso,ccc =CalcRate(),IIP=Parameters(kineticEnergy=1e51, ejectaMass=16, Dstar=1.34e-4, delta=12, epsCR=0.1, s=2, epsB=0.01, Rstar=6e13)):
    signal, background = get_sigback(IIP,ccc, distance, dec, tsa, tso)
    #pvalue = probability_of_discovery(signal, background)
    return signal, background, #pvalue
'''
