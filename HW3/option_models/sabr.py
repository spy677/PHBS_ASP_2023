# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import pyfeng as pf
import abc

class ModelABC(abc.ABC):
    beta = 1   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr = None, None

    ### Numerical Parameters
    dt = 0.1
    n_path = 10000

    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0.0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.beta = beta
        self.intr = intr

    def base_model(self, sigma=None):
        if sigma is None:
            sigma = self.sigma

        if self.beta == 0:
            return pf.Norm(sigma, intr=self.intr)
        elif self.beta == 1:
            return pf.Bsm(sigma, intr=self.intr)
        else:
            raise ValueError(f'0<beta<1 not supported')

    def vol_smile(self, strike, spot, texp=1.0):
        ''''
        From the price from self.price() compute the implied vol
        Use self.bsm_model.impvol() method
        '''
        price = self.price(strike, spot, texp, cp=1)
        iv = self.base_model().impvol(price, strike, spot, texp, cp=1)
        return iv

    @abc.abstractmethod
    def price(self, strike, spot, texp=1.0, cp=1):
        """
        Vanilla option price

        Args:
            strike:
            spot:
            texp:
            cp:

        Returns:

        """
        return NotImplementedError

    def sigma_path(self, texp):
        """
        Path of sigma_t over the time discretization

        Args:
            texp:

        Returns:

        """
        n_dt = int(np.ceil(texp / self.dt))
        tobs = np.arange(1, n_dt + 1) / n_dt * texp
        dt = texp / n_dt
        assert texp == tobs[-1]
        Z1=np.random.standard_normal((n_dt, self.n_path))
        Z_t = np.cumsum(Z1* np.sqrt(dt), axis=0)
        sigma_t = np.exp(self.vov * (Z_t - self.vov/2 * tobs[:, None]))
        sigma_t = self.sigma*np.insert(sigma_t, 0, np.ones(sigma_t.shape[1]), axis=0)

        return sigma_t, Z1

    def intvar_normalized(self, sigma_path):
        """
        Normalized integraged variance I_t = \int_0^T sigma_t dt / (sigma_0^2 T)

        Args:
            sigma_path: sigma path

        Returns:

        """
        weight = np.ones(sigma_path.shape[0])
        weight[[0, -1]] = 0.5
        weight /= weight.sum()
        intvar = np.sum(weight[:, None] * np.square(sigma_path), axis=0)/self.sigma**2
        return intvar

class ModelBsmMC(ModelABC):
    """
    MC for Bsm SABR (beta = 1)
    """

    beta = 1.0   # fixed (not used)

    def price(self, strike, spot, texp=1.0, cp=1):
        '''
        Your MC routine goes here.
        (1) Generate the paths of sigma_t. 

        vol_path = self.sigma_path(texp)  # the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)

        (2) Simulate S_0, ...., S_T.

        Z = np.random.standard_normal()

        (3) Calculate option prices (vector) for all strikes
        '''
        #generate sigma path
        vol_path, z2 = self.sigma_path(texp)  
        vol_pathnot=vol_path[:-1, :]

        z1=np.random.standard_normal((10, 10000))
        wt=self.rho*z1+np.sqrt(1-self.rho**2) * np.random.standard_normal((vol_pathnot.shape[0], vol_pathnot.shape[1]))

        deltast=np.exp((wt*vol_pathnot)*np.sqrt(self.dt)-vol_pathnot**2*self.dt/2)
        logst=np.cumprod(np.insert(deltast, 0, np.ones(deltast.shape[1]), axis=0), axis=0)
        
        S_T = spot * logst[-1, :]
        df = np.exp(-self.intr * texp)
        p = df * np.mean(np.fmax(cp*(S_T - strike[:, None]), 0.0), axis=1)
        return p

class ModelNormMC(ModelBsmMC):
    """
    MC for Normal SABR (beta = 0)
    """

    beta = 0   # fixed (not used)

    def price(self, strike, spot, texp=1.0, cp=1):
        '''
        Your MC routine goes here.
        (1) Generate the paths of sigma_t. 

        vol_path = self.sigma_path(texp)  # the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)

        (2) Simulate S_0, ...., S_T.

        Z = np.random.standard_normal()

        (3) Calculate option prices (vector) for all strikes
        '''
        vol_path, z1 = self.sigma_path(texp)  
        sigma_t = vol_path[-1, :]
        vol_pathnot=vol_path[:-1, :]
        wt=self.rho*z1+np.sqrt(1-self.rho**2)*np.random.standard_normal((vol_pathnot.shape[0], vol_pathnot.shape[1]))
        stso=np.sum(wt*np.sqrt(self.dt)*vol_pathnot, axis=0)
        S_T = spot + stso
        df = np.exp(-self.intr * texp)
        p = df * np.mean(np.fmax(cp*(S_T - strike[:, None]), 0.0), axis=1)
        return p

class ModelBsmCondMC(ModelBsmMC):
    """
    Conditional MC for Bsm SABR (beta = 1)
    """

    def price(self, strike, spot, texp=1.0, cp=1):
        '''
        Your MC routine goes here.
        (1) Generate the paths of sigma_t and normalized integrated variance

        vol_path = self.sigma_path(texp)  # the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)
        I_t = self.intvar_normalized(vol_path) 

        (2) Calculate the equivalent spot and volatility of the BS model

        vol = 
        spot_equiv = 

        (3) Calculate option prices (vector) by averaging the BS prices

        m = self.base_model(vol)
        p = np.mean(m.price(strike[:, None], spot_equiv, texp, cp), axis=1)
        '''
        vol_path, z1= self.sigma_path(texp)  # the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)
        weightneg = np.ones(vol_path.shape[0])*(-1)
        weightnegpos = np.square(weightneg)
        weight01=(weightneg+weightnegpos)/2
        weight = weight01*4+(np.ones(vol_path.shape[0])-weight01)*2
        weight[[0, -1]] = 0.5
        weight /= weight.sum()
        I_t = np.sum(weight[:, None] * np.square(vol_path), axis=0)
        vol = np.sqrt((1-self.rho**2)*I_t)
        spot_equiv = spot * np.exp(self.rho/self.vov*(sigma_t-self.sigma)-self.rho**2*self.sigma**2*texp*I_t/2)

        m = self.base_model(vol)
        p = np.mean(m.price(strike[:, None], spot_equiv, texp, cp), axis=1)
        
        return p


class ModelNormCondMC(ModelNormMC):
    """
    Conditional MC for Bsm SABR (beta = 0)
    """

    def price(self, strike, spot, texp=1.0, cp=1):
        '''
        Your MC routine goes here.
        (1) Generate the paths of sigma_t and normalized integrated variance

        vol_path = self.sigma_path(texp)  # the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)
        I_t = self.intvar_normalized(vol_path) 

        (2) Calculate the equivalent spot and volatility of the Bachelier model

        vol = 
        spot_equiv = 

        (3) Calculate option prices (vector) by averaging the BS prices

        m = self.base_model(vol)
        p = np.mean(m.price(strike[:, None], spot_equiv, texp, cp), axis=1)
        '''

        vol_path, z1= self.sigma_path(texp)  # the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)
        I_t = self.intvar_normalized(vol_path) 
        vol = self.sigma*np.sqrt((1-self.rho**2)*I_t)
        spot_equiv = spot + self.rho/self.vov*(sigma_t-self.sigma)
        m = self.base_model(vol)
        p = np.mean(m.price(strike[:, None], spot_equiv, texp, cp), axis=1)
        
        return p