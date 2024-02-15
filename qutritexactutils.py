import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from scipy.signal import find_peaks
pi = np.pi

import scipy
import datetime
import math as math
import cmath
sqrt = cmath.sqrt

class Qutrit_utils:

    def set_params(self, sys_params, w_params, fl_params, ec, ej, num_transmons, pump, g, P1, P2, transition, wpeaks, delta, readout):
        self.sys_params = sys_params
        self.wc = self.sys_params[0]
        self.w_params = w_params
        self.fl_params = fl_params
        self.ec = ec
        self.ej = ej
        self.num_transmons = num_transmons
        self.pump = pump
        self.g = g
        self.P1 = P1
        self.P2 = P2
        self.transition = transition
        self.wpeaks = wpeaks
        self.delta = delta 
        self.readout = readout
    def combin(self, a, b, size):
        lists = [([a] * size) for x in range(size-1)] 

        for i in range(size-1):
            lists[i][i+1] = b
        return lists
    
    def make_sops(self, a, b, c, size): #make all s operators, a is identity operator, 
                                     #b is destruction operator, size is #transmons
        s1 = self.combin(a, b, size)
        s2 = self.combin(a, c, size)

        both = []
        for i in range(self.num_transmons):
            both.append(s1[i])
            both.append(s2[i])
        for i in range(len(both)):
            both[i] = tensor(both[i])

        return both
    

    def makeops(self, kappa, P1, gamma, num_transmons):

        a = destroy(3)
        s1 = Qobj([[0,1,0],[0,0,0],[0,0,0]])
        s2 = Qobj([[0,0,0],[0,0,1],[0,0,0]])

        s = self.make_sops(qeye(3), s1, s2, num_transmons+1)
        for i in range(num_transmons):
            a = tensor(a, qeye(3))
        s_g = self.make_sops(qeye(3), s1, s2, num_transmons+1)
        # print('length s_g', len(s_g))

        c_ops = []
        c_ops.append(sqrt(kappa) * a) #loss from cavity
        c_ops.append(sqrt(P1) * a.dag()) #to populate initial state
        for i in range(len(s)):
            c_ops.append(sqrt(gamma)*s[i])
        # c_ops.append(sqrt(gamma)*s)
        return a, s, c_ops, s_g

    def makelists(self, w_params, fl_params): #makes flux bias and spectrum frequency
        wlist = np.linspace(*w_params)
        fbias = np.linspace(*fl_params)
        fluxfreq = self.we(fbias)

        return wlist, fbias, fluxfreq


    def hcavitytransmonandint(self, num_transmons, a, s, s_g, g, wc, fbias, wpeak, ec, d = [0,0]): #cavity transmon interaction
        h = a.dag() * a * (wc - wpeak)
        for i in range(num_transmons):
            # print('num_transmons', i)
            h += (fbias - wpeak - d[i]) * s[2*i].dag() * s[2*i]
            h += (2*fbias - 2*wpeak- d[i] - ec) * s[2*i + 1].dag() * s[2*i + 1]
            # h += g[i][0] * (a * s[2*i].dag() + s[2*i]*a.dag())
            # h += g[i][1] * (a * s[2*i+1].dag() + s[2*i+1]*a.dag())
            h += g[i][0] * (a * s[2*i].dag() + s[2*i]*a.dag())
            h += g[i][1] * (a * s[2*i+1].dag() + s[2*i+1]*a.dag())
        return h


    def hcoherent(self, pump, P2, s, a):
            h = 0
            if pump == True:
                # for i in range(len(s)):
                    # transmon1pump += + P2*(s[i].dag() + s[i])
                h += P2 * (s[0].dag() + s[0])
            return h

    def makeH(self, fbias, wpeak = 0, g = [[154]]):
        a, s, c_ops, s_g = self.makeops(self.sys_params[1], self.P1, self.sys_params[2], self.num_transmons)
        # wlist, flux, fluxfreq = self.makelists(low_w, high_w, num_w, low_fl, high_fl, num_fl, ec, ej)
        hcavtransandint = self.hcavitytransmonandint(self.num_transmons, a, s, s_g, g, self.wc[0], fbias, wpeak, self.ec)
        hcoherent = self.hcoherent(self.pump, self.P2, s, a)

        H = hcavtransandint + hcoherent
        return H, a, s, c_ops


    def runsim_1(self):
        wc, kappa, gamma = self.sys_params
        wlist, flux, fluxfreq = self.makelists(self.w_params, self.fl_params)

        spec = []
        if type(self.wpeaks) == int:
            for fbias in fluxfreq:
                H, a, s, c_ops = self.makeH(fbias, wpeak = 0, g = self.g) 
                spec.append(spectrum(H, wlist, c_ops, a.dag(), a))
                # spec.append(spectrum(H, wlist, c_ops, s[1].dag()*s[1].dag(), s[1]*s[1]))

        if type(self.wpeaks) == np.ndarray:
            for trans, fbias in enumerate(fluxfreq): 
                H, a, s, c_ops = self.makeH(fbias, self.wpeaks[trans, self.transition], g = self.g)     
                spec.append(spectrum(H, wlist, c_ops, a.dag(), a))

        spec = np.transpose(spec)
        return flux, wlist, spec, fluxfreq, H
    
    def find_peaks(self, wlist, spec, flux, plot, trans=None):
        if trans:= None:
            wpeaks = []
            half = int(np.floor(len(wlist)/2))

            for i in range(len(flux)):
                max1 = np.max(spec[:, i][: half])/1e-2
                max2 = np.max(spec[:, i][: half])/1e-2

                a1 = scipy.signal.find_peaks(spec[:, i][: half])
                a2 = scipy.signal.find_peaks(spec[:, i][half+1:-1])
                

                peak1 = a1[0][a1[1]['peak_heights'].argmax()]
                peak2 = a2[0][a2[1]['peak_heights'].argmax()] + half+1

                wpeaks.append([wlist[peak1], wlist[peak2]])

            wpeaks = np.array(wpeaks)
            plt.plot(flux, wpeaks[:,0])
            plt.plot(flux, wpeaks[:,1])
        else:
            wpeaks = []
            half = int(np.floor(len(wlist)/2))
            val = 1
            for i in range(len(flux)):
                try:
                    a1 = scipy.signal.find_peaks(spec[:, i][: half], height= 1e-9)
                    a2 = scipy.signal.find_peaks(spec[:, i][half+val:-1], height= 1e-9)


                    peak1 = a1[0][a1[1]['peak_heights'].argmax()]
                    peak2 = a2[0][a2[1]['peak_heights'].argmax()] + half+val

                    wpeaks.append([wlist[peak1], wlist[peak2]])
                except ValueError: i +=10

            wpeaks = np.array(wpeaks)
        if plot == True:
            plt.plot(flux, wpeaks[:,0], 'bo')
            plt.plot(flux, wpeaks[:,1], 'ro')


        return wpeaks
    
    def savepeaks(wpeaks, flux, wpeakshigh, wpeakslow, fname):
        f, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize = [4,8])

        wpeakslowadd = 2*wpeaks[:,0] + wpeakslow[:,1]
        ax.plot(flux, wpeakslowadd, label = 'negative transition')
        wpeakshighadd = wpeakshigh[:,0] + 2*wpeaks[:,1]
        ax.plot(flux, wpeakshighadd, label = 'positive transition')

        ax2.plot(flux, wpeaks, label = 'negative transition')
        ax2.plot(flux, wpeaks, label = 'positive transition')
        ax.plot(flux, [2*6940]*len(flux), 'k--', label = 'n * $\omega_c$')

        ax.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax.xaxis.tick_top()
        ax.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()
        ax2.plot(flux, [6940]*len(flux), 'k--', label = '$n *\omega_c$')

        ax.legend(loc = 'center right')

        wpeakssplit1 = wpeaks[:,1] - wpeaks[:,0]
        wpeakssplit2 = wpeakshighadd - wpeakslowadd


        plt.savefig(f, fname + 'combinedplot')
        np.savetxt(fname + 'peaks', np.transpose([wpeaks, wpeakssplit1,  wpeakslowadd, wpeakshighadd, wpeakssplit2, flux]), header = 'wpeaks, wpeakssplit1,  wpeakslowadd, wpeakshighadd, wpeakssplit2, flux', delimiter='%t')

        return wpeakssplit1, wpeakssplit2
    
    def makep_plot(self, flux, fluxfreq, wlist, spec, lines):

        fig, ax = plt.subplots(1, figsize = (7,7))
        im = ax.pcolor(flux,wlist,spec, shading = 'auto')
        fig.colorbar(im, ax=ax)
        ax.set_ylim(self.w_params[0], self.w_params[1])
        ax.set_ylabel('Frequency $ \\nu_{rf} $, (MHz)')
        ax.set_xlabel('Flux bias, $\Phi/\Phi_0$')
        if lines == True:
            ax.plot(flux, self.sys_params[0]*len(flux), 'w--')
            ax.plot(flux, fluxfreq, 'w--')
        return fig, ax
        # return
    
    def save_all(self, fname, fig, ax, spec, flux, wlist, fluxfreq, params, trans = None):
        a = np.array([wlist, flux, fluxfreq, spec])
        now = datetime.datetime.now().strftime('%Y_%m_%d_%H')
        if trans == 0:
            fname = fname + '_negtrans_' + now
        if trans == 1:
            fname = fname + '_postrans_' + now
        else:
            fname = fname + now


        # keys = params.keys()
        # vals = params.values()
        np.save(fname + '_data', a, allow_pickle=True)
        np.save(fname + '_params', params, allow_pickle=True)
        plt.savefig(fname + '_fig')
        # return print('saved')

    def run_and_get_allplots(self, params, fname):
        self.set_params(**params) ##sets the above parameters for the simulation
        flux, wlist, spec_single, fluxfreq, H = self.runsim_1()
        fig, ax = self.makep_plot(flux, fluxfreq, wlist, spec_single, False)
        wpeaks = self.find_peaks(wlist, spec_single, flux, plot = True) #find the transition peaks
        self.save_all(fname, fig, ax, flux, wlist, spec_single, fluxfreq, params)

        self.wpeaks = wpeaks
        self.w_params = [-600, 600, 1001]


        #negative transition
        self.w_params = [20, 6*self.g[0], 301]
        self.transition = 0
        self.pump = True

        flux, wlist, spec, fluxfreq, H = self.runsim_1()
        fig2, ax = self.makep_plot(flux, fluxfreq, wlist, spec, lines = False)
        wpeakslow = self.find_peaks(wlist, spec, flux, plot = True, trans=0) #find the transition peaks
        self.save_all(fname, fig2, ax, spec, flux, wlist, fluxfreq, params, trans=0)

        #positive transition
        self.transition = 1
        self.w_params = [-(6 * self.g[0]), -20, 301]
        flux, wlist, spec, fluxfreq, H = self.runsim_1()
        fig3, ax = self.makep_plot(flux, fluxfreq, wlist, spec, lines = False)
        wpeakshigh = self.find_peaks(wlist, spec, flux, plot = True) #find the transition peaks
        self.save_all(fname, fig3, ax, spec, flux, wlist, fluxfreq, params, trans=1)


        f, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize = [4,8])

        wpeakslowadd = 2*wpeaks[:,0] + wpeakslow[:,1]/2
        ax.plot(flux, wpeakslowadd, label = 'negative transition')
        wpeakshighadd = wpeakshigh[:,0]/2 + 2*wpeaks[:,1]
        ax.plot(flux, wpeakshighadd, label = 'positive transition')

        ax2.plot(flux, wpeaks, label = 'negative transition')
        ax2.plot(flux, wpeaks, label = 'positive transition')
        ax.plot(flux, [2*6940]*len(flux), 'k--', label = 'n * $\omega_c$')

        ax.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax.xaxis.tick_top()
        ax.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()
        ax2.plot(flux, [6940]*len(flux), 'k--', label = '$n *\omega_c$')

        ax.legend(loc = 'center right')

        wpeakssplit1 = wpeaks[:,1] - wpeaks[:,0]
        wpeakssplit2 = wpeakshighadd - wpeakslowadd


        plt.savefig(fname + 'combinedplot')
        np.savetxt(fname + 'peaks', np.transpose([wpeaks[:,0], wpeaks[:,1], wpeakssplit1,  wpeakslowadd, wpeakshighadd, wpeakssplit2, flux]), header = 'wpeaks0 \t wpeaks1 \t wpeakssplit1 \t wpeakslowadd \t wpeakshighadd \t wpeakssplit2 T5 flux', delimiter='\t')
        return wpeakssplit1, wpeakssplit2

    def we(self, flux): #turns flux bias values into frequencies we can then use in qutip spectrum function
        E_j = self.ej * np.abs(np.cos(pi*flux))
        w_e = np.sqrt(8*self.ec * E_j)
        return w_e


    
    def load_old(self, fname):
        b = np.load(fname + '_data.npy', allow_pickle=True)
        c = np.load(fname + '_params.npy', allow_pickle=True)
        return b, c, print('loaded')
    


    def cuberoot(self, v):
        if not isinstance(v, complex):
            v = complex(v, 0)
        return v ** (1.0 / 3.0)

    def A(self, g, f):
        A = -f**2 - 6 * g**2
        return A
    
    def B(self, g,f):
        B = sqrt(-4*f**4 * g**2 - 13*f**2 * g**4 - 32*g**6)
        return B
    
    def C(self, g,f):
        b = self.B(g,f)
        C = 2*f**3 + 3*sqrt(3) * b - 9*f*g**2
        return self.cuberoot(C)

    def lam3(self, g,f,w):
        a = self.A(g,f)
        b = self.B(g,f)
        c = self.C(g,f)

        term1 = (self.cuberoot(2)*(a))/(3*c)
        term2 = c/(3*self.cuberoot(2)) 
        term3 = (f + 6*w)/3
        return(term1 + term2 + term3)

    def lam4(self, g,f,w):
        a = self.A(g,f)
        b = self.B(g,f)
        c = self.C(g,f)

        term1 = (complex(1, sqrt(3)) * a)/(3 * self.cuberoot(2**2)*c)
        term2 = -(complex(1, -sqrt(3)) * c)/(6 * self.cuberoot(2))
        term3 = (f + 6*w)/3

        return term1 + term2 + term3

    def lam5(self, g,f,w):
        a = self.A(g,f)
        b = self.B(g,f)
        c = self.C(g,f)

        term1 = (complex(1, -sqrt(3)) * a)/(3 * 2**(2/3)*c)
        term2 = -(complex(1, sqrt(3)) * c)/(6 * self.cuberoot(2))
        term3 = (f + 6*w)/3

        return term1 + term2 + term3



    def eigens(self, g,f, w):
        a = self.A(g, f)
        b = self.B(g, f)
        c = self.C(g, f)
        eigens = []
        eigens.append(w - g)
        eigens.append(w + g)

        eigens.append(self.lam3(g,f,w))
        eigens.append(self.lam4(g,f,w))
        eigens.append(self.lam5(g,f,w))

        return eigens