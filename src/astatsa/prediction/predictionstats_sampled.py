import numpy as np 
from contracts import contract
import warnings
from reprep.plot_utils.axes import x_axis_set, y_axis_set

__all__ = ['PredictionStatsSampled']


class PredictionStatsSampled(object):

    @contract(label_a='str', label_b='str', probability='>0,<=1')
    def __init__(self, probability, label_a='a', label_b='b'):
        self.probability = probability
        self.label_a = label_a
        self.label_b = label_b
        self.values_a = []
        self.values_b = []

    @contract(a='array,shape(x)', b='array,shape(x)', w='array(bool),shape(x)')
    def update(self, a, b, w):
        af = a.flatten()
        bf = b.flatten()
        wf = w.flatten()
        n = af.size
        accept = np.random.rand(n) < self.probability
        accept = np.logical_and(accept, wf)
        
        self.values_a.extend(af[accept])
        self.values_b.extend(bf[accept])
        
    def publish(self, pub):
        f = pub.figure()
        
        a = np.array(self.values_a)
        b = np.array(self.values_b)
        
        style = dict(markersize=0.3)
        with f.plot('plot1') as pylab:
            pylab.plot(a, b, '.', **style)
            pylab.xlabel(self.label_a)
            pylab.ylabel(self.label_b)

        with f.plot('plot_equal') as pylab:
            pylab.plot(a, b, '.', **style)
            pylab.axis('equal')
            pylab.xlabel(self.label_a)
            pylab.ylabel(self.label_b)

        with f.plot('plot1_axisb') as pylab:
            pylab.plot(a, b, '.', **style)
            x_axis_set(pylab, np.min(b), np.max(b))
            pylab.xlabel(self.label_a)
            pylab.ylabel(self.label_b)

        with f.plot('plot1_axisa') as pylab:
            pylab.plot(a, b, '.', **style)
            y_axis_set(pylab, np.min(a), np.max(a))
            pylab.xlabel(self.label_a)
            pylab.ylabel(self.label_b)
        
        if False:
            n = 100
            a1, a2 = np.percentile(a, [1, 99]) 
            b1, b2 = np.percentile(b, [1, 99])
            a_bins = np.linspace(a1, a2, n)
            b_bins = np.linspace(b1, b2, n)
            H, xedges, yedges = np.histogram2d(a, b, bins=[a_bins, b_bins])
            
            with f.plot('hist') as pylab:
                extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
                pylab.imshow(H, extent=extent, interpolation='nearest')
    
            with f.plot('hist_scaled') as pylab:
                warnings.warn('remove dependency')
                from boot_agents.utils.nonparametric import scale_score
                extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
                S = scale_score(H)
                pylab.imshow(S, extent=extent, interpolation='nearest')
    
            
# 
# class PredictionStatsWeighted(object):
# 
#     @contract(label_a='str', label_b='str')
#     def __init__(self, label_a='a', label_b='b'):
#         self.label_a = label_a
#         self.label_b = label_b
#         self.Ea = MeanVariance()
#         self.Eb = MeanVariance()
#         self.Edadb = Expectation()
#         self.R = None
#         self.R_needs_update = True
#         self.num_samples = 0
#         self.last_a = None
#         self.last_b = None
# 
#     @contract(a='array,shape(x)', b='array,shape(x)', w='array,shape(x)')
#     def update(self, a, b, w):
#         self.Ea.update(a, dt)
#         self.Eb.update(b, dt)
#         da = a - self.Ea.get_mean()
#         db = b - self.Eb.get_mean()
#         self.Edadb.update(da * db, dt)
#         self.num_samples += dt
# 
#         self.R_needs_update = True
#         self.last_a = a
#         self.last_b = b
# 
#     def get_correlation(self):
#         ''' Returns the correlation between the two streams. '''
#         if self.R_needs_update:
#             std_a = self.Ea.get_std_dev()
#             std_b = self.Eb.get_std_dev()
#             p = std_a * std_b
#             zeros = p == 0
#             p[zeros] = 1
#             R = self.Edadb() / p
#             R[zeros] = np.NAN
#             self.R = R
#         self.R_needs_update = False
#         return self.R


    
