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
        f = pub.figure(cols=4)
        
        a = np.array(self.values_a)
        b = np.array(self.values_b)
        
        def labels(pylab):
            pylab.xlabel(self.label_a)
            pylab.ylabel(self.label_b)
            
        style = dict(markersize=0.3)
        with f.plot('plot1') as pylab:
            pylab.plot(a, b, '.', **style)
            labels(pylab)

        with f.plot('plot_equal') as pylab:
            pylab.plot(a, b, '.', **style)
            pylab.axis('equal')
            labels(pylab)

        with f.plot('plot1_axisb') as pylab:
            pylab.plot(a, b, '.', **style)
            x_axis_set(pylab, np.min(b), np.max(b))
            labels(pylab)

        with f.plot('plot1_axisa') as pylab:
            pylab.plot(a, b, '.', **style)
            y_axis_set(pylab, np.min(a), np.max(a))
            labels(pylab)

        style = dict(markersize=0.3)
        with f.plot('plot_nat') as pylab:
            plot_kde_gaussian(pylab, x=a, y=b,
                              xmin=a.min(), xmax=a.max(),
                              ymin=b.min(), ymax=b.max(), ncells=100)
            labels(pylab)

        with f.plot('plot_a_bounds') as pylab:
            plot_kde_gaussian(pylab, x=a, y=b,
                              xmin=a.min(), xmax=a.max(),
                              ymin=a.min(), ymax=a.max(), ncells=100)
            labels(pylab)

        with f.plot('plot_b_bounds') as pylab:
            plot_kde_gaussian(pylab, x=a, y=b,
                              xmin=b.min(), xmax=b.max(),
                              ymin=b.min(), ymax=b.max(), ncells=100)
            labels(pylab)
        
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
     
     
     
def plot_kde_gaussian(pylab, x, y, xmin, xmax, ymin, ymax, ncells):
    X, Y = np.mgrid[xmin:xmax:(1j * ncells), ymin:ymax:(1j * ncells)]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    from scipy import stats
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    pylab.imshow(np.rot90(Z), cmap=pylab.cm.gist_earth_r,
              extent=[xmin, xmax, ymin, ymax])
#     pylab.plot(x, y, 'k.', markersize=0.1)
    x_axis_set(pylab, xmin, xmax)
    y_axis_set(pylab, ymin, ymax)



