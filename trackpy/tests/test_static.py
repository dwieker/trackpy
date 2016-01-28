
from trackpy.static import pairCorrelation2D, pairCorrelation3D
import unittest
import pandas
import numpy as np
import matplotlib.pyplot as plt


class TestPairCorrelation(unittest.TestCase):


    def test_correlation2D_lattice(self):
        ### Lattice Test
        # With proper edge handling, g(r) of the particle at the center should be the same as g(r) for all particles.
        lattice = self._lattice2D()

        # Calculate g_r on the center particle only (index 210)
        edges, g_r_one = pairCorrelation2D(lattice, dr=.1, cutoff=8, p_indices=[210])
        g_r_one /= np.linalg.norm(g_r_one) #We care about the relative difference of g_r in this case, so let's normalize both.

        # Calculate g_r on all particles
        edges, g_r_all = pairCorrelation2D(lattice, dr=.1, cutoff=8)
        g_r_all /= np.linalg.norm(g_r_all)

        # Calculate g_r on all particles
        edges, g_r_no_edge = pairCorrelation2D(lattice, dr=.1, cutoff=8, handle_edge=False)
        g_r_no_edge /= np.linalg.norm(g_r_no_edge)

        # Assert the functions are essentially the same
        self.assertTrue(np.allclose(g_r_all, g_r_one, atol=.02))

        # Turning off edge handling should give incorrect result
        self.assertFalse(np.allclose(g_r_all, g_r_no_edge, atol=.02))

        """
        plt.plot(g_r_all)
        plt.plot(g_r_one)
        plt.plot(g_r_no_edge)
        plt.title("2D")
        plt.show()
        """

    def test_correlation2D_ring(self):
        # Ring test
        # Generate a series of concentric shells, each with the same number of particles.
        # The peaks in g(r) should decay as 1/r.
        ring = self._rings2D()

        edges, g_r = pairCorrelation2D(ring, dr=.1, cutoff=10, p_indices=[0], handle_edge=False)
        g_r /= np.linalg.norm(g_r)
        peaks = g_r[g_r > 0]

        self.assertTrue( len(peaks) == 9 )

        x = np.arange(1,10,1)
        r = peaks.max() * 1/x
        
        """
        plt.plot(edges[:-1], g_r)
        plt.plot(x, r)
        plt.show()
        """

        self.assertTrue( np.allclose(peaks, r, atol=.01) )


    def test_correlation3D_ring(self):
        # Ring test
        # Generate a series of concentric shells, each with the same number of particles.
        # The peaks in g(r) should decay as 1/r^2.
        ring = self._rings3D()
        
        edges, g_r = pairCorrelation3D(ring, dr=.1, cutoff=10, p_indices=[len(ring) - 1], 
                        boundary = (-10., 10., -10., 10., -10., 10.), handle_edge=False)

        g_r /= np.linalg.norm(g_r)
        peaks = g_r[g_r > 0]
        assert len(peaks) == 9
        x = np.arange(1,10,1)
        r = peaks.max() * 1/x**2


        self.assertTrue( np.allclose(peaks, r, atol=.02) )


    def test_correlation3D_lattice(self):
        ### Lattice Test
        # With proper edge handling, g(r) of the particle at the center should be the same as g(r) for all particles.
        lattice = self._lattice3D(n = 20)

        # Calculate g_r on the center particle only
        edges, g_r_one = pairCorrelation3D(lattice, dr=.1, cutoff=7, p_indices=[4649])
        g_r_one /= np.linalg.norm(g_r_one) #We care about the relative difference of g_r in this case, so let's normalize both.

        # Calculate g_r on all particles
        edges, g_r_all = pairCorrelation3D(lattice, dr=.1, cutoff=7)
        g_r_all /= np.linalg.norm(g_r_all)

        # Calculate g_r on all particles
        edges, g_r_no_edge = pairCorrelation3D(lattice, dr=.1, cutoff=7, handle_edge=False)
        g_r_no_edge /= np.linalg.norm(g_r_no_edge)

        # Assert the functions are essentially the same
        self.assertTrue(np.allclose(g_r_all, g_r_one, atol=.04))

        # Turning off edge handling should give incorrect result
        self.assertFalse(np.allclose(g_r_all, g_r_no_edge, atol=.04))

        plt.plot(g_r_all)
        plt.plot(g_r_one)
        plt.plot(g_r_no_edge)
        plt.show()


    def _lattice2D(self, n = 20):
        #Generates 2D lattice, spacing = 1
        x,y = [],[]
        epsilon = 0.0
        for i in range(n):
            for j in range(n):
                x.append(i)
                y.append(j)

        return pandas.DataFrame({'x':x, 'y':y})


    def _rings2D(self):
        #Generates concentric rings, with a particle at the center
        theta = np.linspace(0, 2*np.pi, 10)
        points = np.zeros((100,2))

        i = 0
        epsilon = .02
        for r in range(10):
            points[i:i+10, 0] = (r + epsilon)*np.cos(theta)
            points[i:i+10, 1] = (r + epsilon)*np.sin(theta)
            i += 10
        points[:10] = 0

        return pandas.DataFrame(points, columns = ['x', 'y'])


    def _lattice3D(self, n = 20):
        #Generates 3D lattice, spacing = 1
        x,y,z = [],[],[]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    x.append(i)
                    y.append(j)
                    z.append(k)

        return pandas.DataFrame({'x':x, 'y':y, 'z':z})


    def _rings3D(self):
        #Generates concentric spherical shells, with a particle at the center
        epsilon = .02
        r = np.arange(1, 10, 1) + epsilon

        refx, refy, refz = self._points_ring3D(r, 0, 500)
        print refx.shape
        df = pandas.DataFrame({'x': refx, 'y': refy, 'z': refz})
        
        # The last index is the center particle, which is used to calculate g_r 
        df = df.append(pandas.DataFrame({'x': [0.], 'y': [0.], 'z':[0.]}))
        return df


    def _points_ring3D(self, r_edges, dr, n):
        """Returns x, y, z array of points comprising shells extending from r to r_dr. 
           n determines the number of points in the shell. Rings are generated by 
           constructing a unit sphere and projecting every point onto a shell of thickness dr"""

        refx = np.empty((len(r_edges), n))
        refy = refx.copy(); refz = refx.copy()
        for i, r in enumerate(r_edges):
            ref = 2*np.random.random(size=(n, 3)) - 1
            ref /= np.linalg.norm(ref, axis=1).repeat(3).reshape((len(ref), 3))
            ref *= dr*np.random.random(size=(len(ref), 3))+ r
            refx[i] = ref[:,0]; refy[i] = ref[:,1]; refz[i] = ref[:,2]; 
        return refx.flatten(), refy.flatten(), refz.flatten()

if __name__ == '__main__':
    unittest.main()
   