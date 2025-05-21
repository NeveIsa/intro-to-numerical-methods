from abc import ABC, abstractmethod
import numpy as np

class LinearTransform(ABC):
    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """Return the transformation matrix."""
        pass

    def __call__(self, vec: np.ndarray) -> np.ndarray:
        """Apply the transformation to a 2D vector."""
        return self.matrix @ vec

class Rotation(LinearTransform):
    def __init__(self,theta,deg=True):
        if deg:
            self.theta = theta*np.pi/180
        else:
            self.theta = theta

    @property
    def matrix(self):
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        return np.array([[c , s], [-s, c]]).T

class Reflection(LinearTransform):
    def __init__(self, theta, deg=True):
        if deg:
            self.theta = theta*np.pi/180
        else:
            self.theta = theta
    @property
    def matrix(self): 
        c = np.cos(2*self.theta)
        s = np.sin(2*self.theta)
        return np.array([[c,s],[s,-c]]).T

class Stretching(LinearTransform):
    def __init__(self, xstretch, ystretch):
        self.a = xstretch
        self.b = ystretch

    @property
    def matrix(self):
        return np.array([[self.a,0],[0,self.b]])

class Projection(LinearTransform):
    def __init__(self,projx,projy):
        if projx > 0:
            projx = 1
        else:
            projx = 0

        if projy > 0:
            projy = 1
        else:
            projy = 0

        self.projx,self.projy = projx,projy

    @property
    def matrix(self):
        return np.array([[self.projx,0],[0, self.projy]])





