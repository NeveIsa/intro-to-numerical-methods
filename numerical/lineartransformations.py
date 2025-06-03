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
            self.theta = np.deg2rad(theta)
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
            self.theta = np.deg2rad(theta)
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
    def __init__(self,direction):
        
        vec = np.array(direction)
        self.normal = vec / np.linalg.norm(vec) 

    @property
    def matrix(self):
        return np.outer(self.normal,self.normal) 





