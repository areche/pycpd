from builtins import super
import numpy as np
from .expectation_maximization_registration import expectation_maximization_registration

class scale_translate_registration(expectation_maximization_registration):
    def __init__(self, t=None, s=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.D != 2 and self.D != 3:
            message = 'Pan & Scale registration only supports 2D or 3D point clouds. Instead got {}.'.format(self.D)
            raise ValueError(message)
        if s == 0:
            raise ValueError('A zero scale factor is not supported.')
        self.t = np.atleast_2d(np.zeros((1, self.D))) if t is None else t
        self.s = 1 if s is None else s

    def update_transform(self):
        muX = np.divide(np.sum(np.dot(self.P, self.X), axis=0), self.Np)
        muY = np.divide(np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)

        self.XX = self.X - np.tile(muX, (self.N, 1))
        YY      = self.Y - np.tile(muY, (self.M, 1))

        self.A = np.dot(np.transpose(self.XX), np.transpose(self.P))
        self.A = np.dot(self.A, YY)

        U, _, V = np.linalg.svd(self.A, full_matrices=True)
        C = np.ones((self.D, ))
        C[self.D-1] = np.linalg.det(np.dot(U, V))

        self.YPY = np.dot(np.transpose(self.P1), np.sum(np.multiply(YY, YY), axis=1))
        self.s = np.trace(np.transpose(self.A)) / self.YPY
        self.t = np.transpose(muX) - self.s * np.transpose(muY)

    def transform_point_cloud(self, Y=None):
        if Y is None:
            self.TY = self.s * self.Y + self.t
            return
        else:
            return self.s * Y + self.t

    def update_variance(self):
        qprev = self.q

        trAR = np.trace(self.A)
        xPx = np.dot(np.transpose(self.Pt1), np.sum(np.multiply(self.XX, self.XX), axis =1))
        self.q = (xPx - 2 * self.s * trAR + self.s * self.s * self.YPY) / (2 * self.sigma2) + self.D * self.Np/2 * np.log(self.sigma2)
        self.err = np.abs(self.q - qprev)
        self.sigma2 = (xPx - self.s * trAR) / (self.Np * self.D)
        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def get_registration_parameters(self):
        return self.s, self.t
