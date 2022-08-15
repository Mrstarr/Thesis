from zlib import Z_BEST_COMPRESSION
import tensorflow as tf
import numpy as np
from gpflow.models.model import GPModel
#from gpflow.param import Param, DataHolder
import gpflow.mean_functions
from gpflow import likelihoods
from GPy import Model


class OSGPR_VFE1(GPModel):
    """
    Online Sparse Variational GP regression.
    
    Streaming Gaussian process approximations
    Thang D. Bui, Cuong V. Nguyen, Richard E. Turner
    NIPS 2017
    """

    def __init__(self, X, Y, kern, mu_old, Su_old, Kaa_old, Z_old, Z, mean_function=Zero()):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate gpflow objects
        mu_old, Su_old are mean and covariance of old q(u)
        Z_old is the old inducing inputs
        This method only works with a Gaussian likelihood.
        """

        """
        X = DataHolder(X, on_shape_change='pass')
        Y = DataHolder(Y, on_shape_change='pass')
        likelihood = likelihoods.Gaussian()
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.Z = Param(Z)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]

        # self.mu_old = DataHolder(mu_old, on_shape_change='pass')
        self.M_old = Z_old.shape[0]
        self.Su_old = DataHolder(Su_old, on_shape_change='pass')
        self.Kaa_old = DataHolder(Kaa_old, on_shape_change='pass')
        self.Z_old = DataHolder(Z_old, on_shape_change='pass')
        """


        likelihood = likelihoods.Gaussian()
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.Z = Z
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]

        # self.mu_old = DataHolder(mu_old, on_shape_change='pass')
        self.M_old = Z_old.shape[0]
        self.Su_old = Su_old
        self.Kaa_old = Kaa_old
        self.Z_old = Z_old

    def _build_common_terms(self):
        Mb = tf.shape(self.Z)[0]
        Ma = self.M_old
        # jitter = settings.numerics.jitter_level
        jitter = 1e-4
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        Saa = self.Su_old
        ma = self.mu_old

        # a is old inducing points, b is new
        # f is training points
        # s is test points
        Kbf = self.kern.K(self.Z, self.X)
        Kbb = self.kern.K(self.Z) + tf.eye(Mb, dtype=tf.dtypes.float32) * jitter
        Kba = self.kern.K(self.Z, self.Z_old)
        Kaa_cur = self.kern.K(self.Z_old) + tf.eye(Ma, dtype=tf.dtypes.float32) * jitter
        Kaa = self.Kaa_old + tf.eye(Ma, dtype=tf.dtypes.float32) * jitter

        err = self.Y - self.mean_function(self.X)

        Sainv_ma = tf.matrix_solve(Saa, ma)
        Sinv_y = self.Y / sigma2
        c1 = tf.matmul(Kbf, Sinv_y)
        c2 = tf.matmul(Kba, Sainv_ma)
        c = c1 + c2

        Lb = tf.cholesky(Kbb)
        Lbinv_c = tf.matrix_triangular_solve(Lb, c, lower=True)
        Lbinv_Kba = tf.matrix_triangular_solve(Lb, Kba, lower=True)
        Lbinv_Kbf = tf.matrix_triangular_solve(Lb, Kbf, lower=True) / sigma
        d1 = tf.matmul(Lbinv_Kbf, tf.transpose(Lbinv_Kbf))

        LSa = tf.cholesky(Saa)
        Kab_Lbinv = tf.transpose(Lbinv_Kba)
        LSainv_Kab_Lbinv = tf.matrix_triangular_solve(
            LSa, Kab_Lbinv, lower=True)
        d2 = tf.matmul(tf.transpose(LSainv_Kab_Lbinv), LSainv_Kab_Lbinv)

        La = tf.cholesky(Kaa)
        Lainv_Kab_Lbinv = tf.matrix_triangular_solve(
            La, Kab_Lbinv, lower=True)
        d3 = tf.matmul(tf.transpose(Lainv_Kab_Lbinv), Lainv_Kab_Lbinv)

        D = tf.eye(Mb, dtype=tf.dtypes.float32) + d1 + d2 - d3
        D = D + tf.eye(Mb, dtype=tf.dtypes.float32) * jitter
        LD = tf.cholesky(D)

        LDinv_Lbinv_c = tf.matrix_triangular_solve(LD, Lbinv_c, lower=True)

        return (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD,
                Lbinv_Kba, LDinv_Lbinv_c, err, d1)

    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. 
        """

        Mb = tf.shape(self.Z)[0]
        Ma = self.M_old
        jitter = settings.numerics.jitter_level
        # jitter = 1e-4
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        N = self.num_data

        Saa = self.Su_old
        ma = self.mu_old

        # a is old inducing points, b is new
        # f is training points
        Kfdiag = self.kern.Kdiag(self.X)
        (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD,
            Lbinv_Kba, LDinv_Lbinv_c, err, Qff) = self._build_common_terms()

        LSa = tf.cholesky(Saa)
        Lainv_ma = tf.matrix_triangular_solve(LSa, ma, lower=True)

        bound = 0
        # constant term
        bound = -0.5 * N * np.log(2 * np.pi)
        # quadratic term
        bound += -0.5 * tf.reduce_sum(tf.square(err)) / sigma2
        # bound += -0.5 * tf.reduce_sum(ma * Sainv_ma)
        bound += -0.5 * tf.reduce_sum(tf.square(Lainv_ma))
        bound += 0.5 * tf.reduce_sum(tf.square(LDinv_Lbinv_c))
        # log det term
        bound += -0.5 * N * tf.reduce_sum(tf.log(sigma2))
        bound += - tf.reduce_sum(tf.log(tf.diag_part(LD)))

        # delta 1: trace term
        bound += -0.5 * tf.reduce_sum(Kfdiag) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.diag_part(Qff))

        # delta 2: a and b difference
        bound += tf.reduce_sum(tf.log(tf.diag_part(La)))
        bound += - tf.reduce_sum(tf.log(tf.diag_part(LSa)))

        Kaadiff = Kaa_cur - tf.matmul(tf.transpose(Lbinv_Kba), Lbinv_Kba)
        Sainv_Kaadiff = tf.matrix_solve(Saa, Kaadiff)
        Kainv_Kaadiff = tf.matrix_solve(Kaa, Kaadiff)

        bound += -0.5 * tf.reduce_sum(
            tf.diag_part(Sainv_Kaadiff) - tf.diag_part(Kainv_Kaadiff))

        return bound

    def build_predict(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. 
        """

        # jitter = settings.numerics.jitter_level
        jitter = 1e-4

        # a is old inducing points, b is new
        # f is training points
        # s is test points
        Kbs = self.kern.K(self.Z, Xnew)
        (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD,
            Lbinv_Kba, LDinv_Lbinv_c, err, Qff) = self._build_common_terms()

        Lbinv_Kbs = tf.matrix_triangular_solve(Lb, Kbs, lower=True)
        LDinv_Lbinv_Kbs = tf.matrix_triangular_solve(LD, Lbinv_Kbs, lower=True)
        mean = tf.matmul(tf.transpose(LDinv_Lbinv_Kbs), LDinv_Lbinv_c)

        if full_cov:
            Kss = self.kern.K(Xnew) + jitter * tf.eye(tf.shape(Xnew)[0], dtype=tf.dtypes.float32)
            var1 = Kss
            var2 = - tf.matmul(tf.transpose(Lbinv_Kbs), Lbinv_Kbs)
            var3 = tf.matmul(tf.transpose(LDinv_Lbinv_Kbs), LDinv_Lbinv_Kbs)
            var = var1 + var2 + var3
        else:
            var1 = self.kern.Kdiag(Xnew)
            var2 = -tf.reduce_sum(tf.square(Lbinv_Kbs), 0)
            var3 = tf.reduce_sum(tf.square(LDinv_Lbinv_Kbs), 0)
            var = var1 + var2 + var3

        return mean + self.mean_function(Xnew), var

