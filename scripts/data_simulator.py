import numpy as np
import scipy as sp


class SimulateBiGMM:
    def __init__(
        self,
        n_kernels,
        means_loc,
        means_scale,
        covariance_x_loc,
        covariance_x_scale,
        covariance_y_loc,
        covariance_y_scale,
    ):

        """

		define a bi-variate GMM, details of model explained in the blog post
		math reference: https://bengio.abracadoudou.com/cv/publications/pdf/rr02-12.pdf
        code referrence: https://github.com/freelunchtheorem/Conditional_Density_Estimation/blob/master/cde/density_simulation/GMM.py
		p(x,y) = sum(w_k N() N())
		p(y/x) = sum(W_k(x) N())
		p(x) = N()
		p(y) = N()

        means : (nb_kernels, 2)
        covariances : n_kernels, 2, 2)
        weights : (n_kernels)

		"""

        ndim = 2
        self.n_kernels = n_kernels

        # Parameter for distribution from where means of X and Y are sampled
        self.means = np.random.normal(
            loc=np.zeros([ndim]) + means_loc, scale=means_scale, size=[n_kernels, ndim]
        )
        self.means_x = self.means[:, :1]
        self.means_y = self.means[:, 1:]

        # Generate Covariance Matrix
        self.covariances = np.zeros(shape=(n_kernels, ndim, ndim))

        self.covariances_x = np.abs(
            np.random.normal(
                loc=covariance_x_loc, scale=covariance_x_scale, size=(n_kernels, 1, 1)
            )
        )
        self.covariances_y = np.abs(
            np.random.normal(
                loc=covariance_y_loc, scale=covariance_y_scale, size=(n_kernels, 1, 1)
            )
        )

        self.covariances[:, :1, :1] = self.covariances_x
        self.covariances[:, 1:, 1:] = self.covariances_y

        # Generate Random Weights
        self.weights = np.random.uniform(0.0, 1.0, size=[n_kernels])
        self.weights = self.weights / np.sum(self.weights)

        # density_xy is p(x,y), density_x is p(x) and density_y is p(y)
        self.density_xy, self.density_x, self.density_y = [], [], []

        for i in range(n_kernels):
            self.density_xy.append(
                sp.stats.multivariate_normal(
                    mean=self.means[i,], cov=self.covariances[i]
                )
            )
            self.density_x.append(
                sp.stats.multivariate_normal(
                    mean=self.means_x[i,], cov=self.covariances_x[i]
                )
            )
            self.density_y.append(
                sp.stats.multivariate_normal(
                    mean=self.means_y[i,], cov=self.covariances_y[i]
                )
            )

    def simulate_xy(self, n_samples=1000):

        """
		   Draws samples from the unconditional distribution p(x,y)
		    Args:
		      n_samples: (int) number of samples 
		    Returns:
		      (X,Y) : ((n_samples,1),(n_samples,1))
		    
		"""

        w = np.random.multinomial(n_samples, self.weights)

        samples = np.vstack(
            [normal.rvs(size=n) for normal, n in zip(self.density_xy, w)]
        )
        np.random.shuffle(samples)

        x_samples = samples[:, :1]
        y_samples = samples[:, 1:]

        return x_samples, y_samples

    def simulate_conditional(self, X, n_samples=1000):

        """ Draws random samples from the conditional distribution p(y|x) 
		Args:
		  X: float x to be conditioned on when drawing a sample from y ~ p(y|x)
		Returns:
		  Conditional random samples y drawn from p(y|x) -(n_samples, 1)
		"""

        w_norm = self.normalize_weights(X)[0]

        n_samples_comp = np.random.multinomial(n_samples, w_norm)

        y_samples_ = []
        for normal, n in zip(self.density_y, n_samples_comp):
            if n != 0:
                y_samples_.append(normal.rvs(size=n))
        y_samples = np.hstack(y_samples_)

        np.random.shuffle(y_samples)

        return X, y_samples

    def compute_cdf(self, X, ys):
        """
		 compute p(y/x=X) ; X is a given float
         ys: points on which to compute p(y/x), (n, 1)
		"""

        P_y = np.stack(
            [self.density_y[i].pdf(ys) for i in range(self.n_kernels)], axis=1
        )
        W_x = self.normalize_weights(X)

        cond_prob = np.sum(np.multiply(W_x, P_y), axis=1)

        return cond_prob

    def compute_pdf(self, xmin, xmax, ymin, ymax):

        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()]).transpose()

        # Compute p(x,y)

        P_x = np.stack(
            [self.density_x[i].pdf(positions[:, 0]) for i in range(self.n_kernels)],
            axis=1,
        )  
        P_y = np.stack(
            [self.density_y[i].pdf(positions[:, 1]) for i in range(self.n_kernels)],
            axis=1,
        )  
        W = self.weights

        prob = np.sum(np.multiply(W, np.multiply(P_y, P_x)), axis=1)
        pp = np.reshape(prob, xx.shape)

        return xx, yy, pp

    def normalize_weights(self, X):
        """
        X: float
        """
        w_p = np.stack(
            [
                np.array([self.weights[i] * self.density_x[i].pdf(X)])
                for i in range(self.n_kernels)
            ],
            axis=1,
        )

        normalizing_term = np.sum(w_p, axis=1)

        result = w_p / normalizing_term[:, None]

        return result
