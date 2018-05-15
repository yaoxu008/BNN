# ----------------------------------------------------------------
# This code is modified based on VahidooX/DeepCCA/linear_cca.py
# LICENSE: https://github.com/VahidooX/DeepCCA/blob/master/LICENSE
# ----------------------------------------------------------------
import numpy


def cca(H1, H2, outdim):
    """
        An implementation of linear CCA
        # Arguments:
            H1 and H2: the matrices containing the data for view 1 and view 2. Each row is a sample.
            outdim_size: specifies the number of new features
        # Returns
            A and B: the linear transformation matrices
            mean1 and mean2: the means of data for both views
    """

    r1 = 1e-7
    r2 = 1e-7

    m = H1.shape[0]
    o = H1.shape[1]

    mean1 = numpy.mean(H1, axis=0)
    mean2 = numpy.mean(H2, axis=0)
    H1_ = H1 - numpy.tile(mean1, (m, 1))
    H2_ = H2 - numpy.tile(mean2, (m, 1))

    Sigma12 = (1.0 / (m - 1)) * numpy.dot(H1_.transpose(), H2_)
    Sigma11 = (1.0 / (m - 1)) * numpy.dot(H1_.transpose(), H1_) + r1 * numpy.identity(o)
    Sigma22 = (1.0 / (m - 1)) * numpy.dot(H2_.transpose(), H2_) + r2 * numpy.identity(o)

    [D1, V1] = numpy.linalg.eigh(Sigma11)
    [D2, V2] = numpy.linalg.eigh(Sigma22)
    Sigma11RootInv = numpy.dot(numpy.dot(V1, numpy.diag(D1 ** -0.5)), V1.transpose())
    Sigma22RootInv = numpy.dot(numpy.dot(V2, numpy.diag(D2 ** -0.5)), V2.transpose())

    Tval = numpy.dot(numpy.dot(Sigma11RootInv, Sigma12), Sigma22RootInv)

    [U, D, V] = numpy.linalg.svd(Tval)
    V = V.transpose()
    A = numpy.dot(Sigma11RootInv, U[:, 0:outdim])
    B = numpy.dot(Sigma22RootInv, V[:, 0:outdim])
    D = D[0:outdim]
    corr = numpy.sum(D)

    return A, B, corr
