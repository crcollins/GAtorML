import os

import numpy
import scipy.stats
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


from molml.features import EncodedBond, EncodedAngle
from molml.features import BagOfBonds
from molml.utils import get_connections


def read_cry(path):
    unit = []
    coords = []
    elements = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                unit.append([float(x) for x in parts])
            if len(parts) == 1:
                energy = parts[0]
            if len(parts) == 4:
                elements.append(parts[0])
                coords.append([float(x) for x in parts[1:]])
    return numpy.array(unit), numpy.array(coords), elements, energy


def unit_vector(vector):
    return vector / numpy.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return numpy.arccos(numpy.clip(numpy.dot(v1_u, v2_u), -1.0, 1.0))


def get_unit_values(unit):
    vol = numpy.linalg.det(unit)
    alpha = angle_between(unit[0], unit[1])
    beta = angle_between(unit[0], unit[2])
    gamma = angle_between(unit[1], unit[2])
    return vol, alpha, beta, gamma

def encode(values, start, end, slope, segments=100):
    theta = numpy.linspace(start, end, segments)
    diff = values - theta[:, None]
    smooth = scipy.stats.norm.pdf
    return smooth(slope * diff).T


def gauss_kernel(x, y, gamma=1e-5):
    dists = cdist(x, y, 'sqeuclidean')
    dists *=-gamma
    numpy.exp(dists, dists)
    return dists


def get_diff(X, y):
    idxs1, idxs2 = numpy.tril_indices(len(y), -1)
    vals, vecs = numpy.linalg.eig(X.T.dot(X))
    proj = X.dot(vecs[:, :2]).real


def draw_cell(elements, coords, unit, connectivity=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:,0], coords[:,1], coords[:,2])
    if connectivity:
        connectivity = get_connections(elements, coords)
        for key, values in connectivity.items():
            x0, y0, z0 = coords[key]
            for val in values:
                x1, y1, z1 = coords[val]
                ax.plot([x0, x1], [y0, y1], [z0, z1], 'k-')

    plt.show()


def main(X, y):
    X = numpy.hstack([X, numpy.ones((len(X), 1))])
    for frac in [.8]:
        print
        print frac
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-frac)

        clf = Ridge(alpha=1e-1)
        clf.fit(X_train, y_train - y_train.mean())
        print y_train.std(), y_test.std()
        print numpy.abs(clf.predict(X_train) + y_train.mean() - y_train).mean()
        print numpy.abs(clf.predict(X_test) + y_train.mean() - y_test).mean()


def get_outers(X, y):
    idxs = numpy.tril_indices(X.shape[0], -1)
    X_outer = (X - X[:, None])[idxs]
    y_outer = (y > y[:, None])[idxs]
    return X_outer, y_outer



def corrupt_ranking(y, fracs=None):
    idxs = numpy.tril_indices(y.shape[0], -1)
    size = len(idxs[0])
    y_outer = (y > y[:, None])[idxs]
    if fracs is None:
        fracs = [0., 0.01, 0.05, 0.1, 0.2, 0.3]
    for frac in fracs:
        corr_idxs = numpy.random.choice(size, int(size * frac), replace=False)
        y_corr = y_outer.copy()
        y_corr[corr_idxs] ^= y_corr[corr_idxs]
        corr = numpy.zeros((y.shape[0], y.shape[0]))
        corr[idxs] = y_corr
        corr.T[idxs] = numpy.logical_not(y_corr)
        ordering = numpy.argsort(corr.sum(0))
        plt.plot(y[ordering])
    plt.show()


paths = ["data/%s/"%x+sorted(os.listdir("data/"+x))[-1] for x in os.listdir("data")]
paths_start = ["data/%s/"%x+sorted(os.listdir("data/"+x))[0] for x in os.listdir("data")]
data_start = [read_cry(x) for x in paths_start]
data = [read_cry(x) for x in paths]
data_start = data

in_data = [(x[2], x[1]) for x in data_start]
units = [x[0] for x in data_start]
y = numpy.array([float(x[-1]) for x in data])




print "OTHER"
other = numpy.array([get_unit_values(unit) for unit in units])
main(other, y)



print "-"*50
print "Encoded Other"
vol = encode(other[:, 0], 900, 1250, 0.2)
alpha = encode(other[:, 1], 0.1, 3., 20)
beta = encode(other[:, 2], 0.1, 3., 20)
gamma = encode(other[:, 3], 0.1, 3., 20)
X = numpy.hstack([vol, alpha, beta, gamma])
main(X, y)
print "-"*50


print "="*50
print "="*50
X = numpy.hstack([alpha, beta, gamma])
main(X, y)
print "+"*50
X = numpy.hstack([vol])
main(X, y)
print "="*50
print "="*50



print "BOND"
trans = EncodedBond()
enc_bond = trans.fit_transform(in_data)
print "ANGLE"
trans2 = EncodedAngle(n_jobs=4)
enc_angle = trans2.fit_transform(in_data)
X = numpy.hstack([enc_bond, enc_angle])
main(X, y)
print "-"*50



print "Normal and other"
X = numpy.hstack([enc_bond, enc_angle, other])
main(X, y)
print "-"*50


print "Normal and encoded other"
X = numpy.hstack([enc_bond, enc_angle, vol, alpha, beta, gamma])
main(X, y)

def pca_plot(X, y):
    X_outer, y_outer = get_outers(X, y)
    pca = PCA(n_components=2)
    X_new = pca.fit_transform(X_outer)
    pos = numpy.where(y_outer)
    neg = numpy.where(numpy.logical_not(y_outer))
    comp0 = X_new[:, 0]
    comp1 = X_new[:, 1]
    plt.plot(comp0[pos], comp1[pos], '.', alpha=.2)
    plt.plot(comp0[neg], comp1[neg], '.', alpha=.2)
    plt.show()


