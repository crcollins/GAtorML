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


def main_lr(X, y, weight=False):

    for frac in [.8]:
        print
        print frac
        X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y, test_size=1-frac)

        X_train, y_train = get_outers(X_train_, y_train_)
        X_test, y_test = get_outers(X_test_, y_test_)

        clf = LogisticRegression()
        clf = SVC()
        clf = RandomForestClassifier(max_depth=10, n_estimators=100)
        if weight:
            print "Weighted"
            weights = numpy.square((y_train - y_train.mean()) / y_train.std())
            clf.fit(X_train, y_train, sample_weight=weights)
        else:
            clf.fit(X_train, y_train)
        print classification_report(y_train, clf.predict(X_train))
        print classification_report(y_test, clf.predict(X_test))
        print confusion_matrix(y_train, clf.predict(X_train))
        print confusion_matrix(y_test, clf.predict(X_test))

        get_plot(y_train_, clf.predict(X_train))
        get_plot(y_test_, clf.predict(X_test))
        plt.show()


def get_plot(y, pair_preds):
    n = y.shape[0]
    idxs = numpy.tril_indices(n, -1)
    y_outer = numpy.zeros((n, n))
    y_outer[idxs] = pair_preds
    y_outer.T[idxs] = ~pair_preds
    sums = y_outer.sum(0)
    ordering = numpy.argsort(sums)
    plt.plot(y[numpy.argsort(y)])
    plt.plot(y[ordering])


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


if __name__ == "__main__":
    dirs = [os.path.join("data", x) for x in os.listdir("data")]

    paths_final = [os.path.join(x, sorted(os.listdir(x))[-1]) for x in dirs]
    paths_start = [os.path.join(x, sorted(os.listdir(x))[0]) for x in dirs]

    print "Load data"
    data_final = [read_cry(x) for x in paths_final]
    data_start = [read_cry(x) for x in paths_start]

    in_data_final = [(x[2], x[1]) for x in data_final]
    in_data_start = [(x[2], x[1]) for x in data_start]

    units_final = [x[0] for x in data_final]
    units_start = [x[0] for x in data_start]

    y_final = numpy.array([float(x[-1]) for x in data_final])
    y_start = numpy.array([float(x[-1]) for x in data_start])


    print "OTHER"
    other_final = numpy.array([get_unit_values(unit) for unit in units_final])
    other_start = numpy.array([get_unit_values(unit) for unit in units_start])

    print "Encoded Other"
    for other in (other_start, ):
        vol = encode(other[:, 0], 900, 1250, 0.2)
        alpha = encode(other[:, 1], 0.1, 3., 20)
        beta = encode(other[:, 2], 0.1, 3., 20)
        gamma = encode(other[:, 3], 0.1, 3., 20)

    for in_data in (in_data_start, ):
        print "EncodedBond All"
        trans = EncodedBond(n_jobs=4, max_depth=0, end=25)
        enc_bond_all = trans.fit_transform(in_data)
        print "EncodedBond Connected"
        trans1 = EncodedBond(n_jobs=4, max_depth=10, end=25)
        enc_bond_conn = trans1.fit_transform(in_data)
        enc_bond_inter = enc_bond_all - enc_bond_conn
        print "EncodedAngle"
        trans2 = EncodedAngle(n_jobs=4)
        enc_angle = trans2.fit_transform(in_data)

    groups = (
        ("Other", (other, )),
        ("Other Enc", (vol, alpha, beta, gamma)),
        ("Other Vol", (vol, )),
        ("Other Angle Enc", (alpha, beta, gamma)),
        ("Geom All", (enc_bond_all, )),
        ("Geom Intra", (enc_bond_conn, )),
        ("Geom Inter", (enc_bond_inter, )),
        ("Geom Both", (enc_bond_conn, enc_bond_inter)),
    )

    y = y_final
    for name, group in groups:
        print "="*50
        print name
        print "="*50
        X = numpy.hstack(group)
        main(X, y)
        for b in (False, True):
            main_lr(X, y, b)
