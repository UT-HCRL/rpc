from bisect import bisect
from enum import Enum

import numpy as np
from scipy.special import binom


class BezierParam(Enum):
    POS = 0
    VEL = 1
    ACC = 2


class BezierCurve:

    def __init__(self, points, a=0, b=1):

        assert b > a

        self.points = points
        self.h = points.shape[0] - 1
        self.d = points.shape[1]
        self.a = a
        self.b = b
        self.duration = b - a

    def __call__(self, t):

        c = np.array([self.berstein(t, n) for n in range(self.h + 1)])
        return c.T.dot(self.points)

    def berstein(self, t, n):

        c1 = binom(self.h, n)
        c2 = (t - self.a) / self.duration
        c3 = (self.b - t) / self.duration
        value = c1 * c2 ** n * c3 ** (self.h - n)

        return value

    def start_point(self):

        return self.points[0]

    def end_point(self):

        return self.points[-1]

    def derivative(self):

        points = (self.points[1:] - self.points[:-1]) * (self.h / self.duration)

        return BezierCurve(points, self.a, self.b)

    def l2_squared(self):

        A = np.zeros((self.h + 1, self.h + 1))
        for m in range(self.h + 1):
            for n in range(self.h + 1):
                A[m, n] = binom(self.h, m) * binom(self.h, n) / binom(2 * self.h, m + n)
        A *= self.duration / (2 * self.h + 1)
        A = np.kron(A, np.eye(self.d))

        p = self.points.flatten()

        return p.dot(A.dot(p))

    def plot2d(self, samples=51, **kwargs):

        import matplotlib.pyplot as plt

        options = {'c':'b'}
        options.update(kwargs)
        t = np.linspace(self.a, self.b, samples)
        plt.plot(*self(t).T, **options)

    def plot3d(self, ax, samples=51, **kwargs):

        options = {'c':'b'}
        options.update(kwargs)
        t = np.linspace(self.a, self.b, samples)
        ax.plot(*self(t).T, **options)

    def get_sample_points(self, samples=11):
        t = np.linspace(self.a, self.b, samples)
        points = self(t).astype(np.float32)
        return t, points

    def scatter2d(self, **kwargs):

        import matplotlib.pyplot as plt

        options = {'fc':'orange', 'ec':'k', 'zorder':3}
        options.update(kwargs)
        plt.scatter(*self.points.T, **options)

    def plot_2dpolygon(self, **kwargs):

        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from scipy.spatial import ConvexHull

        options = {'fc':'lightcoral'}
        options.update(kwargs)
        hull = ConvexHull(self.points)
        ordered_points = hull.points[hull.vertices]
        poly = Polygon(ordered_points, **options)
        plt.gca().add_patch(poly)


class CompositeBezierCurve:

    def __init__(self, beziers):

        for bez1, bez2 in zip(beziers[:-1], beziers[1:]):
            assert bez1.b == bez2.a
            assert bez1.d == bez2.d

        self.beziers = beziers
        self.N = len(self.beziers)
        self.d = beziers[0].d
        self.a = beziers[0].a
        self.b = beziers[-1].b
        self.duration = self.b - self.a
        self.transition_times = [self.a] + [bez.b for bez in beziers]

    def find_segment(self, t):

        return min(bisect(self.transition_times, t) - 1, self.N - 1)

    def __call__(self, t):

        i = self.find_segment(t)

        return self.beziers[i](t)

    def start_point(self):

        return self.beziers[0].start_point()

    def end_point(self):

        return self.beziers[-1].end_point()

    def derivative(self):

        return CompositeBezierCurve([b.derivative() for b in self.beziers])

    def bound_on_integral(self, f):

        return sum(bez.bound_on_integral(f) for bez in self.beziers)

    def plot2d(self, **kwargs):

        for bez in self.beziers:
            bez.plot2d(**kwargs)

    def plot3d(self, ax, **kwargs):

        for bez in self.beziers:
            bez.plot3d(ax, **kwargs)

    def scatter2d(self, **kwargs):

        for bez in self.beziers:
            bez.scatter2d(**kwargs)

    def plot_2dpolygon(self, **kwargs):

        for bez in self.beziers:
            bez.plot_2dpolygon(**kwargs)
