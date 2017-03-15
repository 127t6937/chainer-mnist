import chainer
import chainer.functions as F
import chainer.links as L


class MLP1(chainer.Chain):

    def __init__(self, n_units, n_out=10, activation=F.relu):
        super(MLP1, self).__init__(
            l1=L.Linear(None, n_out)
        )

    def __call__(self, x):
        return self.l1(x)


class MLP2(chainer.Chain):

    def __init__(self, n_units, n_out=10, activation=F.relu):
        super(MLP2, self).__init__(
            l1=L.Linear(None, n_units),
            l2=L.Linear(None, n_out)
        )

    def __call__(self, x):
        h1 = activation(self.l1(x))
        return self.l2(h1)


class MLP3(chainer.Chain):

    def __init__(self, n_units, n_out=10, activation=F.relu):
        super(MLP3, self).__init__(
            l1=L.Linear(None, n_units),
            l2=L.Linear(None, n_units),
            l3=L.Linear(None, n_out),
        )

    def __call__(self, x):
        h1 = activation(self.l1(x))
        h2 = activation(self.l2(h1))
        return self.l3(h2)


class MLP4(chainer.Chain):

    def __init__(self, n_units, n_out=10, activation=F.relu):
        super(MLP4, self).__init__(
            l1=L.Linear(None, n_units),
            l2=L.Linear(None, n_units),
            l3=L.Linear(None, n_units),
            l4=L.Linear(None, n_out)
        )

    def __call__(self, x):
        h = activation(self.l1(x))
        h = activation(self.l2(h))
        h = activation(self.l3(h))
        return self.l4(h)


class MLP5(chainer.Chain):

    def __init__(self, n_units, n_out=10, activation=F.relu):
        super(MLP5, self).__init__(
            l1=L.Linear(None, n_units),
            l2=L.Linear(None, n_units),
            l3=L.Linear(None, n_units),
            l4=L.Linear(None, n_units),
            l5=L.Linear(None, n_out)
        )

    def __call__(self, x):
        h = activation(self.l1(x))
        h = activation(self.l2(h))
        h = activation(self.l3(h))
        h = activation(self.l4(h))
        return self.l5(h)
