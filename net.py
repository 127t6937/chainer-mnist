import chainer
import chainer.functions as F
import chainer.links as L


class MLP1(chainer.Chain):

    def __init__(self, n_units, n_out=10, activation=F.relu):
        self.activation = activation
        super(MLP1, self).__init__(
            l1=L.Linear(None, n_out)
        )

    def __call__(self, x):
        return self.l1(x)


class MLP2(chainer.Chain):

    def __init__(self, n_units, n_out=10, activation=F.relu):
        self.activation = activation
        super(MLP2, self).__init__(
            l1=L.Linear(None, n_units),
            l2=L.Linear(None, n_out)
        )

    def __call__(self, x):
        h1 = self.activation(self.l1(x))
        return self.l2(h1)


class MLP3(chainer.Chain):

    def __init__(self, n_units, n_out=10, activation=F.relu):
        self.activation = activation
        super(MLP3, self).__init__(
            l1=L.Linear(None, n_units),
            l2=L.Linear(None, n_units),
            l3=L.Linear(None, n_out),
        )

    def __call__(self, x):
        h1 = self.activation(self.l1(x))
        h2 = self.activation(self.l2(h1))
        return self.l3(h2)


class MLP4(chainer.Chain):

    def __init__(self, n_units, n_out=10, activation=F.relu):
        self.activation = activation
        super(MLP4, self).__init__(
            l1=L.Linear(None, n_units),
            l2=L.Linear(None, n_units),
            l3=L.Linear(None, n_units),
            l4=L.Linear(None, n_out)
        )

    def __call__(self, x):
        h = self.activation(self.l1(x))
        h = self.activation(self.l2(h))
        h = self.activation(self.l3(h))
        return self.l4(h)


class MLP5(chainer.Chain):

    def __init__(self, n_units, n_out=10, activation=F.relu):
        self.activation = activation
        super(MLP5, self).__init__(
            l1=L.Linear(None, n_units),
            l2=L.Linear(None, n_units),
            l3=L.Linear(None, n_units),
            l4=L.Linear(None, n_units),
            l5=L.Linear(None, n_out)
        )

    def __call__(self, x):
        h = self.activation(self.l1(x))
        h = self.activation(self.l2(h))
        h = self.activation(self.l3(h))
        h = self.activation(self.l4(h))
        return self.l5(h)


class CNN1(chainer.Chain):

    def __init__(self, n_out=10):
        super(CNN1, self).__init__(
            conv1=F.Convolution2D(None, 20, ksize=3),
            fc1=F.Linear(None, 500),
            fc2=F.Linear(None, n_out)
        )

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


class CNN2(chainer.Chain):

    def __init__(self, n_out=10):
        super(CNN2, self).__init__(
            conv1=F.Convolution2D(None, 32, ksize=3),
            conv2=F.Convolution2D(None, 32, ksize=3),
            fc1=F.Linear(None, 500),
            fc2=F.Linear(None, n_out)
        )

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=2)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


class CNN3(chainer.Chain):

    def __init__(self, n_out=10):
        super(CNN3, self).__init__(
            conv1=F.Convolution2D(None, 32, ksize=3),
            conv2=F.Convolution2D(None, 32, ksize=3),
            conv3=F.Convolution2D(None, 32, ksize=3),
            fc1=F.Linear(None, 500),
            fc2=F.Linear(None, n_out)
        )

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=2)
        h = F.max_pooling_2d(F.relu(self.conv3(h)), ksize=2)
        h = F.relu(self.fc1(h))
        return self.fc2(h)
