using Boltzmann
using MNIST

X, y = traindata()
X = X[:, 1:1000]                     # take only 1000 observations for speed
X = X / (maximum(X) - (minimum(X)))  # normalize to [0..1]

layers = [("vis", BaseRBM(Units(784, :gaussian), Units(256))),
          ("hid1", BaseRBM(Units(256), Units(100))),
          ("hid2", BaseRBM(Units(100), Units(100)))]
dbn = DBN(layers)
fit(dbn, X)
transform(dbn, X)

