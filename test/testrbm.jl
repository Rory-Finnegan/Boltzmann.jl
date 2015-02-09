
using Boltzmann
using Base.Test

X = rand(1000, 2000)


function rbm_smoke_test()
    model = BaseRBM(Units(1000), Units(500))
    fit(model, X)
end

function gaussian_rbm_smoke_test()
    model = BaseRBM(Units(1000, :gaussian), Units(500))
    fit(model, X)
end


rbm_smoke_test()
gaussian_rbm_smoke_test()
