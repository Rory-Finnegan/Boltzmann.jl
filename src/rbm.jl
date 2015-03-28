
using Distributions
using ArrayViews
using Base.LinAlg.BLAS

import Base.length


abstract RBM

typealias Mat{T} AbstractArray{T, 2}
typealias Vec{T} AbstractArray{T, 1}


const UNIT_CLASSES = [:bernoulli, :gaussian]


type Units
    bias::Vector{Float64}
    class::Symbol

    function Units(num::Int, class::Symbol=:bernoulli)
        #@assert class in UNIT_CLASSES
        new(zeros(num), class)
    end

    function Units(params::Dict)
        @assert haskey(params, "num")
        if haskey(params, "class")
            return new(zeros(params["num"]), symbol(params["class"]))
        else
            return new(zeros(params["num"]), :bernoulli)
        end
    end

    function Base.show(io::IO, units::Units)
        num = length(units)
        print(io, "Units(class=$(units.class), num=$num)")
    end
end

length(units::Units) = length(units.bias)
insert!(units::Units, index::Int, item::Any) = insert!(units.bias, index, item)
deleteat!(units::Units, index::Int) = deleteat!(units.bias, index)


type BaseRBM <: RBM
    W::Matrix{Float64}
    visible::Units
    hidden::Units
    dW_prev::Matrix{Float64}
    persistent_chain::Matrix{Float64}
    momentum::Float64
    directory

    function BaseRBM(visible::Units, hidden::Units; sigma=0.001, momentum=0.9, directory="./")
        n_hid = length(hidden)
        n_vis = length(visible)

        new(
            rand(Normal(0, sigma), (n_hid, n_vis)),
            visible, hidden,
            zeros(n_hid, n_vis),
            Array(Float64, 0, 0),
            momentum,
            directory
        )
    end

    function BaseRBM(params::Dict)
        @assert haskey(params, "visible")
        @assert haskey(params, "hidden")

        visible = Units(params["visible"])
        hidden = Units(params["hidden"])
        sigma = 0.001
        momentum = 0.9
        directory="./"

        sigma = haskey(params, "sigma") ? params["sigma"] : sigma
        momentum = haskey(params, "momentum") ? params["momentum"] : momentum
        directory = haskey(params, "directory") ? params["directory"] : directory

        n_vis = length(visible)
        n_hid = length(hidden)

        new(
            rand(Normal(0, sigma), (n_hid, n_vis)),
            visible, hidden,
            zeros(n_hid, n_vis),
            Array(Float64, 0, 0),
            momentum,
            directory
        )
    end

    function Base.show(io::IO, rbm::BaseRBM)
        n_vis = length(rbm.visible)
        n_hid = length(rbm.hidden)
        print(io, "BaseRBM($n_vis, $n_hid, momentum=$(rbm.momentum))")
    end
end


function logistic(x)
    return 1 ./ (1 + exp(-x))
end


function mean_hiddens(rbm::RBM, vis::Mat{Float64})
    p = rbm.W * vis .+ rbm.hidden.bias
    return logistic(p)
end


function sample_hiddens(rbm::RBM, vis::Mat{Float64})
    p = mean_hiddens(rbm, vis)
    return float(rand(size(p)) .< p)
end


function sample_visibles(rbm::RBM, hid::Mat{Float64})
    if rbm.visible.class == :bernoulli
        p = rbm.W' * hid .+ rbm.visible.bias
        p = logistic(p)
        return float(rand(size(p)) .< p)
    elseif rbm.visible.class == :gaussian
        mu = logistic(rbm.W' * hid .+ rbm.visible.bias)
        sigma2 = 0.01                   # using fixed standard diviation
        samples = zeros(size(mu))
        for j=1:size(mu, 2), i=1:size(mu, 1)
            samples[i, j] = rand(Normal(mu[i, j], sigma2))
        end
        return samples
    end
end


function gibbs(rbm::RBM, vis::Mat{Float64}; n_times=1)
    v_pos = vis
    h_pos = sample_hiddens(rbm, v_pos)
    v_neg = sample_visibles(rbm, h_pos)
    h_neg = sample_hiddens(rbm, v_neg)
    for i=1:n_times-1
        v_neg = sample_visibles(rbm, h_neg)
        h_neg = sample_hiddens(rbm, v_neg)
    end
    return v_pos, h_pos, v_neg, h_neg
end


function free_energy(rbm::RBM, vis::Mat{Float64})
    vb = sum(vis .* rbm.visible.bias, 1)
    Wx_b_log = sum(log(1 + exp(rbm.W * vis .+ rbm.hidden.bias)), 1)
    return - vb - Wx_b_log
end

# This function is very memory intensive. Force subsampling of the input
# space with vectors should allow people to trade of itereations of memory consumption
function score_samples(rbm::RBM, X::Mat{Float64}, scoring_ratio)
    # Randomly select the scoring sample view
    sample_size = int(size(X, 2) * scoring_ratio)
    start = rand(1:size(X,2)-sample_size)
    vis = view(X, :, start:start+sample_size)

    # Still have to do a copy
    n_feat, n_samples = size(vis)
    vis_corrupted = copy(vis)
    idxs = rand(1:n_feat, n_samples)
    for (i, j) in zip(idxs, 1:n_samples)
        vis_corrupted[i, j] = 1 - vis_corrupted[i, j]
    end

    # These are expensive, but necessary for now.
    fe = free_energy(rbm, vis)
    fe_corrupted = free_energy(rbm, vis_corrupted)

    return n_feat * log(logistic(fe_corrupted - fe))
end


function update_weights!(rbm::RBM, h_pos, v_pos, h_neg, v_neg, lr, buf)
    dW = buf
    # dW = (h_pos * v_pos') - (h_neg * v_neg')
    gemm!('N', 'T', 1.0, h_neg, v_neg, 0.0, dW)
    gemm!('N', 'T', 1.0, h_pos, v_pos, -1.0, dW)
    # rbm.W += lr * dW
    axpy!(lr, dW, rbm.W)
    # rbm.W += rbm.momentum * rbm.dW_prev
    axpy!(lr * rbm.momentum, rbm.dW_prev, rbm.W)
    # save current dW
    copy!(rbm.dW_prev, dW)
end


function contdiv(rbm::RBM, vis::Mat{Float64}, n_gibbs::Int)
    v_pos, h_pos, v_neg, h_neg = gibbs(rbm, vis, n_times=n_gibbs)
    return v_pos, h_pos, v_neg, h_neg
end


function persistent_contdiv(rbm::RBM, vis::Mat{Float64}, n_gibbs::Int)
    if size(rbm.persistent_chain) != size(vis)
        # persistent_chain not initialized or batch size changed, re-initialize
        rbm.persistent_chain = vis
    end
    # take positive samples from real data
    v_pos, h_pos, _, _ = gibbs(rbm, vis)
    # take negative samples from "fantasy particles"
    rbm.persistent_chain, _, v_neg, h_neg = gibbs(rbm, vis, n_times=n_gibbs)
    return v_pos, h_pos, v_neg, h_neg
end


function fit_batch!(rbm::RBM, vis::Mat{Float64};
                    persistent=true, buf=None, lr=0.1, n_gibbs=1)
    buf = buf == None ? zeros(size(rbm.W)) : buf
    # v_pos, h_pos, v_neg, h_neg = gibbs(rbm, vis, n_times=n_gibbs)
    sampler = persistent ? persistent_contdiv : contdiv
    v_pos, h_pos, v_neg, h_neg = sampler(rbm, vis, n_gibbs)
    lr = lr / size(v_pos, 1)
    update_weights!(rbm, h_pos, v_pos, h_neg, v_neg, lr, buf)
    rbm.hidden.bias += vec(lr * (sum(h_pos, 2) - sum(h_neg, 2)))
    rbm.visible.bias += vec(lr * (sum(v_pos, 2) - sum(v_neg, 2)))
    return rbm
end


function fit(rbm::RBM, X::Mat{Float64};
             persistent=true, lr=0.1, n_iter=10, batch_size=100, n_gibbs=1, scoring_ratio=0.1)
    @assert minimum(X) >= 0 && maximum(X) <= 1
    n_samples = size(X, 2)
    n_batches = int(ceil(n_samples / batch_size))
    w_buf = zeros(size(rbm.W))
    batch = Array(Float64, size(X, 1), batch_size)  # pre-allocate batch
    for itr=1:n_iter
        tic()
        for i=1:n_batches
            # println("fitting $(i)th batch")
            batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, size(X, 2))]
            batch = full(batch)
            fit_batch!(rbm, batch, persistent=persistent,
                       buf=w_buf, n_gibbs=n_gibbs)
        end
        toc()
        pseudo_likelihood = mean(score_samples(rbm, X, scoring_ratio))
        info("Iteration #$itr, pseudo-likelihood = $pseudo_likelihood")
    end
    return rbm
end


function transform(rbm::RBM, X::Mat{Float64})
    return mean_hiddens(rbm, X)
end


function generate(rbm::RBM, vis::Vec{Float64}; n_gibbs=1)
    return gibbs(rbm, reshape(vis, length(vis), 1); n_times=n_gibbs)[3]
end

function generate(rbm::RBM, X::Mat{Float64}; n_gibbs=1)
    return gibbs(rbm, X; n_times=n_gibbs)[3]
end


function features(rbm::RBM; transpose=true)
    return if transpose rbm.W' else rbm.W end
end
# synonym for backward compatibility
components(rbm::RBM; transpose=true) = features(rbm, transpose)
