using SNdata
using Turing
using DataFrames
using LinearAlgebra
using Optim
using StatsBase
using Effort
using Plots
using CairoMakie
using PairPlots

##    Loading catalogue and preprocessing obs_data 

struct DESY5SN_info
    data::DataFrame
    obs_flatdata::Vector{Float64}
    covariance::Matrix{Float64}
    precision::Matrix{Float64}
    std::Vector{Float64}

    function DESY5SN_info()
        
        DES_data = load_SN("DESY5")
        
        obs_flatdata = @. DES_data.data.MU - (5 * log10((1 + DES_data.data.zHEL) / (1 + DES_data.data.zHD)))
        covariance = DES_data.covariance + diagm(DES_data.data.MUERR_FINAL .^ 2)
        precision = inv(covariance)
        std = sqrt.(diag(covariance))
        
        new(DES_data.data, obs_flatdata, covariance, precision, std)
    end
end

struct PantheonPlusSN_info
    data::DataFrame
    obs_flatdata::Vector{Float64}
    covariance::Matrix{Float64}
    precision::Matrix{Float64}
    std::Vector{Float64}

    function PantheonPlusSN_info()
        
        PP_data = load_SN("PantheonPlus")
        
        z_mask = PP_data.data.zHD.>0.01
        masked_PP_df = PP_data.data[z_mask, :]
        
        obs_flatdata = @. masked_PP_df.m_b_corr - 5 * (log10((1 + masked_PP_df.zHEL) / (1 + masked_PP_df.zHD)))
        covariance = PP_data.covariance[z_mask, z_mask]
        precision = inv(covariance)
        std = sqrt.(diag(covariance))
        
        new(masked_PP_df, obs_flatdata, covariance, precision, std)
    end
end

struct Union3SN_info
    data::DataFrame
    obs_flatdata::Vector{Float64}
    covariance::Matrix{Float64}
    precision::Matrix{Float64}
    std::Vector{Float64}

    function Union3SN_info()
        
        un3_data = load_SN("Union3")
        
        obs_flatdata = un3_data.data.mb
        covariance = Hermitian(un3_data.covariance)
        precision = un3_data.inv_covariance
        std = sqrt.(diag(covariance))
        
        new(un3_data.data, obs_flatdata, covariance, precision, std)
    end
end

##    Chi2 function

function chi2(flatdiff::Vector, precision::Matrix)
    return flatdiff' * precision * flatdiff
end

##    Computing classical loglikelihood 

function compute_DESY5SNLikelihood(desy5::DESY5SN_info, Ωcb0, h, mν, w0, wa; Mb=0.)
    
    zHD = desy5.data.zHD
    
    function D_L(z)
        return Effort._r_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa)*(1+z)*h
    end
    
    flattheory = @. 5 * log10(D_L(zHD) / h) + 25
    flatdata = @. desy5.obs_flatdata - Mb
    flatdiff = flattheory - flatdata
    loglikelihood = - 0.5 * chi2(flatdiff, desy5.precision)

    return loglikelihood
end

function compute_PantheonPlusSNLikelihood(pp::PantheonPlusSN_info, Ωcb0, h, mν, w0, wa; Mb=0.)

    zHD = pp.data.zHD
    
    function D_L(z)
        return Effort._r_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa)*(1+z)*h
    end

    flattheory = @. 5 * log10(D_L(zHD) / h) + 25
    flatdata = @. pp.obs_flatdata - Mb
    flatdiff = flattheory - flatdata
    loglikelihood = - 0.5 * chi2(flatdiff, pp.precision)

    return loglikelihood
end

function compute_Union3SNLikelihood(un3::Union3SN_info, Ωcb0, h, mν, w0, wa; dM=0.)
    
    zcmb = un3.data.zcmb
    
    function D_L(z)
        return Effort._r_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa)*(1+z)*h
    end
    
    flattheory = @. 5 * log10(100*D_L(zcmb)) + 25
    flatdata = @. un3.obs_flatdata - dM
    flatdiff = flattheory - flatdata
    loglikelihood = - 0.5 * chi2(flatdiff, un3.precision)

    return loglikelihood
end

##    Computing Turing.jl model of the logposterior (priors+likelihood)

@model function DESY5SNLikelihood(obs_flatdata, desy5_info)

    zHD = desy5_info.data.zHD
    cov = desy5_info.covariance

    Ωm ~ Uniform(0.01, 0.9)
    w0 ~ Uniform(-3, 1)
    wa ~ Uniform(-20, 2)
    Mb ~ Uniform(-5, 5)
    
    h = 0.6736
    mν = 0.
    
    function D_L(z)
        return Effort._r_z(z, Ωm, h; mν=mν, w0=w0, wa=wa)*(1+z)*h
    end
    
    flattheory = @. 5 * log10(D_L(zHD) / h) + 25
    obs_flatdata ~ MvNormal(flattheory .+ Mb, cov)
end

@model function PantheonPlusSNLikelihood(obs_flatdata, pp_info)

    zHD = pp_info.data.zHD
    cov = pp_info.covariance

    Ωm ~ Uniform(0.01, 0.9)
    w0 ~ Uniform(-3, 1)
    wa ~ Uniform(-20, 2)
    Mb ~ Uniform(-20, -18)
    
    h = 0.6736
    mν = 0.
    
    function D_L(z)
        return Effort._r_z(z, Ωm, h; mν=mν, w0=w0, wa=wa)*(1+z)*h
    end

    flattheory = @. 5 * log10(D_L(zHD) / h) + 25
    obs_flatdata ~ MvNormal(flattheory .+ Mb, cov)
end

@model function Union3SNLikelihood(obs_flatdata, un3_info)

    zCMB = un3_info.data.zcmb
    cov = un3_info.covariance

    Ωm ~ Uniform(0.01, 0.9)
    w0 ~ Uniform(-3, 1)
    wa ~ Uniform(-20, 2)
    dM ~ Uniform(-20, 20)
    
    h = 0.6736
    mν = 0.
    
    function D_L(z)
        return Effort._r_z(z, Ωm, h; mν=mν, w0=w0, wa=wa)*(1+z)*h
    end

    flattheory = @. 5 * log10(100*D_L(zCMB)) + 25
    obs_flatdata ~ MvNormal(flattheory .+ dM, cov)
end

##    Plotting chain

function plotting_chains(chain::Chains, labels::Dict, axis::Dict)
    
    c1 = Makie.wong_colors()[1]
    layers_series_1 = (
        PairPlots.Scatter(filtersigma=10, color=c1, markersize=2),
        PairPlots.Contourf(strokewidth=1.5, strokecolor=c1, color=(c1,0.2), bandwidth=1.5, sigmas=[1,3], linestyle=:dash),
        PairPlots.MarginHist(color=(c1, 0.2)),
        PairPlots.MarginStepHist(color=(c1, 0.8)),
        PairPlots.MarginDensity(
            color=c1,
            linewidth=1.5f0),
        PairPlots.PairPlots.MarginConfidenceLimits(PairPlots.margin_confidence_default_formatter, (0.16,0.5,0.84)),
        #PairPlots.MarginQuantileText(color=:black,font=:regular),
            #PairPlots.margin_confidence_default_formatter, (0.16,0.5,0.84)),
    )
    
    fig = pairplot(
        chain => layers_series_1,
        axis = axis,
        labels = labels
    )
    for i in 1:10
        ax = fig.content[i]
        ax.xticksize = 5
        ax.yticksize = 5
        ax.xlabelsize = 20
        ax.ylabelsize = 20
        ax.xticklabelsize = 18
        ax.yticklabelsize = 18
        ax.titlesize = 20
    end

    display(fig)
end
