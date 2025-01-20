module SNdata

using Artifacts, ArtifactUtils
using DelimitedFiles
using DataFrames
using LinearAlgebra
using CSV
using LaTeXStrings
include("Likelihood_utils.jl")

export SN, load_SN
export 

function __init__()

    print("Loading data and covariance through artifact")
    global DESY5SN_cov = vec(readdlm(joinpath(artifact"DESY5SN", "STAT+SYS.txt")))
    global DESY5SN_data = CSV.read(joinpath(artifact"DESY5SN", "DES-SN5YR_HD.csv"), DataFrame)
    global PantheonPlusSN_cov = vec(readdlm(joinpath(artifact"PantheonPlusSN", "Pantheon+SH0ES_STAT+SYS.cov")))
    global PantheonPlusSN_data = CSV.read(joinpath(artifact"PantheonPlusSN", "Pantheon+SH0ES.dat"), DataFrame)
    global Union3SN_cov = vec(readdlm(joinpath(artifact"Union3SN", "mag_covmat.txt")))
    global Union3SN_data = CSV.read(joinpath(artifact"Union3SN", "lcparam_full.txt"), DataFrame, delim=" ")

    

end

struct SN

    catalogue::String                  # DESY5, PantheonPlus, Union3
    nSN::Integer                       # Number of SNe
    covariance::Matrix{Float64}        # Covariance matrix
    inv_covariance::Matrix{Float64}    # Inverse of covariance matrix
    data::DataFrame                    # SN DataFrame

    function SN(catalogue, nSN, covariance, inv_covariance, data)
        new(catalogue, nSN, covariance, inv_covariance, data)
    end
end
    
function load_SN(catalogue::String)

    if catalogue=="DESY5"
        cov_data = DESY5SN_cov
        sn_data = DESY5SN_data
    elseif catalogue=="PantheonPlus"
        cov_data = PantheonPlusSN_cov
        sn_data = PantheonPlusSN_data
    elseif catalogue=="Union3"
        cov_data = Union3SN_cov
        sn_data = Union3SN_data
    else
        return "Invalid Catalogue Name"
    end

    if cov_data[1] != nrow(sn_data)
        throw(DimensionMismatch("Number of covariance entries does not match data entries."))
    else
        nSN = Int(cov_data[1])
    end

    cov_mat = reshape(cov_data[2:end], nSN, nSN)
    inv_cov_mat = inv(cov_mat)

    return SN(catalogue, nSN, cov_mat, inv_cov_mat, sn_data)
    
end

end # module SNdata
