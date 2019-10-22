#!/usr/bin/env julia

# InitialCondition.jl

include("Constants.jl")

using .Constants

using LinearAlgebra

set_zero_subnormals(true)

@inline function linspace(x_start::Float64, x_fin::Float64, x_len::Int64)::Vector{Float64}
    local delta_x::Float64 = @fastmath (x_fin - x_start) / (x_len - 1)
    return @fastmath [(x_start + (i - 1) * delta_x) for i in 1:x_len]
end

function main()::Nothing
    local lp::Bool = false

    # Assignment of material variables
    local macro_tot::Vector{Float64} = Constants.tot_const
    local macro_scat::Vector{Float64} = Constants.scat_const

    # Assignemnt of initial calculation variables
    local total_chord::Float64 = @fastmath sum(Constants.chord)  # cm
    local prob::Vector{Float64} = @fastmath @. Constants.chord / total_chord

    # X values of flux for plotting
    local cell_vector::Vector{Float64} = linspace(0.0, Constants.thickness, Constants.num_cells)
    local delta_x::Float64 = @fastmath Constants.thickness / (Constants.num_cells - 1)

    # Angular flux profiles
    local psi_vectors::Array{Float64, 2} = @fastmath fill(0.0, Constants.num_cells, Constants.num_materials * 2)

    # Flux profiles
    local phi_results::Array{Float64} = fill(0.0, Constants.num_cells, Constants.num_materials)
    local phi_total::Vector{Float64} = fill(0.0, Constants.num_cells)

    # "Initial" conditions: 1/cm^2-s-MeV
    local psi_m_init::Vector{Float64} = vec([
        0.08904936657443904  # Material 1: 1/cm^3-s-MeV-strad
        0.08881318029270834  # Material 2: 1/cm^3-s-MeV-strad
    ])
    local psi_p_init::Vector{Float64} = vec([
        Constants.incident_angular_flux  # Material 1: 1/cm^3-s-MeV-strad
        Constants.incident_angular_flux  # Material 2: 1/cm^3-s-MeV-strad
    ])
    @inbounds psi_vectors[1, 1] = @fastmath psi_p_init[1] * prob[1]
    @inbounds psi_vectors[1, 2] = @fastmath psi_m_init[1] * prob[1]
    @inbounds psi_vectors[1, 3] = @fastmath psi_p_init[2] * prob[2]
    @inbounds psi_vectors[1, 4] = @fastmath psi_m_init[2] * prob[2]
    @inbounds phi_results[1, 1] = @fastmath (psi_vectors[1, 1] + psi_vectors[1, 2]) / prob[1]
    @inbounds phi_results[1, 2] = @fastmath (psi_vectors[1, 3] + psi_vectors[1, 4]) / prob[2]

    # Construct the multiplier matrix
    local function build_matrix()::Array{Float64, 2}
        local term_1::Float64 = @inbounds @fastmath 0.5 * macro_scat[1] - macro_tot[1] - Constants.chord[1]^(-1)
        local term_2::Float64 = @inbounds @fastmath 0.5 * macro_scat[1]
        local term_3::Float64 = @inbounds @fastmath Constants.chord[2]^(-1)
        local term_4::Float64 = 0.0
        local term_5::Float64 = @inbounds @fastmath -0.5 * macro_scat[1]
        local term_6::Float64
        if @fastmath (lp)
            term_6 = @inbounds @fastmath macro_tot[1] - 0.5 * macro_scat[1] + Constants.chord[1]^(-1)
        else
            term_6 = @inbounds @fastmath macro_tot[1] - 0.5 * macro_scat[1] - Constants.chord[1]^(-1)
        end
        local term_7::Float64 = 0.0
        local term_8::Float64
        if @fastmath (lp)
            term_8 = @inbounds @fastmath - Constants.chord[2]^(-1)
        else
            term_8 = @inbounds @fastmath Constants.chord[2]^(-1)
        end
        local term_9::Float64 = @inbounds @fastmath Constants.chord[1]^(-1)
        local term_10::Float64 = 0.0
        local term_11::Float64 = @inbounds @fastmath 0.5 * macro_scat[2] - macro_tot[2] - Constants.chord[2]^(-1)
        local term_12::Float64 = @inbounds @fastmath 0.5 * macro_scat[2]
        local term_13::Float64 = 0.0
        local term_14::Float64
        if @fastmath (lp)
            term_14 = @inbounds @fastmath - Constants.chord[1]^(-1)
        else
            term_14 = @inbounds @fastmath Constants.chord[1]^(-1)
        end
        local term_15::Float64 = @inbounds @fastmath -0.5 * macro_scat[2]
        local term_16::Float64
        if @fastmath (lp)
            term_16 = @inbounds @fastmath macro_tot[2] - 0.5 * macro_scat[2] + Constants.chord[2]^(-1)
        else
            term_16 = @inbounds @fastmath macro_tot[2] - 0.5 * macro_scat[2] - Constants.chord[2]^(-1)
        end

        return [
            term_1 term_2 term_3 term_4
            term_5 term_6 term_7 term_8
            term_9 term_10 term_11 term_12
            term_13 term_14 term_15 term_16
        ]
    end
    local multiplier_matrix::Array{Float64, 2} = build_matrix()

    println(eigvals(multiplier_matrix))

    for i in 2:Constants.num_cells
        local x_value::Float64 = @inbounds cell_vector[i]

        # Solve the matrix exponential
        local psi_results::Vector{Float64} = @inbounds @fastmath exp(multiplier_matrix * delta_x) * vec(psi_vectors[i - 1, :])
        @inbounds psi_vectors[i, :] = deepcopy(psi_results)
        #@show psi_results

        # Compute phi from psi
        @inbounds phi_results[i, 1] = @fastmath (psi_results[1] + psi_results[2]) / prob[1]
        @inbounds phi_results[i, 2] = @fastmath (psi_results[3] + psi_results[4]) / prob[2]
    end

    # Save flux data
    local flux_total::IOStream = open("out/ic_total_flux.out", "w")
    local flux_1::IOStream = open("out/ic_flux_1.out", "w")
    local flux_2::IOStream = open("out/ic_flux_2.out", "w")
    local psi_1_m::IOStream = open("out/ic_psi_1_m.out", "w")
    local psi_1_p::IOStream = open("out/ic_psi_1_p.out", "w")
    local psi_2_m::IOStream = open("out/ic_psi_2_m.out", "w")
    local psi_2_p::IOStream = open("out/ic_psi_2_p.out", "w")
    for i = 1:Constants.num_cells
        phi_total[i] = @fastmath prob[1] * phi_results[i, 1] + prob[2] * phi_results[i, 2]
        write(flux_total, string(cell_vector[i]), ",", string(phi_total[i]), "\n")
        write(flux_1, string(cell_vector[i]), ",", string(phi_results[i, 1]), "\n")
        write(flux_2, string(cell_vector[i]), ",", string(phi_results[i, 2]), "\n")
        write(psi_1_p, string(cell_vector[i]), ",", string(psi_vectors[i, 1] / prob[1]), "\n")
        write(psi_1_m, string(cell_vector[i]), ",", string(psi_vectors[i, 2] / prob[1]), "\n")
        write(psi_2_p, string(cell_vector[i]), ",", string(psi_vectors[i, 3] / prob[2]), "\n")
        write(psi_2_m, string(cell_vector[i]), ",", string(psi_vectors[i, 4] / prob[2]), "\n")
    end
    close(flux_total)
    close(flux_1)
    close(flux_2)
    close(psi_1_m)
    close(psi_1_p)
    close(psi_2_m)
    close(psi_2_p)

    return nothing
end

main()
