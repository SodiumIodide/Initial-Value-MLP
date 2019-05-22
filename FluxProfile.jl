#!/usr/bin/env julia

# FluxProfile.jl

include("GeometryGen.jl")
include("MeshMap.jl")
include("RunningStatistics.jl")
include("Constants.jl")

using .GeometryGen
using .MeshMap
using .RunningStatistics
using .Constants

using GaussQuadrature
using Random

set_zero_subnormals(true)

@inline function linspace(x_start::Float64, x_fin::Float64, x_len::Int64)::Vector{Float64}
    local delta_x::Float64 = @fastmath (x_fin - x_start) / (x_len - 1)
    return @fastmath [(x_start + (i - 1) * delta_x) for i in 1:x_len]
end

function main()::Nothing
    # Assignment of material variables
    local macro_tot::Vector{Float64} = Constants.tot_const
    local macro_scat::Vector{Float64} = Constants.scat_const
    local phi_mat::Array{Float64, 2} = fill(0.0, Constants.num_cells, Constants.num_materials)
    local phi_mat_stat::Array{RunningStat, 2} = [RunningStat() for i in 1:Constants.num_cells, j in 1:Constants.num_materials]
    local psi_mat::Array{Float64, 3} = fill(0.0, Constants.num_cells, Constants.num_ords, Constants.num_materials)
    local psi_mat_stat::Array{RunningStat, 3} = [RunningStat() for i in 1:Constants.num_cells, j in 1:Constants.num_ords, k in 1:Constants.num_materials]

    # Assignment of initial calculation variables
    local phi_outer::Vector{Float64} = fill(0.0, Constants.num_cells)
    local total_chord::Float64 = @fastmath sum(Constants.chord)  # cm
    local prob::Vector{Float64} = @fastmath @. Constants.chord / total_chord

    # X values of flux for plotting
    local cell_vector::Vector{Float64} = linspace(0.0, Constants.thickness, Constants.num_cells)

    # Legendre Gauss Quadrature over chosen ordinates
    local (mu::Vector{Float64}, weights::Vector{Float64}) = legendre(Constants.num_ords)
    # Overload for S2
    if (length(mu) == 2)
        mu = vec([-1.0, 1.0])
    end

    # Boundary conditions: 1/cm^3-s-MeV-strad
    # Left boundary
    # Isotropic source:
    local psi_bound_l::Vector{Float64} = fill(Constants.incident_angular_flux, Constants.num_ords)
    # Vacuum source:
    #local psi_bound_l::Vector{Float64} = fill(0.0, Constants.num_ords)
    # Beam source (in conj. with vacuum):
    #psi_bound_l[Constants.num_ords] = 1.0 / (mu[Constants.num_ords] * weights[Constants.num_ords])

    # Right boundary
    # Isotropic source:
    #local psi_bound_r::Vector{Float64} = fill(2.0, Constants.num_ords)
    # Vacuum source:
    local psi_bound_r::Vector{Float64} = fill(0.0, Constants.num_ords)
    # Beam source (in conj. with vacuum):
    #psi_bound_r[Constants.num_ords] = 1.0 / (mu[Constants.num_ords] * weights[Constants.num_ords])

    local generator::MersenneTwister = MersenneTwister(Constants.seed)

    # Calculation: iterations
    local cont_calc_outer::Bool = true
    # Start counter at zero
    local iterations_outer::Int64 = 0

    # Outer loop over overall mixed system
    while (cont_calc_outer)
        local cont_calc_inner::Bool = true
        # Start inner iterations at zero
        local iterations_inner::Int64 = 0

        # Fill Markovian geometry
        @inbounds local (x_delta::Vector{Float64}, x_arr::Vector{Float64}, materials::Vector{Int32}, num_r_cells::Int64) = GeometryGen.get_geometry(Constants.chord[1], Constants.chord[2], Constants.thickness, Constants.num_geom_divs, rng=generator)

        # Calculational arrays
        local psi::Array{Float64, 2} = fill(0.0, num_r_cells, Constants.num_ords)
        local psi_i_p::Array{Float64, 2} = fill(0.0, num_r_cells, Constants.num_ords)
        local psi_i_m::Array{Float64, 2} = fill(0.0, num_r_cells, Constants.num_ords)

        # Initial source terms
        local scat_source::Vector{Float64} = fill(0.0, num_r_cells)
        local spont_source::Vector{Float64} = fill(0.0, num_r_cells)
        local tot_source::Vector{Float64} = fill(0.0, num_r_cells)

        # Phi calculations
        local phi_morph_new::Vector{Float64} = fill(1.0, num_r_cells)
        local phi_morph_old::Vector{Float64} = fill(0.0, num_r_cells)

        # Flag to tell whether to use the inner loop data in averaging or not
        local converged::Bool = false
        # Inner loop over generated geometry
        while (cont_calc_inner)
            phi_morph_old = deepcopy(phi_morph_new)
            @simd for c = 1:num_r_cells
                @inbounds scat_source[c] = @fastmath macro_scat[materials[c]] / 2.0 * phi_morph_new[c]
                @inbounds spont_source[c] = @fastmath Constants.spont_source_const[materials[c]] / 2.0
                @inbounds tot_source[c] = scat_source[c] + spont_source[c]
            end

            # Forward sweep (left to right)
            # First cell (left boundary)
            # Ordinate loop, only consider the pos. ords for forward motion
            @simd for m = @fastmath convert(Int64, Constants.num_ords / 2 + 1):Constants.num_ords
                @inbounds psi[1, m] = @fastmath (1.0 + (macro_tot[materials[1]] * x_delta[1]) / (2.0 * abs(mu[m])))^(-1) * (psi_bound_l[m] + (tot_source[1] * x_delta[1]) / (2.0 * abs(mu[m])))
                @inbounds psi_i_p[1, m] = @fastmath 2.0 * psi[1, m] - psi_bound_l[m]
            end
            # Rest of the cells (sans left bounding cell)
            for c = 2:num_r_cells
                @simd for m = @fastmath convert(Int64, Constants.num_ords / 2 + 1):Constants.num_ords
                    # Continuity of boundaries
                    @inbounds psi_i_m[c, m] = @fastmath psi_i_p[c - 1, m]
                    @inbounds psi[c, m] = @fastmath (1.0 + (macro_tot[materials[c]] * x_delta[c]) / (2.0 * abs(mu[m])))^(-1) * (psi_i_m[c, m] + (tot_source[c] * x_delta[c]) / (2.0 * abs(mu[m])))
                    @inbounds psi_i_p[c, m] = @fastmath 2.0 * psi[c, m] - psi_i_m[c, m]
                end
            end

            # Backward sweep (right to left)
            # First cell (right boundary)
            # Ordinate loop, only consider neg. ords for backwards motion
            @simd for m = @fastmath 1:convert(Int64, Constants.num_ords / 2)
                @inbounds psi[num_r_cells, m] = @fastmath (1.0 + (macro_tot[materials[num_r_cells]] * x_delta[num_r_cells]) / (2.0 * abs(mu[m])))^(-1) * (psi_bound_r[m] + (tot_source[num_r_cells] * x_delta[num_r_cells]) / (2.0 * abs(mu[m])))
                @inbounds psi_i_m[num_r_cells, m] = @fastmath 2.0 * psi[num_r_cells, m] - psi_bound_r[m]
            end
            # Rest of the cells (sans right bounding cell)
            for c = @fastmath (num_r_cells - 1):-1:1
                @simd for m = @fastmath 1:convert(Int64, Constants.num_ords / 2)
                    # Continuity of boundaries
                    @inbounds psi_i_p[c, m] = @fastmath psi_i_m[c + 1, m]
                    @inbounds psi[c, m] = @fastmath (1.0 + (macro_tot[materials[c]] * x_delta[c]) / (2.0 * abs(mu[m])))^(-1) * (psi_i_p[c, m] + (tot_source[c] * x_delta[c]) / (2.0 * abs(mu[m])))
                    @inbounds psi_i_m[c, m] = @fastmath 2.0 * psi[c, m] - psi_i_p[c, m]
                end
            end

            # Calculate phi from psi
            @simd for c = 1:num_r_cells
                local weighted_sum::Float64 = 0.0
                @simd for m = 1:Constants.num_ords
                    @inbounds @fastmath weighted_sum += weights[m] * psi[c, m]
                end
                @inbounds phi_morph_new[c] = weighted_sum  # 1/cm^2-s-MeV
            end

            # Relative error for inner loop
            @fastmath iterations_inner += 1
            local err_inner::Float64 = @inbounds @fastmath maximum(@. abs(phi_morph_old - phi_morph_new) / phi_morph_new)
            if @fastmath (err_inner <= Constants.inner_tolerance)
                cont_calc_inner = false
                converged = true
            elseif @fastmath (iterations_inner > Constants.num_iter_inner)
                cont_calc_inner = false
                println("No convergence on loop number ", iterations_outer, ": quit after maximum ", iterations_inner, " iterations; data will not be used")
            end
        end  # Inner loop

        if (converged)
            @fastmath iterations_outer += 1

            # Obtain material balances
            # Average flux in structured cells
            phi_mat = material_calc(phi_morph_new, x_delta, num_r_cells, materials, Constants.struct_thickness, Constants.num_cells, Constants.num_materials)

            # Average angular flux in structured cells
            @simd for m = 1:Constants.num_ords
                psi_mat[:, m, :] = material_calc(psi[:, m], x_delta, num_r_cells, materials, Constants.struct_thickness, Constants.num_cells, Constants.num_materials)
            end

            # Average flux computed from the inner loops, saved to a running statistics computation
            @simd for k = 1:Constants.num_materials
                # Save flux statistics
                @simd for c = 1:Constants.num_cells
                    if @inbounds @fastmath(phi_mat[c, k] != 0.0)
                        @inbounds push(phi_mat_stat[c, k], phi_mat[c, k])
                    end
                    @simd for m = 1:Constants.num_ords
                        if @inbounds @fastmath(psi_mat[c, m, k] != 0.0)
                            @inbounds push(psi_mat_stat[c, m, k], psi_mat[c, m, k])
                        end
                    end
                end  # Cell loop
            end  # Material loop

            if @fastmath (iterations_outer % Constants.num_say == 0)
                println("Realization number ", iterations_outer)
            end

            if @fastmath (iterations_outer > Constants.num_iter_outer)
                cont_calc_outer = false
            end
        end  # Logical test for inner convergence: don't average nonconverged samples
    end  # Outer loop

    # Save flux data
    local flux_total::IOStream = open("out/total_flux.out", "w")
    local flux_1::IOStream = open("out/flux_1.out", "w")
    local flux_2::IOStream = open("out/flux_2.out", "w")
    local psi_1_m::IOStream = open("out/psi_1_m.out", "w")
    local psi_1_p::IOStream = open("out/psi_1_p.out", "w")
    local psi_2_m::IOStream = open("out/psi_2_m.out", "w")
    local psi_2_p::IOStream = open("out/psi_2_p.out", "w")
    for i = 1:Constants.num_cells
        phi_outer[i] = prob[1] * mean(phi_mat_stat[i, 1]) + prob[2] * mean(phi_mat_stat[i, 2])
        write(flux_total, string(cell_vector[i]), ",", string(phi_outer[i]), "\n")
        write(flux_1, string(cell_vector[i]), ",", string(mean(phi_mat_stat[i, 1])), "\n")
        write(flux_2, string(cell_vector[i]), ",", string(mean(phi_mat_stat[i, 2])), "\n")
        write(psi_1_m, string(cell_vector[i]), ",", string(mean(psi_mat_stat[i, 1, 1])), "\n")
        write(psi_1_p, string(cell_vector[i]), ",", string(mean(psi_mat_stat[i, 2, 1])), "\n")
        write(psi_2_m, string(cell_vector[i]), ",", string(mean(psi_mat_stat[i, 1, 2])), "\n")
        write(psi_2_p, string(cell_vector[i]), ",", string(mean(psi_mat_stat[i, 2, 2])), "\n")
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
