#!/usr/bin/env julia

# SlabBoundaries.jl

include("RunningStatistics.jl")
include("GeometryGen.jl")
include("Constants.jl")

using .RunningStatistics
using .GeometryGen
using .Constants

using GaussQuadrature
using Random

set_zero_subnormals(true)

@inline function compute_error(old_value::Float64, new_value::Float64)::Float64
    if @fastmath (new_value == 0.0)
        return 0.0
    end
    return @fastmath abs(old_value - new_value) / new_value
end

function main()::Nothing
    local converge::Bool = false

    # Assignment of material variables
    local macro_tot::Vector{Float64} = Constants.tot_const
    local macro_scat::Vector{Float64} = Constants.scat_const
    local phi_mat_new::Array{Float64, 2} = fill(0.0, Constants.num_cells, Constants.num_materials)
    local phi_mat_old::Array{Float64, 2} = fill(1.0, Constants.num_cells, Constants.num_materials)

    # Assignment of initial calculation variables
    local phi_outer::Vector{Float64} = fill(0.0, Constants.num_cells)
    local total_chord::Float64 = @fastmath sum(Constants.chord)  # cm
    local prob::Vector{Float64} = @fastmath @. Constants.chord / total_chord

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
    local ord_leak_l_stat::Vector{RunningStat} = [RunningStat() for m in 1:Constants.num_ords]
    local ord_leak_r_stat::Vector{RunningStat} = [RunningStat() for m in 1:Constants.num_ords]
    local leakage_l_stat::Vector{RunningStat} = [RunningStat() for k in 1:Constants.num_materials]
    local leakage_r_stat::Vector{RunningStat} = [RunningStat() for k in 1:Constants.num_materials]

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

            local leakage_l_tally::Vector{Float64} = fill(0.0, Constants.num_materials)
            local leakage_r_tally::Vector{Float64} = fill(0.0, Constants.num_materials)
            local old_leakage_l::Vector{Float64} = @. mean(leakage_l_stat)
            local old_leakage_r::Vector{Float64} = @. mean(leakage_r_stat)

            # Average leakage from material boundaries (for balance/tally purposes)
            @simd for k = 1:Constants.num_materials
                # Calculate the average negative leakage from left boundary
                if @inbounds @fastmath (materials[1] == k)
                    @simd for m = @fastmath 1:convert(Int64, Constants.num_ords / 2)
                        @inbounds @fastmath push(ord_leak_l_stat[m], psi_i_m[1, m] * abs(mu[m]) * weights[m])
                        @inbounds @fastmath leakage_l_tally[k] += abs(mu[m]) * weights[m] * psi_i_m[1, m]
                    end
                end
                # Calculate the average positive leakage from right boundary
                if @inbounds @fastmath (materials[num_r_cells] == k)
                    @simd for m = convert(Int64, Constants.num_ords / 2 + 1):Constants.num_ords
                        @inbounds @fastmath push(ord_leak_r_stat[m], psi_i_p[num_r_cells, m] * abs(mu[m]) * weights[m])
                        @inbounds @fastmath leakage_r_tally[k] += abs(mu[m]) * weights[m] * psi_i_p[num_r_cells, m]
                    end
                end
                push(leakage_l_stat[k], leakage_l_tally[k] / prob[k])
                push(leakage_r_stat[k], leakage_r_tally[k] / prob[k])
            end

            local left_error::Float64 = @fastmath maximum(@. compute_error(old_leakage_l, mean(leakage_l_stat)))
            local right_error::Float64 = @fastmath maximum(@. compute_error(old_leakage_r, mean(leakage_r_stat)))

            if @fastmath (converge)
                if @fastmath ((left_error <= Constants.leakage_tolerance))
                cont_calc_outer = false
                end
            else
                if @fastmath (iterations_outer > Constants.num_iter_outer)
                    cont_calc_outer = false
                end
            end

            if @fastmath (iterations_outer % Constants.num_say == 0)
                println("Realization number ", iterations_outer, "; Reflection error: ", left_error)
            end
        end  # Logical test for inner convergence: don't average nonconverged samples
    end  # Outer loop

    # Save leakage data
    local output::IOStream = open("out/leakage.txt", "w")
    write(output, "Total Leakage Left,", string(mean(leakage_l_stat[1]) + mean(leakage_l_stat[2])), "\n")
    write(output, "Total Leakage Right,", string(mean(leakage_r_stat[1]) + mean(leakage_r_stat[2])), "\n")
    for k = 1:Constants.num_materials
        write(output, "Material ", string(k), " Leakage Left,", string(mean(leakage_l_stat[k])), "\n")
        write(output, "Material ", string(k), " Leakage Right,", string(mean(leakage_r_stat[k])), "\n")
    end
    write(output, "Number of Ordinates,", string(Constants.num_ords), "\n")
    for m = @fastmath 1:convert(Int64, Constants.num_ords / 2)
        @inbounds write(output, "Ordinate ", string(m), ",", string(mu[m]), "\n")
    end
    for m = @fastmath convert(Int64, Constants.num_ords / 2 + 1):Constants.num_ords
        @inbounds write(output, "Ordinate ", string(m), ",", string(mu[m]), "\n")
    end
    for m = @fastmath 1:convert(Int64, Constants.num_ords / 2)
        @inbounds write(output, "Weight ", string(m), ",", string(weights[m]), "\n")
    end
    for m = @fastmath convert(Int64, Constants.num_ords / 2 + 1):Constants.num_ords
        @inbounds write(output, "Weight ", string(m), ",", string(weights[m]), "\n")
    end
    for m = @fastmath 1:convert(Int64, Constants.num_ords / 2)
        @inbounds write(output, "Total Leakage Ordinate ", string(m), ",", string(mean(ord_leak_l_stat[m])), "\n")
    end
    for m = @fastmath convert(Int64, Constants.num_ords / 2 + 1):Constants.num_ords
        @inbounds write(output, "Total Leakage Ordinate ", string(m), ",", string(mean(ord_leak_r_stat[m])), "\n")
    end
    close(output)

    return nothing
end

main()
