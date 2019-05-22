#!/usr/bin/env julia

# FluxProfile.jl

include("Constants.jl")

using .Constants

using GaussQuadrature

set_zero_subnormals(true)

@inline function linspace(x_start::Float64, x_fin::Float64, x_len::Int64)::Vector{Float64}
    local delta_x::Float64 = @fastmath (x_fin - x_start) / (x_len - 1)
    return @fastmath [(x_start + (i - 1) * delta_x) for i in 1:x_len]
end

function main()::Nothing
    # Assignment of material variables
    local macro_tot::Vector{Float64} = Constants.tot_const
    local macro_scat::Vector{Float64} = Constants.scat_const

    # Assignment of initial calculation variables
    local phi_new_outer::Vector{Float64} = fill(1.0, Constants.num_cells)  # 1/cm^2-s-MeV, assume scalar flux is init. const.
    local phi_old_outer::Vector{Float64} = fill(0.0, Constants.num_cells)  # 1/cm^2-s-MeV
    local phi_new_inner::Array{Float64, 2} = fill(1.0, Constants.num_cells, Constants.num_materials)
    local phi_old_inner::Array{Float64, 2} = fill(0.0, Constants.num_cells, Constants.num_materials)
    # Initial source terms
    local scat_source::Vector{Float64} = fill(0.0, Constants.num_cells)
    local spont_source::Vector{Float64} = fill(0.0, Constants.num_cells)
    local tot_source::Vector{Float64} = fill(0.0, Constants.num_cells)
    local total_chord::Float64 = @fastmath sum(Constants.chord)  # cm
    local prob::Vector{Float64} = @fastmath @. Constants.chord / total_chord

    # Calculational arrays
    local psi::Array{Float64, 3} = fill(1.0, Constants.num_cells, Constants.num_ords, Constants.num_materials)
    local psi_i_p::Array{Float64, 3} = fill(0.0, Constants.num_cells, Constants.num_ords, Constants.num_materials)
    local psi_i_m::Array{Float64, 3} = fill(0.0, Constants.num_cells, Constants.num_ords, Constants.num_materials)

    # Leakage values
    local leakage_l::Vector{Float64} = fill(0.0, Constants.num_materials)
    local leakage_r::Vector{Float64} = fill(0.0, Constants.num_materials)
    local ord_leak_l::Vector{Float64} = fill(0.0, Constants.num_ords)
    local ord_leak_r::Vector{Float64} = fill(0.0, Constants.num_ords)

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

    local alpha::Float64 = 1.0

    # Initial material number
    local material_num::Int32 = 1
    local material_off_num::Int32 = 2

    # Calculation: iterations
    local cont_calc_outer::Bool = true
    # Start counter at zero
    local iterations_outer::Int64 = 0

    # Outer loop over overall mixed system
    while (cont_calc_outer)
        phi_old_outer = deepcopy(phi_new_outer)  # 1/cm^2-s-MeV

        # Switch value of two variables
        (material_num, material_off_num) = (material_off_num, material_num)

        local cont_calc_inner::Bool = true
        # Start inner iterations at zero
        local iterations_inner::Int64 = 0

        # Phi calculations
        phi_new_inner = fill(1.0, Constants.num_cells, Constants.num_materials)
        phi_old_inner = fill(0.0, Constants.num_cells, Constants.num_materials)

        local psi_overall::Array{Float64, 2} = fill(0.0, Constants.num_cells, Constants.num_ords)

        # Flag to tell whether to use the inner loop data in averaging or not
        local converged::Bool = false
        # Inner loop over generated geometry
        while (cont_calc_inner)
            phi_old_inner = deepcopy(phi_new_inner)

            # Determine sources for each cell and group
            @simd for c = 1:Constants.num_cells
                @inbounds scat_source[c] = @fastmath macro_scat[material_num] / 2.0 * phi_new_inner[c, material_num]
                @inbounds spont_source[c] = @fastmath Constants.spont_source_const[material_num] / 2.0
                @inbounds tot_source[c] = scat_source[c] + spont_source[c]
            end

            # Forward sweep (left to right)
            # First cell (left boundary)
            # Ordinate loop, only consider the pos. ords for forward motion
            @simd for m = @fastmath convert(Int64, Constants.num_ords / 2 + 1):Constants.num_ords
                @inbounds psi[1, m, material_num] = @fastmath (1.0 + (macro_tot[material_num] * Constants.struct_thickness) / (2.0 * abs(mu[m])) + alpha * Constants.struct_thickness / (2.0 * Constants.chord[material_num]))^(-1) * (psi_bound_l[m] + (tot_source[1] * Constants.struct_thickness) / (2.0 * abs(mu[m])) + (Constants.struct_thickness * alpha * prob[material_off_num] * psi[1, m, material_off_num]) / (2.0 * prob[material_num] * Constants.chord[material_off_num]))
                @inbounds psi_i_p[1, m, material_num] = @fastmath 2.0 * psi[1, m, material_num] - psi_bound_l[m]
            end
            # Rest of the cells (sans left bounding cell)
            for c = 2:Constants.num_cells
                @simd for m = @fastmath convert(Int64, Constants.num_ords / 2 + 1):Constants.num_ords
                    # Continuity of boundaries
                    @inbounds psi_i_m[c, m, material_num] = @fastmath psi_i_p[c - 1, m, material_num]
                    @inbounds psi[c, m, material_num] = @fastmath (1.0 + (macro_tot[material_num] * Constants.struct_thickness) / (2.0 * abs(mu[m])) + alpha * Constants.struct_thickness / (2.0 * Constants.chord[material_num]))^(-1) * (psi_i_m[c, m, material_num] + (tot_source[c] * Constants.struct_thickness) / (2.0 * abs(mu[m])) + (Constants.struct_thickness * alpha * prob[material_off_num] * psi[c, m, material_off_num]) / (2.0 * prob[material_num] * Constants.chord[material_off_num]))
                    @inbounds psi_i_p[c, m, material_num] = @fastmath 2.0 * psi[c, m, material_num] - psi_i_m[c, m, material_num]
                end
            end

            # Backward sweep (right to left)
            # First cell (right boundary)
            # Ordinate loop, only consider neg. ords for backwards motion
            @simd for m = @fastmath 1:convert(Int64, Constants.num_ords / 2)
                @inbounds psi[Constants.num_cells, m, material_num] = @fastmath (1.0 + (macro_tot[material_num] * Constants.struct_thickness) / (2.0 * abs(mu[m])) + alpha * Constants.struct_thickness / (2.0 * Constants.chord[material_num]))^(-1) * (psi_bound_r[m] + (tot_source[Constants.num_cells] * Constants.struct_thickness) / (2.0 * abs(mu[m])) + (Constants.struct_thickness * alpha * prob[material_off_num] * psi[Constants.num_cells, m, material_off_num]) / (2.0 * prob[material_num] * Constants.chord[material_off_num]))
                @inbounds psi_i_m[Constants.num_cells, m, material_num] = @fastmath 2.0 * psi[Constants.num_cells, m, material_num] - psi_bound_r[m]
            end
            # Rest of the cells (sans right bounding cell)
            for c = @fastmath (Constants.num_cells - 1):-1:1
                @simd for m = @fastmath 1:convert(Int64, Constants.num_ords / 2)
                    # Continuity of boundaries
                    @inbounds psi_i_p[c, m, material_num] = @fastmath psi_i_m[c + 1, m, material_num]
                    @inbounds psi[c, m, material_num] = @fastmath (1.0 + (macro_tot[material_num] * Constants.struct_thickness) / (2.0 * abs(mu[m])) + alpha * Constants.struct_thickness / (2.0 * Constants.chord[material_num]))^(-1) * (psi_i_p[c, m, material_num] + (tot_source[c] * Constants.struct_thickness) / (2.0 * abs(mu[m])) + (Constants.struct_thickness * alpha * prob[material_off_num] * psi[c, m, material_off_num]) / (2.0 * prob[material_num] * Constants.chord[material_off_num]))
                    @inbounds psi_i_m[c, m, material_num] = @fastmath 2.0 * psi[c, m, material_num] - psi_i_p[c, m, material_num]
                end
            end

            # Calculate phi from psi
            @simd for c = 1:Constants.num_cells
                @simd for k = 1:Constants.num_materials
                    local weighted_sum::Float64 = 0.0
                    @simd for m = 1:Constants.num_ords
                        @inbounds @fastmath weighted_sum += weights[m] * psi[c, m, k]
                    end
                    @inbounds phi_new_inner[c, k] = weighted_sum  # 1/cm^2-s-MeV
                end
            end

            # Relative error for inner loop
            @fastmath iterations_inner += 1
            local err_inner::Float64 = @inbounds @fastmath maximum(@. abs(phi_new_inner - phi_old_inner) / phi_new_inner)
            if @fastmath (err_inner <= Constants.inner_tolerance_closure)
                cont_calc_inner = false
                converged = true
            elseif @fastmath (iterations_inner > Constants.num_iter_inner_closure)
                cont_calc_inner = false
                println("No convergence on loop number ", iterations_outer, ": quit after maximum ", iterations_inner, " iterations; data will not be used")
            end
        end  # Inner loop

        if (converged)
            @fastmath iterations_outer += 1

            # Calculate the overall psi
            @simd for c = 1:Constants.num_cells
                @simd for m = 1:Constants.num_ords
                    @simd for k = 1:Constants.num_materials
                        @inbounds @fastmath psi_overall[c, m] += prob[k] * psi[c, m, k]
                    end
                end
            end

            # Calculate phi from the overall psi
            @simd for c = 1:Constants.num_cells
                local weighted_sum::Float64 = 0.0
                @simd for m = 1:Constants.num_ords
                    @inbounds @fastmath weighted_sum += weights[m] * psi_overall[c, m]
                end
                phi_new_outer[c] = weighted_sum  # 1/cm^2-s-MeV
            end

            if @fastmath (iterations_outer % Constants.num_say == 0)
                println("Realization number ", iterations_outer)
            end

            # Relative error for the outer loop
            @fastmath iterations_outer += 1
            local outer_err::Float64 = @inbounds @fastmath maximum(@. abs((phi_new_outer - phi_old_outer)) / phi_new_outer)
            if @fastmath (outer_err <= Constants.outer_tolerance_closure)
                println("Converged after ", iterations_outer, " outer iterations")
                cont_calc_outer = false
            elseif @fastmath (iterations_outer > Constants.num_iter_outer_closure)
                cont_calc_outer = false
                println("No convergence on outer loop; quit after ", iterations_outer, " iterations")
            end
        end  # Logical test for inner convergence: don't average nonconverged samples
    end  # Outer loop

    # Compute leakages
    for k = 1:Constants.num_materials
        # Leakage in neg. direction from left face
        for m = @fastmath 1:convert(Int64, Constants.num_ords / 2)
            ord_leak_l[m] += psi_i_m[1, m, k] * abs(mu[m]) * weights[m] * prob[k]
            leakage_l[k] += abs(mu[m]) * weights[m] * psi_i_m[1, m, k]
        end
        # Leakage in pos. direction from right face
        for m = @fastmath convert(Int64, Constants.num_ords / 2 + 1):Constants.num_ords
            ord_leak_r[m] += psi_i_p[Constants.num_cells, m, k] * abs(mu[m]) * weights[m] * prob[k]
            leakage_r[k] += abs(mu[m]) * weights[m] * prob[k] * psi_i_p[Constants.num_cells, m, k]
        end
    end

    # Save flux data
    local flux_total::IOStream = open("out/total_flux_lp.out", "w")
    local flux_1::IOStream = open("out/flux_1_lp.out", "w")
    local flux_2::IOStream = open("out/flux_2_lp.out", "w")
    local psi_1_m::IOStream = open("out/psi_1_m_lp.out", "w")
    local psi_1_p::IOStream = open("out/psi_1_p_lp.out", "w")
    local psi_2_m::IOStream = open("out/psi_2_m_lp.out", "w")
    local psi_2_p::IOStream = open("out/psi_2_p_lp.out", "w")
    for i = 1:Constants.num_cells
        write(flux_total, string(cell_vector[i]), ",", string(phi_new_outer[i]), "\n")
        write(flux_1, string(cell_vector[i]), ",", string(phi_new_inner[i, 1]), "\n")
        write(flux_2, string(cell_vector[i]), ",", string(phi_new_inner[i, 2]), "\n")
        write(psi_1_m, string(cell_vector[i]), ",", string(psi[i, 1, 1]), "\n")
        write(psi_1_p, string(cell_vector[i]), ",", string(psi[i, 2, 1]), "\n")
        write(psi_2_m, string(cell_vector[i]), ",", string(psi[i, 1, 2]), "\n")
        write(psi_2_p, string(cell_vector[i]), ",", string(psi[i, 2, 2]), "\n")
    end
    close(flux_total)
    close(flux_1)
    close(flux_2)
    close(psi_1_m)
    close(psi_1_p)
    close(psi_2_m)
    close(psi_2_p)

    # Save leakage data
    local output::IOStream = open("out/leakage_lp.txt", "w")
    write(output, "Total Leakage Left,", string(leakage_l[1] + leakage_l[2]), "\n")
    write(output, "Total Leakage Right,", string(leakage_r[1] + leakage_r[2]), "\n")
    for k = 1:Constants.num_materials
        write(output, "Material ", string(k), " Leakage Left,", string(leakage_l[k]), "\n")
        write(output, "Material ", string(k), " Leakage Right,", string(leakage_r[k]), "\n")
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
        @inbounds write(output, "Total Leakage Ordinate ", string(m), ",", string(ord_leak_l[m]), "\n")
    end
    for m = @fastmath convert(Int64, Constants.num_ords / 2 + 1):Constants.num_ords
        @inbounds write(output, "Total Leakage Ordinate ", string(m), ",", string(ord_leak_r[m]), "\n")
    end
    close(output)

    return nothing
end

main()
