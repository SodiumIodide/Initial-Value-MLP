#!/usr/bin/env julia

# InitialValue.jl

using LinearAlgebra

set_zero_subnormals(true)

function main()::Nothing
    local num_cells::Int64 = 100
    local x_start::Float64 = 0.0
    local thickness::Float64 = 2.0
    local omega::Float64 = 0.2

    local function func_y(x_value::Float64)::Float64
        return @fastmath cosh(omega * x_value) - sinh(omega * x_value) / tanh(omega * thickness)
    end

    local function func_u(x_value::Float64)::Float64
        return @fastmath omega * sinh(omega * x_value) - omega * cosh(omega * x_value) / tanh(omega * thickness)
    end

    local y_values::Vector{Float64} = fill(0.0, num_cells)
    local u_values::Vector{Float64} = fill(0.0, num_cells)

    @inbounds y_values[1] = 1.0

    local delta_x::Float64 = @fastmath (thickness - x_start) / (num_cells)
    local soln_con::Float64 = @fastmath - (delta_x * omega)^2 - 2.0

    local y_mult_matrix::Array{Float64, 2} = @fastmath zeros(num_cells - 1, num_cells - 1)
    @inbounds y_mult_matrix[1, 1] = soln_con
    @inbounds y_mult_matrix[1, 2] = 1.0
    @simd for i = 2:@fastmath(num_cells - 2)
        @inbounds @fastmath y_mult_matrix[i, i - 1] = 1.0
        @inbounds y_mult_matrix[i, i] = soln_con
        @inbounds @fastmath y_mult_matrix[i, i + 1] = 1.0
    end
    @inbounds @fastmath y_mult_matrix[num_cells - 1, num_cells - 2] = 1.0
    @inbounds @fastmath y_mult_matrix[num_cells - 1, num_cells - 1] = soln_con

    local y_res_vector::Vector{Float64} = @fastmath fill(0.0, num_cells - 1)
    @inbounds y_res_vector[1] = -1.0

    @inbounds y_values[2:end] = @fastmath y_mult_matrix \ y_res_vector

    local u_mult_matrix::Array{Float64, 2} = @fastmath fill(0.0, num_cells - 1, num_cells - 1)
    @inbounds u_mult_matrix[1, 2] = 1.0
    @simd for i = 2:@fastmath(num_cells - 2)
        @inbounds @fastmath u_mult_matrix[i, i - 1] = -1.0
        @inbounds @fastmath u_mult_matrix[i, i + 1] = 1.0
    end
    @inbounds @fastmath u_mult_matrix[num_cells - 1, num_cells - 2] = 1.0

    @inbounds u_values[2:end] = @fastmath u_mult_matrix * y_values[2:end]
    @inbounds @fastmath u_values[2] -= 1.0
    @fastmath u_values ./= (2.0 * delta_x)

    @inbounds @fastmath u_values[1] = u_values[3] - 2.0 * delta_x * omega^2 * y_values[2]

    @show u_values[1]

    return nothing
end

main()
