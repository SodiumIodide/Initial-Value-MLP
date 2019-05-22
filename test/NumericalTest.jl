#!/usr/bin/env julia

# NumericalTest.jl

using LinearAlgebra
using DataFrames
using CSV

set_zero_subnormals(true)

@inline function linspace(x_start::Float64, x_fin::Float64, x_len::Int64)::Vector{Float64}
    local delta_x::Float64 = @fastmath (x_fin - x_start) / (x_len - 1)
    return @fastmath [(x_start + (i - 1) * delta_x) for i in 1:x_len]
end

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

    local x_values::Vector{Float64} = linspace(x_start, thickness, num_cells)
    local y_values::Vector{Float64} = fill(0.0, num_cells)
    local u_values::Vector{Float64} = fill(0.0, num_cells)

    # Initial value
    @inbounds y_values[1] = func_y(x_values[1])
    #@inbounds u_values[1] = func_u(x_values[1])
    @inbounds u_values[1] = -0.526387850916331  # Determined from central difference formula

    #@show u_values[1]
    #@show -0.526387850916331

    local multiplier_matrix::Array{Float64, 2} = @fastmath [
        0.0 omega^2
        1.0 0.0
    ]

    local y_benchmark::Vector{Float64} = @. func_y(x_values)
    local u_benchmark::Vector{Float64} = @. func_u(x_values)

    for i in 2:num_cells
        local curr_x::Float64 = @inbounds x_values[i]

        # Solve the matrix exponential
        local results_vector::Vector{Float64} = @inbounds @fastmath exp(multiplier_matrix * curr_x) * vec([u_values[1] y_values[1]])

        @inbounds u_values[i] = results_vector[1]
        @inbounds y_values[i] = results_vector[2]
    end

    local tabular::DataFrame = DataFrame(xvals=x_values, yvals=y_values, uvals=u_values, ybench=y_benchmark, ubench=u_benchmark)

    CSV.write("test_out.csv", tabular)

    return nothing
end

main()
