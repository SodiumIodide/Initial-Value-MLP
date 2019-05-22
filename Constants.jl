module Constants
    set_zero_subnormals(true)

    # Constant parameters
    global const num_iter_outer = convert(Int64, 5e5)
    global const num_iter_inner = convert(Int64, 1e6)
    global const num_cells = convert(Int64, 5e2)
    global const num_ords = convert(Int64, 2)
    global const num_materials = convert(Int32, 2)
    global const num_geom_divs = convert(Int64, 1e2)
    global const inner_tolerance = convert(Float64, 1e-7)
    global const outer_tolerance = convert(Float64, 1e-9)
    global const leakage_tolerance = convert(Float64, 1e-5)
    global const inner_tolerance_closure = eps(Float64)
    global const outer_tolerance_closure = eps(Float64)
    global const num_iter_outer_closure = convert(Int64, 1e12)
    global const num_iter_inner_closure = convert(Int64, 1e12)

    global const seed = convert(Int64, 1234)

    global const incident_angular_flux = 1.0

    # Material properties
    global const thickness = 4.0  # cm
    global const struct_thickness = @fastmath thickness / convert(Float64, num_cells)
    global const tot_const = @fastmath vec([
        0.5  # 1/cm
        2.0  # 1/cm
    ])
    global const scat_ratio = 0.3
    global const scat_const = @fastmath @. tot_const * scat_ratio  # 1/cm
    global const chord = @fastmath vec([
        (thickness / 2.0)  # cm
        (thickness / 2.0) # cm
    ])
    global const spont_source_const = vec([
        0.0  # 1/cm^3
        0.0  # 1/cm^3
    ])

    global const num_say = convert(Int64, 1e3)
end
