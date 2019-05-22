# Inital Value MLP

The Modified Levermore-Pomraning (MLP) closure is effectively similar to the standard LP closure, but with the removal of the magnitude operation on the cosine coefficient, or effectively taking the real value of the quadrature ordinate rather than the absolute value. This has the unfortunate effect of providing negative values to the matrix representation, resulting in negative Eigenvalues, which can cause numerical instabilities in the solution of the system.

The assumption is that by solving an S_2 transport model for a finite width slab system with a boundary source, correct values for the reflection and known values for the incident flux may be determined. These values should be able to be used for an initial value system that is solveable via matrix exponentiation (e.g. a linear first-order ordinary differential system of equations).

As shown here, the matrix exponentiation form is tested and shown valid for a simple system of hyperbolic trigonometric terms, but when applied to the MLP problem, the result is unstable. This leads to the presumption that the MLP system, despite being defined for a known solution, is still an ill-posed system numerically.

Additionally, while the MLP model is known to have a deterministic solution for a semi-infinite slab (such that the transmission of radiation goes to zero), the matrix exponential method is still unstable in predicting a profile through such a system as well.
