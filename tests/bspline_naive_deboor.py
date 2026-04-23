from gtm.gtm_splines.bspline_prediction_vectorized import compute_multivariate_bspline_basis, Naive_Basis, bspline_prediction_vectorized
import torch
from gtm.gtm_splines.splines_utils import ReLULeR


# Comparing Naive Basis Approach and Deborr to ensure they return same values
if __name__ == "__main__":
    K=20
    degree=K
    
    spline_order = 3
    number_of_bound_knots_per_side = spline_order

    params_a = torch.randn([K+spline_order-1]).unsqueeze(1)

    spline_range = torch.FloatTensor([[ -4 ],[4]])

    distance_between_knots_in_bounds = (
                spline_range[1, 0] - spline_range[0, 0]
            ) / (degree - 1)


    knots = torch.linspace(
                spline_range[0, 0]
                - (number_of_bound_knots_per_side) * distance_between_knots_in_bounds,
                spline_range[1, 0]
                + (number_of_bound_knots_per_side) * distance_between_knots_in_bounds,
                degree + 2 * number_of_bound_knots_per_side,  # 2* because of two sides
                dtype=torch.float32,
            ).unsqueeze(1)


    span_factor = 0


    samples_linespace = torch.linspace(spline_range[0, 0]*1.4, spline_range[1, 0]*1.4, 10000)
    input_a  = samples_linespace.unsqueeze(1)


    pred = bspline_prediction_vectorized(params_a,
        input_a,
        knots,
        degree,
        spline_range,
        monotonically_increasing=False,
        derivativ=0,
        return_penalties=False,
        calc_method="deBoor", 
        span_factor=0.1,
        span_restriction="reluler",
        covariate=False,
        params_covariate=False,
        covariate_effect="multiplicativ",
        penalize_towards=0,
        order=3,
        varying_degrees=True,
        params_a_mask=None)
    
    
    clamper = ReLULeR(spline_range)
    samples_well_behaved = clamper.forward(input_a).squeeze(1)
    basis_matrix = Naive_Basis(samples_well_behaved, spline_range, K, span_factor, knots, derivativ=0, order=3)

    pred_basis = basis_matrix @ params_a
    
    
    print(torch.all(torch.abs(pred - pred_basis.squeeze()) < 1e-6))