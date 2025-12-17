from typing import List, Literal, Tuple, Optional # Still needed

import itertools
import optuna
import scipy
import torch
from optuna.samplers import TPESampler
from optuna.study import Study
from torch import nn, Tensor, FloatTensor
from torch.utils.data import DataLoader
from torch.nn.utils.convert_parameters import vector_to_parameters

from gtm.gtm_layers.decorrelation_layer import Decorrelation
from gtm.gtm_layers.layer_utils import generate_diagonal_matrix
# from gtm.layers.flip import Flip
from gtm.gtm_layers.transformation_layer import *
from gtm.gtm_plots_analysis.compute_conditional_independence_kld import *
from gtm.gtm_plots_analysis.plot_conditional_dependence_pair import \
    plot_conditional_dependence_pair
from gtm.gtm_plots_analysis.plot_conditional_independence_graph import \
    plot_graph_conditional_independencies
from gtm.gtm_plots_analysis.plot_conditional_independence_graphs_pairplots import \
    plot_graph_conditional_independencies_with_pairplots
from gtm.gtm_plots_analysis.plot_densities import plot_densities
from gtm.gtm_plots_analysis.plot_marginals import plot_marginals
from gtm.gtm_plots_analysis.plot_metric_hist import plot_metric_hist
from gtm.gtm_plots_analysis.plot_metric_scatter import plot_metric_scatter
from gtm.gtm_plots_analysis.plot_splines import plot_splines
from gtm.gtm_training.objective_functions import log_likelihood, training_objective
from gtm.gtm_training.training_helpers import (if_float_create_lambda_penalisation_matrix, train_freq)
from gtm.gtm_training.training_bayes.training_helpers import train_bayes
from gtm.gtm_training.training_bayes.objective_functions import bayesian_training_objective

# from gtm.simulation_study.simulation_study_helpers import plot_marginals, plot_densities

class GTM(nn.Module):
    """
    Graphical Transformation Model (GTM).

    GTMs are highly flexible density estimation models. The combine marginal transformations, named the transfomation layer,
    with a sequence normalizing flow like layers, named decorrelation layers.
    The main advantage of GTMs is that they combine high flexibility in learning complex marginals and dependency structures with
    interpretability of the conditional dependency structure between the varaibles.
    The interpretability stems from the GTMs decorrelation layers that can be interpreted as creating a pseudo correlation matrix.
    This means that at each point in the pace of the data, the GTM returns a local pseudo correlation matrix and hence local pseudo conditional correlations.

    Parameters
    ----------
    number_variables : int
        The number of variables in the data.
    number_transformation_layers : Literal[0, 1] = 1, optional
        The number of transformation layers. Either 0 or 1. Default is 1.
    number_decorrelation_layers : int, optional
        The number of decorrelation layers. Default is 3. a Minimum of 3 is recommended to ensure that variable ordering does not influence the results.
    degree_transformations : int | list[int] = 15, optional
        The degree of the transformation layer splines. One can either pass an integer which is then used as the degree for all variables or a
        list of integers as long as the number of variables, which then results in varying degrees across variables.
        Varying degrees make sense if different dimensions have marginals that vary strongly in their deviation from a standard gaussian distribution.
        Default is 15.
    degree_decorrelation : int = 30, optional
        The degree of the decorrelation layer splines. The degree is the same for every spline across every decorrelation layer. Default is 30.
    spline_transformation : Literal["bspline", "bernstein"] = "bspline", optional
        The type of spline used in the transformation layer. Either "bspline" or "bernstein". Default is "bspline".
        Bsplines tend to be more flexible and due to the implementation via de Boor's algorithm they do increase compute time to much when increasing the degree.
        Bernstein polynomials on the other hand are faster a smaller degrees and require less penalization for smoothness at lower degrees.
    spline_decorrelation : Literal["bspline", "bernstein"] = "bspline", optional
        The type of spline used in the decorrelation layer. Either "bspline" or "bernstein". Default is "bspline".
        Bsplines tend to be more flexible and due to the implementation via de Boor's algorithm they do increase compute time to much when increasing the degree.
        Bernstein polynomials on the other hand are faster a smaller degrees and require less penalization for smoothness at lower degrees.
        The decorrelation layers require most of the computational, hence for high dimensional data and not to complex dependencies Bernstein polynomials can be more efficient.
    transformation_spline_range : List[float, float] = [-15, 15], optional
        The range of the transformation splines, hence the minimum and maximum value across all variables.
        This is the range in which the transformation layer is defined.
        Therefor it is recommended to scale the data to a common range before training the model.
        A further recommendation is to set the spline range 10-20% larger than the range of the data to allow for better out of sample prediction in validation and test sets.
        Outside of the range the transformation layer uses a span restriction which results the border case prediction.
        The same is true for the decorrelation layer.
        Default spline range is [-15, 15].
    device : str | torch.device = "cpu", optional
        The device on which the model is trained. Default is "cpu". Can also be a torch.device object such as a torch.device("cuda:0") for GPU training.
    -----------
    """
    def __init__(
        self,
        number_variables: int,
        number_transformation_layers: Literal[0, 1] = 1,
        number_decorrelation_layers: int = 3,
        degree_transformations: int | list[int] = 15,
        degree_decorrelation: int = 30,
        spline_transformation: Literal["bspline", "bernstein"] = "bspline",
        spline_decorrelation: Literal["bspline", "bernstein"] = "bspline",
        transformation_spline_range: Tuple[float, float] = (-15, 15),
        inference: Literal['frequentist', 'bayesian'] = 'frequentist',
        hyperparameter: dict[str, dict[str, float]] | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Parameters
        ----------
        number_variables : int
            The number of variables in the data.
        number_transformation_layers : Literal[0, 1] = 1, optional
            The number of transformation layers. Either 0 or 1. Default is 1.
        number_decorrelation_layers : int, optional
            The number of decorrelation layers. Default is 3. a Minimum of 3 is recommended to ensure that variable ordering does not influence the results.
        degree_transformations : int | list[int] = 15, optional
            The degree of the transformation layer splines. One can either pass an integer which is then used as the degree for all variables or a
            list of integers as long as the number of variables, which then results in varying degrees across variables.
            Varying degrees make sense if different dimensions have marginals that vary strongly in their deviation from a standard gaussian distribution.
            Default is 15.
        degree_decorrelation : int = 30, optional
            The degree of the decorrelation layer splines. The degree is the same for every spline across every decorrelation layer. Default is 30.
        spline_transformation : Literal["bspline", "bernstein"] = "bspline", optional
            The type of spline used in the transformation layer. Either "bspline" or "bernstein". Default is "bspline".
            Bsplines tend to be more flexible and due to the implementation via de Boor's algorithm they do increase compute time to much when increasing the degree.
            Bernstein polynomials on the other hand are faster a smaller degrees and require less penalization for smoothness at lower degrees.
        spline_decorrelation : Literal["bspline", "bernstein"] = "bspline", optional
            The type of spline used in the decorrelation layer. Either "bspline" or "bernstein". Default is "bspline".
            Bsplines tend to be more flexible and due to the implementation via de Boor's algorithm they do increase compute time to much when increasing the degree.
            Bernstein polynomials on the other hand are faster a smaller degrees and require less penalization for smoothness at lower degrees.
            The decorrelation layers require most of the computational, hence for high dimensional data and not to complex dependencies Bernstein polynomials can be more efficient.
        transformation_spline_range : List[float, float] = [-15, 15], optional
            The range of the transformation splines, hence the minimum and maximum value across all variables.
            This is the range in which the transformation layer is defined.
            Therefor it is recommended to scale the data to a common range before training the model.
            A further recommendation is to set the spline range 10-20% larger than the range of the data to allow for better out of sample prediction in validation and test sets.
            Outside of the range the transformation layer uses a span restriction which results the border case prediction.
            The same is true for the decorrelation layer.
            Default spline range is [-15, 15].
        device : str | torch.device = "cpu", optional
            The device on which the model is trained. Default is "cpu". Can also be a torch.device object such as a torch.device("cuda:0") for GPU training.
        -----------
        """
        super(GTM, self).__init__()

        # Parameters below are left out of init for the initial package release
        decorrelation_spline_range: List[List[float]] = [[-15], [15]]
        span_factor: torch.Tensor = torch.tensor(0.1)
        span_restriction : Literal['reluler'] = "reluler"
        number_covariates: bool | int = False  # False means no covariates, or its count #ALLWAYS FALSE FOR THIS VERSION, Till the number of cov is finally implemented.
        initial_log_transform: bool = False
        covariate_effect: str = "multiplicativ"
        calc_method_bspline: Literal['deBoor'] = "deBoor"
        affine_decorr_layer: bool = False
        degree_multi: bool | int = False
        spline_order: int = 3
        transform_only: bool = False

        #Loading Values in the model
        self.transform_only: Literal[False] = transform_only

        self.number_variables: int = number_variables
        self.inference: Literal['frequentist'] | Literal['bayesian'] = inference

        # Repeat polynomial ranges for all variables as this is the range for the bsplines essentially
        self.transformation_spline_range: List[List[float]] = [
            [transformation_spline_range[0]] * self.number_variables,
            [transformation_spline_range[1]] * self.number_variables
        ]
            
        self.decorrelation_spline_range: List[List[float]] = [
            decorrelation_spline_range[0] * self.number_variables,
            decorrelation_spline_range[1] * self.number_variables
        ]
        
        # required for the varying degree of the transformation layer to work
        # if it is a number then transform into a repeating list of length of number of varaibles
        if isinstance(degree_transformations, int):
            degree_transformations = [degree_transformations] * self.number_variables
        self.degree_transformations: List[int] = degree_transformations
        self.degree_decorrelation: int = degree_decorrelation
        
        self.spline_transformation: Literal['bspline'] | Literal['bernstein'] = spline_transformation
        self.spline_decorrelation: Literal['bspline'] | Literal['bernstein'] = spline_decorrelation
        
        self.span_factor: Tensor = span_factor
        self.span_restriction = span_restriction

        self.device: str | torch.device = device

        self.number_covariates: Literal[False] = number_covariates

        self.num_trans_layers: Literal[0] | Literal[1] = number_transformation_layers

        self.initial_log_transform: Literal[False] = initial_log_transform

        self.covariate_effect: str = covariate_effect

        self.calc_method_bspline = calc_method_bspline

        self.spline_order = spline_order

        self.affine_decorr_layer: Literal[False] = affine_decorr_layer

        self.degree_multi: Literal[False] = degree_multi
        
        if self.inference == "bayesian":
            
            _COMMON_BAYES: dict[str, float] = {
                "sigma_a": 1,
                "sigma_b": 5e-5,
                "RW1": {
                    "tau_a": 1,
                    "tau_b": 5e-4
                    },
                "RW2": {
                    "tau_a": 1,
                    "tau_b": 5e-4
                    }
                }
            
            _DEFAULT_BAYES_HYPERPARAMS: dict[str, dict[str, float]] = {
                "transformation": _COMMON_BAYES.copy(),
                "decorrelation": _COMMON_BAYES.copy(),
            }
            
            if "bernstein" in {self.spline_transformation, self.spline_decorrelation}:
                raise NotImplementedError(
                    "Bayesian inference with Bernstein polynomials is not yet implemented."
                )
                
            self.hyperparameter: dict[str, dict[str, float]] = (
                _DEFAULT_BAYES_HYPERPARAMS if hyperparameter is None else hyperparameter
            )
        else:
            self.hyperparameter = {}
            

        if self.num_trans_layers > 0:
            self.transformation = Transformation(
                degree=self.degree_transformations,
                number_variables=self.number_variables,
                spline_range=self.transformation_spline_range,
                hyperparameters= self.hyperparameter.get('transformation', {}),
                span_factor=self.span_factor,
                number_covariates=self.number_covariates,
                spline=self.spline_transformation,
                initial_log_transform=self.initial_log_transform,
                calc_method_bspline=self.calc_method_bspline,
                span_restriction=self.span_restriction,
                spline_order=self.spline_order,
                inference=self.inference, 
                device=device,
            )

        if self.num_trans_layers > 1:
            raise NotImplementedError(
                "Model is only implemented to have 0 or 1 transformation layer. This is enough as a TM with enough degrees can model any arbitrary continious distribution."
            )

        self.flip_matrix: torch.Tensor = generate_diagonal_matrix(self.number_variables).to(device=self.device)

        self.number_decorrelation_layers = number_decorrelation_layers
        if self.number_decorrelation_layers > 0:
            self.decorrelation_layers = nn.ModuleList(
                modules=[
                    Decorrelation(
                        degree=self.degree_decorrelation,
                        number_variables=self.number_variables,
                        spline_range=self.decorrelation_spline_range,
                        hyperparameter=self.hyperparameter.get('decorrelation',  {}),
                        span_factor=self.span_factor,
                        span_restriction=self.span_restriction,
                        spline=self.spline_decorrelation,
                        number_covariates=self.number_covariates,
                        covariate_effect=self.covariate_effect,
                        calc_method_bspline=self.calc_method_bspline,
                        spline_order=self.spline_order,
                        affine_layer=self.affine_decorr_layer,
                        degree_multi=self.degree_multi,
                        inference=self.inference,
                        device=device,
                    )
                    for _ in range(self.number_decorrelation_layers)
                ]
            )

        self.subset_dimension = None
        self.conditional_independence_table = None

    def to(self, device: str | torch.device):
        """
        Pushes GTM and its layers to the reuired device.

        Parameters
        ----------
        device : str | torch.device
            The device on which the model is trained. Default is "cpu". Can also be a torch.device object such as a torch.device("cuda:0") for GPU training.

        Returns
        -------
        GTM pushed to device
        """
        self.device = device
        self.transformation.device = device
        if not self.transform_only:
            for decorrelation_layer in self.decorrelation_layers:
                decorrelation_layer.device = device

        return super().to(device=device)

    def __create_return_dict_nf_mctm__(self, input: Tensor) -> dict[str, Tensor|float|None]:
        
        return {
            "output": (input.clone() if input.dim() > 1 else input.clone().unsqueeze(1)).to(self.device),
            "log_d": torch.zeros(
                input.size() if input.dim() > 1 else input.unsqueeze(1).size()
            ).to(self.device),
            "transformation_second_order_ridge_pen_global": 0,
            "second_order_ridge_pen_global": 0,
            "first_order_ridge_pen_global": 0,
            "param_ridge_pen_global": 0,
            "lambda_matrix_global": torch.eye(self.number_variables).to(self.device),
            "der_lambda_matrix_global": None,
            "der2_lambda_matrix_global": None,
        }
        
        

    def bayesian_param_set(
        self,
        output: dict,
        which: str = "mean",          # 'mean' | 'sample' | 'vector'
        sample_seed: int | None = None,
        update_buffers: bool = True,  # False if you do NOT want BN running stats etc. updated
    ):
        """
        Overwrite this GTM's parameters with a θ coming from VI.

        - which='mean'   -> θ = μ   (usual point export)
        - which='sample' -> θ ~ q(θ)
        - which='vector' -> use the flat vector from output['mu'] (must match VI schema)
        """
        vi = output.get("vi_model", None)
        if vi is None:
            raise ValueError("Expected output to contain 'vi_model' returned by fit_bayes().")

        # choose θ
        with torch.no_grad():
            if which == "mean":
                theta = vi.mu.detach()
            elif which == "sample":
                if sample_seed is not None:
                    torch.manual_seed(sample_seed)
                theta = vi.sample_theta(1).squeeze(0).detach()
            elif which == "vector":
                if "mu" not in output:
                    raise ValueError("which='vector' requires output['mu'].")
                theta = output["mu"].detach()
                if theta.numel() != vi.mu.numel():
                    raise ValueError(f"Vector length {theta.numel()} != VI dim {vi.mu.numel()}.")
            else:
                raise ValueError("which must be 'mean', 'sample', or 'vector'.")

            # map flat θ -> {key: tensor} using the VI schema (matches how μ was built)
            sd_new = vi._theta_to_state_dict(theta)

            # optionally restrict to *parameters only* (skip buffers like BN running stats)
            if not update_buffers:
                param_keys = {k for k, _ in self.named_parameters()}
                sd_new = {k: v for k, v in sd_new.items() if k in param_keys}

            # merge into current state_dict, respecting dtype/device
            full = self.state_dict()
            device = next(self.parameters()).device
            for k, v in sd_new.items():
                if k in full:
                    sd_new[k] = v.to(device=device, dtype=full[k].dtype)
                else:
                    # key not present (e.g., schema drift) -> skip rather than crash
                    sd_new.pop(k)

            full.update(sd_new)
            self.load_state_dict(full, strict=False)


    def forward(self, y, return_lambda_matrix=True):
        """
        GTM forward pass.

        Parameters
        ----------
        y : torch.FloatTensor
            The input data to pass through the model foward pass.
        return_lambda_matrix: bool = True
            Wether to compute the global lambda matrix in the forward pass or not. To not do it save a bit of compute.

        Returns
        -------
        A dictionary containing the latent space named "output", the log determinant "log_d", the differences for thes spline penalites
        "transformation_second_order_ridge_pen_global", "second_order_ridge_pen_global", "second_order_ridge_pen_global", "param_ridge_pen_global"
        and the full model, hence global, lambda matrix "lambda_matrix_global".
        """
        # Some left out arguements for the first release version
        # evaluate and train do not make sense anymore as we do not store basis in transformation layer
        # a main reason is that we now allow training bathwise which prevents this
        covariate = False
        return_scores_hessian = False
        train = True
        evaluate = True
        
        #y=y.to(self.device)
        return_dict_nf_mctm = self.__create_return_dict_nf_mctm__(y)

        if self.subset_dimension is not None:
            # if subset dimension is set then only use this dimension
            y = y[:, self.subset_dimension].unsqueeze(1)

        if self.initial_log_transform == True:
            y = y + 0.01  # log(0) does not work
            log_d = -torch.log(y)  # = log(1/y)
            y = torch.log(y)
        else:
            log_d = 0

        # Training or evaluation
        if train or evaluate:

            if train:
                if self.num_trans_layers > 0:
                    # new input false to not recompute basis each iteration
                    return_dict_transformation = self.transformation(
                        y,
                        covariate,
                        log_d=log_d,
                        return_log_d=True,
                        new_input=False,
                        store_basis=True,
                        return_scores_hessian=return_scores_hessian,
                    )

                    return_dict_nf_mctm["output"] = return_dict_transformation["output"]
                    return_dict_nf_mctm["log_d"] = return_dict_transformation["log_d"]
                else:
                    return_dict_nf_mctm["output"] = y.clone()
                    return_dict_nf_mctm["log_d"] = torch.zeros(
                        y.size(), device=self.device
                    ).float()

            elif evaluate:
                if self.num_trans_layers > 0:
                    # new input true as we need to recompute the basis for the validation/test set
                    return_dict_transformation = self.transformation(
                        y,
                        covariate,
                        log_d=log_d,
                        return_log_d=True,
                        new_input=True,
                        store_basis=True,
                        return_scores_hessian=return_scores_hessian,
                    )

                    return_dict_nf_mctm["output"] = return_dict_transformation["output"]
                    return_dict_nf_mctm["log_d"] = return_dict_transformation["log_d"]
                else:
                    return_dict_nf_mctm["output"] = y.clone()
                    return_dict_nf_mctm["log_d"] = torch.zeros(
                        y.size(), device=self.device
                    ).float()

            if self.transform_only == True:
                return return_dict_nf_mctm

            if self.num_trans_layers > 0 and return_scores_hessian == True:

                return_dict_nf_mctm["der_lambda_matrix_global"] = (
                    return_dict_transformation["scores"]
                )  # .unsqueeze(2)

                return_dict_nf_mctm["der2_lambda_matrix_global_list"] = [
                    return_dict_transformation["hessian"]
                ]  # .unsqueeze(2)]

            if self.number_decorrelation_layers > 0:
                for i in range(self.number_decorrelation_layers):

                    if ((i + 1) % 2) == 0:
                        # even: 2,4, 6, ...
                        return_dict_nf_mctm["output"] = (
                            self.flip_matrix @ return_dict_nf_mctm["output"].T
                        ).T
                    # else:
                    # odd: 1, 3, 5, ...

                    return_dict_decorrelation = self.decorrelation_layers[i](
                        return_dict_nf_mctm["output"],
                        covariate,
                        0,
                        return_log_d=True,
                        return_penalties=True,
                        return_scores_hessian=return_scores_hessian,
                    )

                    return_dict_nf_mctm["output"] = return_dict_decorrelation["output"]
                    return_dict_nf_mctm["log_d"] += return_dict_decorrelation[
                        "log_d"
                    ]  # required if the layers are multiplicative
                    return_dict_nf_mctm[
                        "second_order_ridge_pen_global"
                    ] += return_dict_decorrelation["second_order_ridge_pen_sum"]
                    return_dict_nf_mctm[
                        "first_order_ridge_pen_global"
                    ] += return_dict_decorrelation["first_order_ridge_pen_sum"]
                    return_dict_nf_mctm[
                        "param_ridge_pen_global"
                    ] += return_dict_decorrelation["param_ridge_pen_sum"]

                    if ((i + 1) % 2) == 0:
                        # even
                        if return_lambda_matrix == True:
                            lambda_matrix_upper = (
                                self.flip_matrix
                                @ return_dict_decorrelation["lambda_matrix"]
                                @ self.flip_matrix
                            )

                        if return_lambda_matrix == True:
                            return_dict_nf_mctm["lambda_matrix_global"] = (
                                lambda_matrix_upper
                                @ return_dict_nf_mctm["lambda_matrix_global"]
                            )

                        if return_scores_hessian == True:
                            der_lambda_matrix_upper = (
                                self.flip_matrix
                                @ return_dict_decorrelation["der_lambda_matrix"]
                                @ self.flip_matrix
                            )
                            return_dict_nf_mctm["der_lambda_matrix_global"] = torch.bmm(
                                der_lambda_matrix_upper,
                                return_dict_nf_mctm["der_lambda_matrix_global"],
                            )

                            der2_lambda_matrix_upper = (
                                self.flip_matrix
                                @ return_dict_decorrelation["der2_lambda_matrix"]
                                @ self.flip_matrix
                            )

                            return_dict_nf_mctm[
                                "der2_lambda_matrix_global_list"
                            ].append(
                                torch.bmm(
                                    der2_lambda_matrix_upper,
                                    return_dict_nf_mctm["der_lambda_matrix_global"],
                                )
                                * return_dict_nf_mctm["der_lambda_matrix_global"]
                            )

                            for j in range(i + 1):  # j are all sum elements prior to i
                                return_dict_nf_mctm["der2_lambda_matrix_global_list"][
                                    j
                                ] = torch.bmm(
                                    der_lambda_matrix_upper,
                                    return_dict_nf_mctm[
                                        "der2_lambda_matrix_global_list"
                                    ][j],
                                )

                    else:
                        # odd
                        if return_lambda_matrix == True:
                            return_dict_nf_mctm["lambda_matrix_global"] = (
                                return_dict_decorrelation["lambda_matrix"]
                                @ return_dict_nf_mctm["lambda_matrix_global"]
                            )

                        if return_scores_hessian == True:
                            return_dict_nf_mctm["der_lambda_matrix_global"] = torch.bmm(
                                return_dict_decorrelation["der_lambda_matrix"],
                                return_dict_nf_mctm["der_lambda_matrix_global"],
                            )

                            return_dict_nf_mctm[
                                "der2_lambda_matrix_global_list"
                            ].append(
                                torch.bmm(
                                    return_dict_decorrelation["der2_lambda_matrix"],
                                    return_dict_nf_mctm["der_lambda_matrix_global"],
                                )
                                * return_dict_nf_mctm["der_lambda_matrix_global"]
                            )

                            for j in range(i + 1):  # j are all sum elements prior to i
                                return_dict_nf_mctm["der2_lambda_matrix_global_list"][
                                    j
                                ] = torch.bmm(
                                    return_dict_decorrelation["der_lambda_matrix"],
                                    return_dict_nf_mctm[
                                        "der2_lambda_matrix_global_list"
                                    ][j],
                                )

                    if ((i + 1) % 2) == 0:
                        # even
                        # output = (self.flip_matrix @ output.T).T
                        return_dict_nf_mctm["output"] = (
                            self.flip_matrix @ return_dict_nf_mctm["output"].T
                        ).T
                    # else:
                    #    # odd

                if return_scores_hessian == True:
                    return_dict_nf_mctm["scores"] = (
                        -1
                        * return_dict_nf_mctm["output"]
                        * return_dict_nf_mctm["der_lambda_matrix_global"].squeeze(1)
                    )

                    for j in range(i + 1):  # j are all sum elements prior to i
                        return_dict_nf_mctm["der2_lambda_matrix_global_list"][j] = (
                            -1
                            * return_dict_nf_mctm["output"].unsqueeze(2)
                            * return_dict_nf_mctm["der2_lambda_matrix_global_list"][j]
                        )
                    return_dict_nf_mctm["der2_lambda_matrix_global_list"][i + 1] = (
                        -1
                        * return_dict_nf_mctm["der2_lambda_matrix_global_list"][i + 1]
                    )

                    return_dict_nf_mctm["hessian"] = (
                        torch.stack(
                            return_dict_nf_mctm["der2_lambda_matrix_global_list"], dim=0
                        )
                        .sum(0)
                        .squeeze(2)
                    )

            return return_dict_nf_mctm

    def latent_space_representation(self, y):
        """
        Returns the fully transformed latent space Z for a given input Y. Z is distributed as a Gaussian N(0,I).

        Parameters
        ----------
        y : torch.FloatTensor
            The input data to pass through the model foward pass.
        return_lambda_matrix: bool = True
            Wether to compute the global lambda matrix in the forward pass or not. To not do it save a bit of compute.

        Returns
        -------
        A dictionary containing the latent space named "output", the log determinant "log_d", the differences for thes spline penalites
        "transformation_second_order_ridge_pen_global", "second_order_ridge_pen_global", "second_order_ridge_pen_global", "param_ridge_pen_global"
        and the full model, hence global, lambda matrix "lambda_matrix_global".
        """
        # covariate=False
        return_dict = self.forward(y)  # , covariate) #, train=False, evaluate=True)
        return return_dict["output"]

    def __log_likelihood_loss__(self, y, mean_loss=True):
        # covariate=False, train=True, evaluate=True,

        return_dict_nf_mctm = log_likelihood(
            model=self, samples=y, mean_loss=mean_loss
        )  # train_covariates=covariate, train=train, evaluate=evaluate,

        return_dict_nf_mctm["negative_log_likelihood_data"] = (
            -1 * return_dict_nf_mctm["log_likelihood_data"]
        )

        return return_dict_nf_mctm

    def log_likelihood(self, y: Tensor, mean_loss: bool=False) -> Tensor:  # covariate=False,
        """
        Returns the the log likelihood per sample for input Y.

        Parameters
        ----------
        y : torch.FloatTensor
            The input data for which to compute the log likelihood.
        mean_loss: bool = False
            Whether to return the mean of the log likelihood or not.

        Returns
        -------
        Returns the the log likelihood per sample for input Y.
        """
        return self.__log_likelihood_loss__(y=y, mean_loss=mean_loss)[
            "log_likelihood_data"
        ]  # covariate=False, train=False, evaluate=True

    
    
    def __training_objective__(
        self,
        samples: Tensor,
        penalty_params: FloatTensor,
        train_covariates:Tensor|bool = False,
        lambda_penalty_params: Tensor|bool = False,
        adaptive_lasso_weights_matrix: Tensor|bool = False,
        avg:bool=True,
        lambda_penalty_mode:Literal['square']="square",
        objective_type:Literal['negloglik']="negloglik",
    ) -> dict[str, Tensor]:

        return training_objective(
            model=self,
            samples=samples,
            penalty_params=penalty_params,
            train_covariates=train_covariates,
            lambda_penalty_params=lambda_penalty_params,
            adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix,
            avg=avg,
            lambda_penalty_mode=lambda_penalty_mode,
            objective_type=objective_type,
        )
        
    def __bayesian_training_objective__(
        self, 
        samples: Tensor,
        hyperparameters_transformation: dict[str, float],
        hyperparameters_decorrelation: dict [str, float],
        N_total,
        B,
        objective_type: Literal['negloglik'] = "negloglik"
    ):
        
        return bayesian_training_objective(
            model = self,
            hyperparameter_decorrelation=hyperparameters_decorrelation,
            hyperparameter_transformation =hyperparameters_transformation,
            samples= samples,
            objective_type= objective_type,
            sample_size=N_total,
            batch_size=B
        )

    def train(
        self,
        train_dataloader: DataLoader,
        validate_dataloader: DataLoader | bool = False,
        penalty_splines_params: FloatTensor = None,
        penalty_lasso_conditional_independence: float | bool = False,
        adaptive_lasso_weights_matrix: Tensor | bool = False,
        optimizer: Literal["LBFGS", "Adam"] = "LBFGS",
        learning_rate: float = 1,
        iterations: int = 1000,
        patience: int = 5,
        min_delta: float = 1e-7,
        seperate_copula_training: bool = False,
        max_batches_per_iter: int | bool = False,
        
        #BAYES VI
        hyperparameters=None,
        mu_init: Tensor|None = None,
        mcmc_sample_train=4,
        mcmc_sample_val=16,
        mc_ramp_every=25,
        mc_ramp_max=32,
        sched_factor=0.5,
        sched_patience=6,
        sched_threshold=1.e-4,
        warm_tau_epochs: int = 3,
        warm_sigma_epochs: int = 10,
        beta_kl_start: float = 3.0,
        beta_kl_anneal_epochs: int = 20,
        tau_vi_mode="after_warm",
        tau_kl_beta=1.0,
        tau_vi_sigma_init=0.25,
        conv_use_ema: bool = True,
        conv_window_size: int = 5,
        conv_tol: float = 1e-5,
        conv_min_epochs: int = 10,
        conv_ema_beta: float = 0.9,
        use_empirical_bayes: bool = False,
        eb_warm_then_cavi: bool = True,
        band_tau4: float = 0.20,
        band_decor: float = 0.15,
        lr_mu: float = 1e-3,
        lr_cholesky: float = 1e-4,
        lr_rho: float = 3e-4,
        lr_tau: float = 1.5e-3,
        sched_cooldown: int = 1,
        sched_min_lr: float = 5e-5,
        global_seed: int = 0,
    ) -> dict[str, Tensor]:
        """
        Trains the GTM iteratively using gradient-based optimization.

        Parameters
        ----------
        train_dataloader : DataLoader
            DataLoader containing the training data.

        validate_dataloader : DataLoader or bool, optional
            DataLoader for validation data. If set to False, no validation based early stopping is performed. Default is False.

        penalty_splines_params : torch.FloatTensor, optional
            A tensor of shape (4,) specifying regularization strengths for different spline penalties:
            (0) Ridge penalty on decorrelation layer spline parameters.
            (1) First derivative ridge penalty (via first-order differencing) for decorrelation layer splines.
            (2) Second derivative ridge penalty (via second-order differencing) for decorrelation layer splines.
            (3) Second derivative ridge penalty (via second-order differencing) for transformation layer splines.
            Default is zero for all components.

        penalty_lasso_conditional_independence : float or bool, optional
            Lasso penalty applied to encourage conditional independence based on the pseudo-correlation matrix of the training data.
            If set to a float, standard Lasso is applied. If set to True and `adaptive_lasso_weights_matrix` is provided,
            adaptive Lasso is applied. If False, Lasso is disabled.

        adaptive_lasso_weights_matrix : torch.Tensor or bool, optional
            Tensor specifying adaptive Lasso weights. Should be a square matrix of shape (d, d), where `d` is the number of variables.
            Only the lower triangular part is used. A common approach is to first train without adaptive Lasso and then
            retrain using weights based on the previous run, e.g., `1 / pseudo_corr_matrix.mean().abs()`.
            If False or `penalty_lasso_param` is False, adaptive Lasso is not applied.

        optimizer : {"LBFGS", "Adam"}, optional
            Optimizer used for training. Default is "LBFGS".

        learning_rate : float, optional
            Learning rate for the optimizer. Default is 1.0.
            Has no effect for "LBFGS", as a Wolfe line search is used instead.

        iterations : int, optional
            Maximum number of training iterations. Default is 1000.

        patience : int, optional
            Number of validation steps with no improvement before early stopping. Default is 5.

        min_delta : float, optional
            Minimum required change in validation loss to qualify as improvement. Default is 1e-7.

        seperate_copula_training : bool, optional
            If True, the copula (decorrelation layers) is trained separately after the transformation layers.
            This may help ensure Gaussian marginals in the latent space after transformation. Default is False.

        max_batches_per_iter : int or bool, optional
            If set to an integer, limits the number of batches used per training iteration. If False, all batches are used. Default is False.

        Returns
        -------
        dict
            Dictionary containing:
            - "loss_list_training": list of training losses per iteration
            - "loss_list_validation": list of validation losses per iteration (if validation used)
            - "number_iterations": total number of iterations performed
            - "training_time": total time spent in training (seconds)
            - All additional outputs from the final model's forward pass
        """
        objective_type: Literal['negloglik'] = "negloglik"
        
        ### HARDCODED, Not Implemented yet ###
        train_covariates: bool = False
        validate_covariates: bool = False
        ### HARDCODED, Not Implemented yet ###
        
        verbose: bool = True
        lambda_penalty_mode: Literal['square']= "square"  # Literal["square", "absolute"]
        # ema_decay: float | bool = False, used to have ema_decay in training

        if seperate_copula_training:
            self.transformation.params.requires_grad = False

        if self.spline_decorrelation == "bernstein":
            for layer in self.decorrelation_layers:
                layer.binom_n = layer.binom_n.to(self.device)
                layer.binom_n1 = layer.binom_n1.to(self.device)
                layer.binom_n2 = layer.binom_n2.to(self.device)
        
        if self.spline_transformation == "bernstein":
            self.transformation.binom_n = self.transformation.binom_n.to(self.device)
            self.transformation.binom_n1 = self.transformation.binom_n1.to(self.device)
            self.transformation.binom_n2 = self.transformation.binom_n2.to(self.device)

        if self.inference == 'frequentist':
            
            if penalty_lasso_conditional_independence is not False:
                penalty_lasso_conditional_independence = penalty_lasso_conditional_independence.to(self.device)

            if adaptive_lasso_weights_matrix is not False:
                adaptive_lasso_weights_matrix = adaptive_lasso_weights_matrix.to(self.device)
            
            return_dict_model_training: dict[str, Tensor] = train_freq(
                model=self,
                train_dataloader=train_dataloader,
                validate_dataloader=validate_dataloader,
                train_covariates=train_covariates,
                validate_covariates=validate_covariates,
                penalty_params=penalty_splines_params,
                lambda_penalty_params=penalty_lasso_conditional_independence,
                learning_rate=learning_rate,
                iterations=iterations,
                verbose=verbose,
                patience=patience,
                min_delta=min_delta,
                optimizer=optimizer,
                lambda_penalty_mode=lambda_penalty_mode,
                objective_type=objective_type,
                adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix,
                max_batches_per_iter=max_batches_per_iter,
            )
            
        elif self.inference == 'bayesian':
            return_dict_model_training = train_bayes(
                model=self,
                train_dataloader=train_dataloader,
                validate_dataloader=validate_dataloader,
                iterations=iterations,
                hyperparameters=hyperparameters,
                verbose=verbose,
                max_batches_per_iter=max_batches_per_iter,
                mcmc_sample_train=mcmc_sample_train,
                mcmc_sample_val=mcmc_sample_val,
                mc_ramp_every=mc_ramp_every,
                mc_ramp_max=mc_ramp_max,
                global_seed=global_seed,
                patience_val=patience,
                min_delta=min_delta,
                lr_mu=lr_mu,
                lr_cholesky=lr_cholesky,
                lr_rho=lr_rho,
                lr_tau=lr_tau,
                sched_factor=sched_factor,
                sched_patience=sched_patience,
                sched_threshold=sched_threshold,
                sched_cooldown=sched_cooldown,
                sched_min_lr=sched_min_lr,
                warm_tau_epochs=warm_tau_epochs,
                warm_sigma_epochs=warm_sigma_epochs,
                beta_kl_start=beta_kl_start,
                beta_kl_anneal_epochs=beta_kl_anneal_epochs,
                use_empirical_bayes=use_empirical_bayes,
                eb_warm_then_cavi=eb_warm_then_cavi,
                band_tau4=band_tau4,
                band_decor=band_decor,
                tau_vi_mode=tau_vi_mode,
                tau_kl_beta=tau_kl_beta,
                tau_vi_sigma_init=tau_vi_sigma_init,
                conv_use_ema=conv_use_ema,
                conv_window_size=conv_window_size,
                conv_tol=conv_tol,
                conv_min_epochs=conv_min_epochs,
                conv_ema_beta=conv_ema_beta,
                mu_init=mu_init
            )
        else:
            raise NotImplementedError('Selected Inference is not recognized or is not implemented yet.')

        if seperate_copula_training:
            self.transformation.params.requires_grad = True

        return return_dict_model_training
    
    def pretrain_transformation_layer(
        self,
        train_dataloader: DataLoader,
        validate_dataloader: DataLoader | bool = False,
        penalty_splines_params: FloatTensor = None,
        penalty_lasso_param: float | bool = False,
        learning_rate: float = 1,
        iterations: int = 2000,
        patience: int = 5,
        min_delta: float = 1e-7,
        optimizer: Literal["LBFGS", "Adam"] = "LBFGS",
        max_batches_per_iter: int | bool = False,
        hyperparameters = None,
        mcmc_sample_train=4,
        mcmc_sample_val=16,
        mc_ramp_every=25,
        mc_ramp_max=32,
        rho_lr_multiplier=1.5,
        sched_factor=0.5,
        sched_patience=6,
        sched_threshold=1.e-4,
        #WARMING
        warm_tau_epochs: int = 3,
        warm_sigma_epochs: int = 10,  # try 5–10
        
        #Optimization method
        beta_kl_start: float = 3.0,    # try 1.5–3.0
        beta_kl_anneal_epochs: int = 20,  # how fast to decay to 1.0
    ) -> None:
        
        """
        Pretrains only the transformation layer of the GTM using gradient-based optimization.

        Parameters
        ----------
        train_dataloader : DataLoader
            DataLoader containing the training data.
        validate_dataloader : DataLoader or bool, optional
            DataLoader for validation data. If False, no validation based early stopping is performed. Default is False.
        penalty_splines_params : torch.FloatTensor, optional
            A tensor of shape (4,) specifying regularization strengths for different model spline regularizations:
            (0) decorrelation layer splines parameter ridge penalty (`penalty_decorrelation_ridge_param`),
            (1) decorrelation layer splines first derivative ridge penalty (`penalty_decorrelation_ridge_first_difference`),
            (2) decorrelation layer splines second derivative ridge penalty (`penalty_decorrelation_ridge_second_difference`),
            (3) transformation layer splines second derivative ridge penalty (`penalty_transformation_ridge_second_difference`).
            Only the fourth component is used during transformation pretraining. Default is zero.
        penalty_lasso_param : float or bool, optional
            Lasso penalty towards conditional independence applied to the pseudo correlation matrix.
            Usually not relevant during transformation-only training, but included for flexibility.
            If False, no Lasso is applied.
        learning_rate : float, optional
            Learning rate for the optimizer. Default is 1.0.
            Has no effect for "LBFGS", as a Wolfe line search is used instead.
        iterations : int, optional
            Maximum number of training iterations. Default is 2000.
        patience : int, optional
            Number of validation steps with no improvement before early stopping. Default is 5.
        min_delta : float, optional
            Minimum change in validation loss to qualify as an improvement. Default is 1e-7.
        optimizer : {"LBFGS", "Adam"}, optional
            Optimizer to use for training. Default is "LBFGS".
        max_batches_per_iter : int or bool, optional
            If set, limits the number of batches used per training iteration. Default is False (use all batches).

        Returns
        -------
        dict
            Dictionary with training information, including:
            - "loss_list_training": list of training loss values
            - "loss_list_validation": list of validation loss values (if validation used)
            - "number_iterations": total number of iterations performed
            - "training_time": total training time in seconds
            - other output keys from the forward pass of the final model state.
        """
        
        objective_type:Literal['negloglik'] = "negloglik"
        
        verbose:bool = True
        
        # ema_decay: float | bool = False, used to have ema_decay in training

        # optimizer='LBFGS'
        # warnings.warn("Optimiser for pretrain_transformation_layer is always LBFGS. If this is an issue change the code.")

        self.transform_only = True
        penalty_lasso_conditional_independence = False  # makes objective not check lambda matrix

        lambda_penalty_mode: Literal['square'] = "square"  # Literal["square", "absolute"]
        train_covariates:bool = False
        validate_covariates:bool = False
        
        if self.inference == "frequentist":
            return_dict_model_training: None = train_freq(
                model=self,
                train_dataloader=train_dataloader,
                validate_dataloader=validate_dataloader,
                train_covariates=train_covariates,
                validate_covariates=validate_covariates,
                penalty_params=penalty_splines_params,
                lambda_penalty_params=penalty_lasso_conditional_independence,
                learning_rate=learning_rate,
                iterations=iterations,
                verbose=verbose,
                patience=patience,
                min_delta=min_delta,
                optimizer=optimizer,
                lambda_penalty_mode=lambda_penalty_mode,
                objective_type=objective_type,
                max_batches_per_iter=max_batches_per_iter
                )
        elif self.inference =="bayesian":
            return_dict_model_training = train_bayes(
                model= self,
                train_dataloader=train_dataloader,
                validate_dataloader=validate_dataloader,
                hyperparameters=hyperparameters,
                iterations=iterations,
                verbose=verbose,
                lr=learning_rate,
                mcmc_sample_train=mcmc_sample_train,#4,
                mcmc_sample_val=mcmc_sample_val,#16,
                mc_ramp_every=mc_ramp_every,#25,
                mc_ramp_max=mc_ramp_max,#32,
                patience_val=patience,#15,
                min_delta=min_delta,#15,
                rho_lr_multiplier=rho_lr_multiplier,#1.5,
                sched_factor=sched_factor,#0.5,
                sched_patience=sched_patience,#6,
                sched_threshold=sched_threshold,
                warm_tau_epochs=warm_tau_epochs,
                warm_sigma_epochs= warm_sigma_epochs,  # try 5–10
                
                #Optimization method
                beta_kl_start = beta_kl_start,    # try 1.5–3.0
                beta_kl_anneal_epochs = beta_kl_anneal_epochs,  # how fast to decay to 1.0
            )

        self.transform_only = False

        return return_dict_model_training

    def find_minimal_transformation_degrees(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        iterations: int = 100,
        degrees_try_list: list[int] = list(range(5, 155, 5)),
        max_batches_per_iter: int | bool = False,
    ):
        """
        Searches for the smallest transformation spline degree that is enough to transform the training data into standard Gaussian space for each data dimension individually.

        For each data dimension, this method iteratively trains the transformation layer using spline degrees from the provided list, starting at the smallest and increasing.
        It then perorms a shapiro wilk test on the latent space after transformation to evaluate if the current degree was enough to transform the data into a gaussian.
        If the P-value > 0.05 then the degree is enough, stored and the process continues with the next data dimension.

        Parameters
        ----------
        train_dataloader : DataLoader
            DataLoader containing the training data.

        iterations : int, optional
            Number of training iterations per spline degree tested. Default is 100.

        degrees_try_list : list of int, optional
            List of spline degrees to try for the transformation layer. The method will test each and select the best.
            If a custom list is passed make sure it is increasing in value.
            Default is `list(range(5, 155, 5))`.

        max_batches_per_iter : int or bool, optional
            If set to an integer, limits the number of batches used in each training iteration. If False, all batches are used.
            Default is False.

        Returns
        -------
        The chosen degrees as a list and the respective p-values as a list.
        """

        optimal_degree = []
        optimal_degree_pvalue = []
        for dimension in range(self.number_variables):
            for degree in degrees_try_list:
                print(
                    "Starting run for data dim ", dimension, " with degrees of ", degree
                )
                try:

                    tm_model = GTM(
                        number_variables=1,
                        transformation_spline_range=[
                            self.transformation_spline_range[0][dimension],
                            self.transformation_spline_range[1][dimension],
                        ],
                        number_decorrelation_layers=0,
                        number_transformation_layers=1,
                        degree_transformations=[degree],
                        spline_transformation=self.spline_transformation,
                    )

                    # tm_model = TM(degree=degree, spline_range=[self.transformation_spline_range[0][dimension],
                    #                                        self.transformation_spline_range[1][dimension]])
                    tm_model.subset_dimension = dimension

                    train_dict = tm_model.train(
                        train_dataloader=train_dataloader,
                        validate_dataloader=train_dataloader,
                        iterations=iterations,
                        optimizer="LBFGS",
                        max_batches_per_iter=max_batches_per_iter,
                    )

                    z_tilde = []
                    # y_train_all = []
                    for y_train in train_dataloader:
                        y_train_sub = y_train[:, tm_model.subset_dimension].unsqueeze(1)
                        z_tilde.append(
                            tm_model.after_transformation(y_train_sub).squeeze()
                        )
                        # y_train_all.append(y_train_sub)
                    z_tilde = torch.hstack(z_tilde).detach().numpy()
                    # y_train_all = torch.hstack(y_train_all).detach().numpy()

                    # perform Shapiro-Wilk test for normality
                    pv = scipy.stats.shapiro(
                        z_tilde[:5000]
                    ).pvalue  # becuase warning that pvalue may not be accurate for larger than 5000 obs in the package
                    # print(pv)
                    # plt.hist(y_train_all,bins=100)
                    # plt.hist(z_tilde,bins=100)
                    # plt.show()
                    # plt.hist(z_tilde,bins=100)
                    if pv >= 0.05:

                        print(
                            "pvalue is ",
                            pv,
                            " for data dim ",
                            dimension,
                            " with degrees of ",
                            degree,
                        )

                        optimal_degree.append(degree)
                        optimal_degree_pvalue.append(pv)

                        plt.hist(z_tilde, bins=100)
                        plt.show()

                        # for next loop iteration to not always plot
                        pv = 0

                        break
                    # Handels the case where even the max degree is not enough
                    elif degree == max(degrees_try_list):
                        optimal_degree.append(degree)
                        optimal_degree_pvalue.append(pv)
                except:
                    continue

        return optimal_degree, optimal_degree_pvalue

    def compute_pseudo_precision_matrix(
        self, y: torch.Tensor
    ) -> torch.Tensor:  # , covariate=False):
        """
        Computes the pseudo precision matrix from the data `y` based on the lambda matrices of the full GTM model.

        Parameters
        ----------
        y : torch.Tensor
            A 2D tensor of shape (n_samples, n_variables).

        Returns
        -------
        torch.Tensor
            A tensor of shape (n_samples, n_variables, n_variables) representing the pseudo precision matrix.
        """

        with torch.no_grad():
            return_dict = self.forward(
                y, return_lambda_matrix=True
            )  # covariate=covariate, evaluate=True, train=False, return_lambda_matrix=True)

            precision_matrix = torch.matmul(
                torch.transpose(return_dict["lambda_matrix_global"], 1, 2),
                return_dict["lambda_matrix_global"],
            )

        return precision_matrix

    def compute_pseudo_conditional_correlation_matrix(
        self, y: torch.Tensor
    ) -> (
        torch.Tensor
    ):  # , covariate=False): #TODO: is pseudo conditional indepednence matrix! standardisted p matrix
        """
        Computes the pseudo conditional correlation matrix from the data `y` based on the lambda matrices of the full GTM model.
        This is essentially the standardised precision matrix, that way off diagonal elements represent the pseudo conditional correlations between the different dimensions.

        Parameters
        ----------
        y : torch.Tensor
            A 2D tensor of shape (n_samples, n_variables).

        Returns
        -------
        torch.Tensor
            A tensor of shape (n_samples, n_variables, n_variables) representing the pseudo conditional correlation matrix.
        """

        def p_to_corr(matrix):
            d = matrix.size(0)
            diag_sqrt = torch.diag(matrix) ** 0.5
            matrix_std_multiplied = torch.matmul(
                torch.reshape(diag_sqrt, (d, 1)), torch.reshape(diag_sqrt, (1, d))
            )
            return -1 * matrix / matrix_std_multiplied

        with torch.no_grad():
            precision_matrix = self.compute_pseudo_precision_matrix(
                y
            )  # , covariate=False)
            correlation_matrix_train = torch.stack(
                [
                    p_to_corr(precision_matrix[obs_num, :, :])
                    for obs_num in range(precision_matrix.size(0))
                ]
            )

        return correlation_matrix_train

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Generates samples from the fitted Graphical Transformation Model (GTM).

        This method draws samples from the latent standard Gaussian space,
        passes them through the learned inverse decorrelation layers and then the approximated inverse transformation layer,
        and returns them in the original data space.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        torch.Tensor
            A tensor of shape (n_samples, n_variables) containing the generated samples.
        """

        covariate = False

        # check if transformation contains params_inverse
        if self.transformation.params_inverse is None:
            warnings.warn(
                "Transformation layer does not have an inverse. Inverse is approximated"
            )
            self.approximate_transformation_inverse()

        with torch.no_grad():

            z = (
                torch.distributions.Normal(0, 1)
                .sample((n_samples, self.number_variables))
                .to(device=self.device)
            )

            if self.number_decorrelation_layers > 0:
                for i in range(self.number_decorrelation_layers - 1, -1, -1):

                    if ((i + 1) % 2) == 0:
                        # even
                        z = (self.flip_matrix @ z.T).T
                        # else:
                        #    # odd

                    return_dict = self.decorrelation_layers[i](
                        z,
                        covariate=covariate,
                        return_log_d=False,
                        return_penalties=False,
                        inverse=True,
                    )
                    z = return_dict["output"]

                    if ((i + 1) % 2) == 0:
                        # even
                        z = (self.flip_matrix @ z.T).T
                        # else:
                        #    # odd

            if self.num_trans_layers > 0:
                return_dict = self.transformation(
                    z, covariate, new_input=True, inverse=True
                )
                y = return_dict["output"]
            else:
                y = z

            if self.initial_log_transform == True:
                y = torch.exp(y)
                y = y - 0.01

            return y

    def after_transformation(self, y, covariate=False):
        if self.initial_log_transform == True:
            y = y + 0.01
            y = torch.log(y)
        else:
            pass
        if self.num_trans_layers > 0:
            # new input true as we need to recompute the basis for the validation/test set
            return_dict = self.transformation(y, covariate, new_input=True)
            output = return_dict["output"]
        else:
            output = y
        return output

    def conditional_sample(
        self,
        number_proposed_samples,
        sample_indices,
        sample_ranges,
        fixed_values,
        covariate=False,
    ):
        with torch.no_grad():
            num_dim = self.number_variables

            # Initialize samples with fixed values
            samples = torch.full((number_proposed_samples, num_dim), float("nan"))
            for idx, value in fixed_values.items():
                samples[:, idx] = value  # Set fixed values

            # Sample the desired dimensions uniformly within sample_range
            idx = 0
            for sample_idx in sample_indices:
                min_val, max_val = sample_ranges[idx]
                samples[:, sample_idx] = (
                    torch.rand((number_proposed_samples)) * (max_val - min_val)
                    + min_val
                )

                idx += 1

            # Compute acceptance probabilities using log likelihood
            log_probs = self.log_likelihood(
                samples
            )  # , covariate=covariate)  # Shape: (max_attempts,)
            probs = torch.exp(
                log_probs - torch.max(log_probs)
            )  # Normalize to avoid overflow
            probs /= torch.max(probs)  # Scale probabilities between 0 and 1

            # Accept/reject samples based on computed probabilities
            accept_mask = torch.rand(number_proposed_samples) < probs
            accepted_samples = samples[accept_mask]

            print(accepted_samples.size(0), "samples where accepted.")

            return accepted_samples

    def approximate_transformation_inverse(self) -> None:
        """
        Approximates the inverse of the transformation layer splines and stores the result.

        This method computes an approximate inverse for each univariate transformation
        spline in the transformation layer. The resulting parameters are cached within
        the model and used during sampling to transform latent Gaussian samples back
        to the original data space.

        Returns
        -------
        None
        """
        self.transformation.approximate_inverse(device=self.device)

    def __return_objective_for_hyperparameters__(
        self,
        train_dataloader: DataLoader,
        validate_dataloader: DataLoader| bool = False,
        #train_covariates=False,
        #validate_covariates=False,
        penalty_params: FloatTensor = torch.FloatTensor([0, 0, 0, 0]),
        adaptive_lasso_weights_matrix:bool=False,
        lambda_penalty_param:bool=False,
        learning_rate:float=1.0,
        iterations:int=2000,
        patience:int=5,
        min_delta:float=1e-7,
        optimizer:Literal["LBFGS", "Adam"]="LBFGS",
        lambda_penalty_mode:Literal["square"]="square", #Not used
        objective_type:Literal['negloglik']="negloglik",
        seperate_copula_training:bool=False,
        max_batches_per_iter:bool|int|None=False,
        pretrained_transformation_layer=False,
        cross_validation_folds:bool|None=False,
    )  -> float:

        import copy

        gtm_tuning: GTM = copy.deepcopy(self)

        gtm_tuning.to(device=self.device)
        gtm_tuning.device = self.device

        # Logic here:
        # we only want to pretrain the transformation layer once if we do not do cross validation folds
        # In that case each pretaining is basically the same, as same init params and same hyperparameters
        # If cross_validation_folds is False then only in the first trial we do a pretrain and then store the transformation layer pretrained model
        # In each subsequent trial we load the pretrained model and directly do the joint training
        # This only works if we pretrain without a penalty on the transformation layer
        if pretrained_transformation_layer:
            if hasattr(self, "pretrained_transformation_layer_model_state_dict"):  # pretrained_transformation_layer_model
                
                #gtm_tuning.load_state_dict(self.pretrained_transformation_layer_model.state_dict())
                gtm_tuning.load_state_dict(self.pretrained_transformation_layer_model_state_dict)
                
            else:
                
                gtm_tuning.pretrain_transformation_layer(
                    train_dataloader=train_dataloader,
                    validate_dataloader=validate_dataloader,
                    # train_covariates=train_covariates,
                    # validate_covariates=validate_covariates,
                    penalty_splines_params=penalty_params,
                    penalty_lasso_param=lambda_penalty_param,
                    iterations=iterations,
                    learning_rate=learning_rate,
                    patience=patience,
                    min_delta=min_delta,
                    optimizer=optimizer,
                    max_batches_per_iter=max_batches_per_iter,
                )

                if cross_validation_folds == False:
                    # self.pretrained_transformation_layer_model = copy.deepcopy(gtm_tuning)
                    self.pretrained_transformation_layer_model_state_dict = gtm_tuning.state_dict()
                    
        gtm_tuning.train(
            train_dataloader=train_dataloader,
            validate_dataloader=validate_dataloader,
            # train_covariates=train_covariates,
            # validate_covariates=validate_covariates,
            penalty_splines_params=penalty_params,
            penalty_lasso_conditional_independence=lambda_penalty_param,
            adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix,
            iterations=iterations,
            learning_rate=learning_rate,
            patience=patience,
            min_delta=min_delta,
            optimizer=optimizer,
            seperate_copula_training=seperate_copula_training,
            max_batches_per_iter=max_batches_per_iter,
        )
        
        num_batches: int = 0
        target: float = 0.0
        covar_batch: bool = False
        
        for y_validate in validate_dataloader:
            
            y_validate: Tensor = y_validate.to(self.device)
            num_batches += 1
            
            if objective_type == "negloglik":
                
                batch: Tensor = gtm_tuning.log_likelihood(y=y_validate, mean_loss=covar_batch)
                target += batch.cpu().detach().numpy().mean()
                
            elif objective_type in ("score_matching", "single_sliced_score_matching"):
                
                # TODO: does the -1 * make sense is it not already in the exact_score_loss method
                target += (
                    -1
                    * gtm_tuning.exact_score_matching(y_validate, covar_batch)
                    .cpu()
                    .detach()
                    .numpy()
                    .mean()
                )  # .mean() # maximize the reverse score loss e.g.g minimize the loss
            elif objective_type == "noise_contrastive_estimation":
                target += (
                    gtm_tuning.log_likelihood(y_validate, covar_batch)
                    .cpu()
                    .detach()
                    .numpy()
                    .mean()
                )  # TODO: For Now! Its cheating right?

        target /= num_batches

        # Handelling CUDA Out of Memory Error
        if self.device == "cuda":
            # Explicitly delete the model to free up memory
            del gtm_tuning

            # Clear the cache
            torch.cuda.empty_cache()

        return target

    def hyperparameter_tune_penalties(
        self,
        train_dataloader: DataLoader,
        validate_dataloader: DataLoader,
        penalty_decorrelation_ridge_param: float | str | None = None,
        penalty_decorrelation_ridge_first_difference: float | str | None = None,
        penalty_decorrelation_ridge_second_difference: float | str | None = None,
        penalty_transformation_ridge_second_difference: float | str | None = None,
        penalty_lasso_conditional_independence: float | str | None = None,
        adaptive_lasso_weights_matrix: Tensor | Literal[False] | None  = False,
        optimizer:Literal["LBFGS", "Adam"]| None = "LBFGS",
        learning_rate:float | None =1.0,
        iterations:int | None = 2000,
        patience:Optional[int]=5,
        min_delta:float=1e-7,
        seperate_copula_training:bool=False,
        max_batches_per_iter:bool|int|None=False,
        pretrained_transformation_layer:bool|None=False,
        n_trials:int=15,
        temp_folder:str=".",
        study_name:str=None,
    ) -> Study | None:
        """
        Tunes the regularization hyperparameters of the GTM model using Optuna.

        This method allows either manual specification of penalty values (as floats),
        turning off a penalty by passing None (interpreted as zero),
        or enabling Optuna to sample the penalty using the string "sample".

        The penalties tuned or set include:
        - Ridge penalty on decorrelation layer splines parameters (`penalty_decorrelation_ridge_param`)
        - First derivative ridge penalty on decorrelation splines via finite differencing (`penalty_decorrelation_ridge_first_difference`)
        - Second derivative ridge penalty on decorrelation splines (`penalty_decorrelation_ridge_second_difference`)
        - Second derivative ridge penalty on transformation splines (`penalty_transformation_ridge_second_difference`)
        - Lasso penalty encouraging sparsity in the pseudo precision matrix (`penalty_lasso_conditional_independence`)

        Parameters
        ----------
        train_dataloader : DataLoader
            The training data.
        validate_dataloader : DataLoader
            The validation data used to evaluate candidate hyperparameters.
        penalty_decorrelation_ridge_param : float, str, or None
            Penalty on decorrelation layer spline parameters.
            - Float: fixed penalty value.
            - None: no penalty (interpreted as 0).
            - "sample": Optuna samples a value during tuning.
        penalty_decorrelation_ridge_first_difference : float, str, or None
            Penalty on first derivative of decorrelation splines.
            Same conventions as above.
        penalty_decorrelation_ridge_second_difference : float, str, or None
            Penalty on second derivative of decorrelation splines.
            Same conventions as above.
        penalty_transformation_ridge_second_difference : float, str, or None
            Penalty on second derivative of transformation splines.
            Same conventions as above.
        penalty_lasso_conditional_independence : float, str, or None
            Lasso penalty applied to encourage conditional independence in the pseudo precision matrix.
            If "sample", Optuna selects the penalty value.
        train_covariates : torch.Tensor or bool, optional
            Optional covariates associated with training data. Default is False.
        validate_covariates : torch.Tensor or bool, optional
            Optional covariates associated with validation data. Default is False.
        adaptive_lasso_weights_matrix : torch.Tensor or bool, optional
            Square matrix of adaptive Lasso weights (lower triangular part used).
            Set to False to disable adaptive Lasso.
        optimizer : {"LBFGS", "Adam"}, optional
            Optimizer used for training. Default is "LBFGS".
        learning_rate : float, optional
            Learning rate for the optimizer. Default is 1.0.
            Has no effect for "LBFGS", as a Wolfe line search is used instead.
        iterations : int, optional
            Maximum number of training iterations for each trial. Default is 2000.
        patience : int, optional
            Number of validation rounds without improvement before early stopping. Default is 5.
        min_delta : float, optional
            Minimum improvement in validation loss to qualify as a better model. Default is 1e-7.
        seperate_copula_training : bool, optional
            If True, t
            he decorrelation layers (copula) are trained after the transformation layers. Default is False.
        max_batches_per_iter : int or bool, optional
            If set to an integer, limits number of batches used in each training iteration. Default is False (use all).
        pretrained_transformation_layer : bool, optional
            If True, pretrains the transformation layers before the tuning. Default is False.
        n_trials : int, optional
            Number of Optuna trials (i.e., hyperparameter configurations) to evaluate. Default is 15.
        temp_folder : str, optional
            Path to temporary directory used for Optuna study storage. Default is current directory. Is deleted once tuning is over.
        study_name : str or None, optional
            Name of the Optuna study (used for resuming or caching tuning). Default is None.

        Returns
        -------
        optuna.study.Study
            The Optuna study object containing all trial results, including best hyperparameter configuration.
            The study contains the optimal penalties identified as keys. These are:
            - study[`penalty_decorrelation_ridge_param`]
            - study[`penalty_decorrelation_ridge_first_difference`]
            - study[`penalty_decorrelation_ridge_second_difference`]
            - study[`penalty_transformation_ridge_second_difference`]
            - study[`penalty_lasso_conditional_independence`]
        """
        # All for now fixed parameters
        lambda_penalty_mode = "square"
        objective_type = "negloglik"
        tuning_mode = "optuna"
        cross_validation_folds = False
        train_covariates = False
        validate_covariates = False
        # random_state_KFold=42
        # device=None

        # From Old version where one could pass lists to do tuning using list of hyperparameters passed
        # list_of_lists = [penalty_decorrelation_ridge_param, penalty_decorrelation_ridge_first_difference, penalty_decorrelation_ridge_second_difference,
        # penalty_transformation_ridge_second_difference,
        # penalty_lasso_conditional_independence]
        # hyperparameter_combinations_list = list(itertools.product(*list_of_lists))

        number_covariates = 1 if train_covariates else 0

        if tuning_mode == "optuna":
            # From Old version where one could pass lists to do tuning using list of hyperparameters passed
            # penalty_decorrelation_ridge_param, penalty_decorrelation_ridge_first_difference, penalty_decorrelation_ridge_second_difference, penalty_transformation_ridge_second_difference, penalty_lasso_conditional_independence  = hyperparameter_combinations_list[0]

            def optuna_objective(
                trial,
                train_dataloader=train_dataloader,
                validate_dataloader=validate_dataloader,
            ):
                
                ##### DEFINING PENALISATION PARAMETERS (START) ####
                if penalty_decorrelation_ridge_param is None:
                    
                    penalty_decorrelation_ridge_param_opt = 0
                    
                elif isinstance(penalty_decorrelation_ridge_param, (float, int)):
                    penalty_decorrelation_ridge_param_opt: float | int = penalty_decorrelation_ridge_param
                    
                elif penalty_decorrelation_ridge_param == "sample":
                    penalty_decorrelation_ridge_param_opt = trial.suggest_float(
                        "penalty_decorrelation_ridge_param",
                        0.0000001,
                        30,
                        log=False
                    )  # True
                else:
                    warnings.warn(
                        'penalty_decorrelation_ridge_param not understood. Please provide a float, int None, or the string "sample".'
                    )

                if penalty_decorrelation_ridge_first_difference is None:
                    
                    penalty_decorrelation_ridge_first_difference_opt = 0
                    
                elif isinstance(penalty_decorrelation_ridge_first_difference, (float, int)):
                    
                    penalty_decorrelation_ridge_first_difference_opt: float | int = penalty_decorrelation_ridge_first_difference
                    
                elif penalty_decorrelation_ridge_first_difference == "sample":
                    penalty_decorrelation_ridge_first_difference_opt = trial.suggest_float(
                        "penalty_decorrelation_ridge_first_difference",
                        0.0000001,
                        30,
                        log=False,
                    ) # True
                else:
                    warnings.warn(
                        'penalty_decorrelation_ridge_first_difference not understood. Please provide a float, int None, or the string "sample".'
                    )

                if penalty_decorrelation_ridge_second_difference is None:
                    
                    penalty_decorrelation_ridge_second_difference_opt = 0
                
                elif isinstance(penalty_decorrelation_ridge_second_difference, (float, int)):
                    
                    penalty_decorrelation_ridge_second_difference_opt: float | int = penalty_decorrelation_ridge_second_difference
                    
                elif penalty_decorrelation_ridge_second_difference == "sample":
                    penalty_decorrelation_ridge_second_difference_opt = trial.suggest_float(
                        "penalty_decorrelation_ridge_second_difference",
                        0.0000001,
                        30,
                        log=False
                        )  # True
                else:
                    warnings.warn(
                        'penalty_decorrelation_ridge_second_difference not understood. Please provide a float, int None, or the string "sample".'
                    )

                if penalty_transformation_ridge_second_difference is None:
                    
                    penalty_transformation_ridge_second_difference_opt = 0
                    
                elif isinstance(penalty_transformation_ridge_second_difference, (float, int)):
                    
                    penalty_transformation_ridge_second_difference_opt: float | int = penalty_transformation_ridge_second_difference
                    
                elif penalty_transformation_ridge_second_difference == "sample":
                    penalty_transformation_ridge_second_difference_opt = trial.suggest_float(
                        "penalty_transformation_ridge_second_difference",
                        0.0000001,
                        30,
                        log=False
                        )  # True
                else:
                    warnings.warn(
                        'penalty_transformation_ridge_second_difference not understood. Please provide a float, int None, or the string "sample".'
                    )

                if penalty_lasso_conditional_independence is None:
                    
                    penalty_lasso_conditional_independence_opt = 0
                elif isinstance(penalty_lasso_conditional_independence, (float, int)):
                    
                    penalty_lasso_conditional_independence_opt: float | int = penalty_lasso_conditional_independence
                    
                elif penalty_lasso_conditional_independence == "sample":
                    penalty_lasso_conditional_independence_opt = trial.suggest_float(
                        "penalty_lasso_conditional_independence",
                        0.0000001,
                        1,
                        log=True
                    )
                else:
                    warnings.warn(
                        'penalty_lasso_conditional_independence not understood. Please provide a float, int None, or the string "sample".'
                    )

                # print("This Trial has the Hyperparameters:",
                #    "penalty_decorrelation_ridge_param_opt:", penalty_decorrelation_ridge_param_opt, " ",
                #    "penalty_decorrelation_ridge_first_difference_opt:", penalty_decorrelation_ridge_first_difference_opt, " ",
                #    "penalty_decorrelation_ridge_second_difference_opt:", penalty_decorrelation_ridge_second_difference_opt, " ",
                #    "penalty_transformation_ridge_second_difference_opt:", penalty_transformation_ridge_second_difference_opt, " ",
                #    "penalty_lasso_conditional_independence_opt:", penalty_lasso_conditional_independence_opt)
                
                penalty_lasso_conditional_independence_opt = if_float_create_lambda_penalisation_matrix(
                    lambda_penalty_params=penalty_lasso_conditional_independence_opt,
                    num_vars=self.number_variables
                    )
                
            ##### DEFINING PENALISATION PARAMETERS  (END) ####
                
                penalty_params_opt: Tensor = torch.tensor(
                    data=[
                        penalty_decorrelation_ridge_param_opt,
                        penalty_decorrelation_ridge_first_difference_opt,
                        penalty_decorrelation_ridge_second_difference_opt,
                        penalty_transformation_ridge_second_difference_opt,
                    ]
                )

                if not cross_validation_folds:
                    # define model, train the model with tuning params and return the objective value on the given validation set
                    target:float = self.__return_objective_for_hyperparameters__(
                        train_dataloader=train_dataloader,
                        validate_dataloader=validate_dataloader,
                        #train_covariates=train_covariates, # ALLWAYS FALSE FOR THIS VERSION
                        #validate_covariates=validate_covariates, # ALLWAYS FALSE FOR THIS VERSION
                        penalty_params=penalty_params_opt,
                        adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix,
                        lambda_penalty_param=penalty_lasso_conditional_independence_opt,
                        learning_rate=learning_rate,
                        iterations=iterations,
                        patience=patience,
                        min_delta=min_delta,
                        optimizer=optimizer,
                        lambda_penalty_mode=lambda_penalty_mode,
                        objective_type=objective_type,
                        seperate_copula_training=seperate_copula_training,
                        max_batches_per_iter=max_batches_per_iter,
                        pretrained_transformation_layer=pretrained_transformation_layer,
                        cross_validation_folds=cross_validation_folds,
                    )

                    return target
                else:
                    warnings.warn(
                        "cross validation based hyperparameter tuning is not implemented yet based on dataloaders"
                    )
                    ## Perform cross-validation
                    # for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
                    #    print(f"Fold {fold + 1}: Train size = {len(train_idx)}, Val size = {len(val_idx)}")

            #
            #    # Create subset samplers
            #    train_subset = Subset(dataset, train_idx)
            #    val_subset = Subset(dataset, val_idx)
            #
            #    # Create DataLoaders
            #    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
            #    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
            #
            #    # Example: Iterate through one batch
            #    for batch in train_loader:
            #        x, y = batch
            #        print(f"Train Batch Shape: {x.shape}, Labels: {y.shape}")
            #        break  # Just show one batch per fold

            study: Study = optuna.create_study(
                sampler=TPESampler(
                    n_startup_trials=int(np.floor(n_trials / 2)),  # 7
                    consider_prior=False,  # True # is this useful without a prior weight?
                    # Set consider_prior=False othwise with score matching we got the error: raise ValueError("Prior weight must be positive.")
                    prior_weight=0,  # 1.0, #default value 1.0 but then does not explore the space as good I think
                    multivariate=True,  # experimental but very useful here as our parameters are highly correlated
                ),
                storage = f"sqlite:///{temp_folder}/hyperparameter_tuning_study.db",
                # hyperparameter_tuning_study.db',
                direction="maximize",
                study_name=study_name,
                load_if_exists=True,
            )

            study.optimize(optuna_objective, n_trials=n_trials)
            print("hyperparameter_tuning done")
            return study

    def compute_conditional_independence_table(
        self,
        y: torch.Tensor | None = None,
        evaluation_data_type: Literal[
            "data", "uniform_random_samples", "samples_from_model"
        ] = "data",
        num_processes: int = 10,
        sample_size: int = 1000,
        num_points_quad: int = 20,
        copula_only: bool = False,
        min_val: float = -5,
        max_val: float = 5,
        likelihood_based_metrics: bool = True,
        vi_model: Optional[torch.nn.Module] = None,
        S_posterior: int = 100,
        cred_level: float = 0.95,
    ):
        """
        Computes a table of pairwise conditional dependence statistics (e.g., KLD, IAE, pseudo precision matrix, and pseudo conditional correlation)
        to assess conditional independence relationships between variables.

        For the Likelihood based KLD and IAE, this is done via approximation of conditional log-densities using numerical quadrature, comparing full model predictions
        to models with conditional independence constraints imposed.

        Parameters
        ----------
        y : torch.Tensor, optional
            Input data matrix of shape (n_samples, n_variables) used for evaluation. Required if evaluation_data_type == "data".

        evaluation_data_type : {"data", "uniform_random_samples", "samples_from_model"}, default="data"
            Defines which kind of data is used for evaluation:
            - "data": use the passed tensor `y`.
            - "uniform_random_samples": generate uniform samples in [min_val, max_val].
            - "samples_from_model": generate samples from the trained model.

        num_processes : int, default=10
            Number of parallel processes to use. If <= 1, processes run serially.

        sample_size : int, default=1000
            Number of samples to use for the evaluation if sampling, depending on `evaluation_data_type` being "uniform_random_samples" or "samples_from_model".

        num_points_quad : int, default=20
            Number of quadrature points used for evaluating integrals of conditional distributions.

        copula_only : bool, default=False
            If True, evaluates only the copula likelihood component by disregarding the transformation layer likelihood contribution.
            Can be sensible to get more stable results if marginals are highly complex. Especially since the dependence is only modeled in the decorrelation layers.

        min_val : float, default=-5
            Minimum value for the quadrature bound as well as for uniform random samples when evaluation_data_type == "uniform_random_samples".

        max_val : float, default=5
            Maximum value for the quadrature bound as well as for uniform random samples when evaluation_data_type == "uniform_random_samples".

        likelihood_based_metrics : bool, default=True
            If True, computes log-likelihood based divergence metrics (KLD, IAE). These take time due to the quadratures.
            If False, only precision and conditional correlation metrics are computed.

        vi_model : torch.nn.Module, optional
            Variational inference model used for Bayesian inference. Required if `self.inference == "bayesian"`.

        tau_nodes : torch.Tensor, optional
            Tensor of tau nodes for Bayesian inference. Required if `self.inference == "bayesian"`.

        S_posterior : int, default=100
            Number of posterior samples for Bayesian inference.

        cred_level : float, default=0.95
            Credible level for Bayesian inference.

        Returns
        -------
        self.conditional_independence_table : pandas.DataFrame
            A table containing one row per variable pair with the following columns:
            - var_row: index of first variable
            - var_col: index of second variable
            - precision_abs_mean: mean absolute entry in the pseudo-precision matrix
            - precision_square_mean: mean squared entry in the pseudo-precision matrix
            - cond_correlation_abs_mean: mean absolute pseudo-conditional correlation
            - cond_correlation_square_mean: mean squared pseudo-conditional correlation
            - kld: Kullback-Leibler divergence (only if likelihood_based_metrics is True)
            - iae: Integrated Absolute Error (only if likelihood_based_metrics is True)
        """
        x = False
        optimized = True

        if self.inference == "frequentist":
            self.conditional_independence_table = compute_conditional_independence_kld(
                self,
                y=y,
                x=x,
                evaluation_data_type=evaluation_data_type,
                num_processes=num_processes,
                sample_size=sample_size,
                num_points_quad=num_points_quad,
                optimized=optimized,
                copula_only=copula_only,
                min_val=min_val,
                max_val=max_val,
                likelihood_based_metrics=likelihood_based_metrics,
            )
        elif self.inference == "bayesian":
            self.conditional_independence_table = compute_conditional_independence_kld_bayesian(
                self,
                vi_model=vi_model,
                y=y,
                evaluation_data_type=evaluation_data_type,
                num_processes=num_processes,
                sample_size=sample_size,
                num_points_quad=num_points_quad,
                optimized=optimized,
                copula_only=copula_only,
                min_val=min_val,
                max_val=max_val,
                likelihood_based_metrics=likelihood_based_metrics,
                S_posterior=S_posterior,
                cred_level=cred_level,
            )
        else:
            raise ValueError(f"Inference type {self.inference} not understood or implemented.")

        return self.conditional_independence_table

    # def plot_densities(self, y, x_lim=None, y_lim=None, density_plot=True, storage=None, show_plot=True):
    def plot_densities(
        self,
        y: list | tuple | None,
        x_lim: tuple[float, float] | None = None,
        y_lim: tuple[float, float] | None = None,
        density_plot: bool = True,
        storage: str | None = None,
        show_plot: bool = True,
    ) -> None:
        """
        Plots the densities of the given data for each pair in a grid with optional axis limits and storage options.

        This method creates a density plot of the input data `y` for every pair in the data. It can optionally
        limit the x and y axis ranges, save the plot to storage, and control whether the plot
        is displayed immediately or merely stored.

        Parameters
        ----------
        y : list or tuple or None
            The data to plot densities for.
        x_lim : tuple of float, optional
            Limits for the x-axis as (xmin, xmax). If None, axis limits are determined automatically.
        y_lim : tuple of float, optional
            Limits for the y-axis as (ymin, ymax). If None, axis limits are determined automatically.
        density_plot : bool, optional
            If True, plots a density estimate; if False, plots scatterplot only. Default is True.
        storage : str or None, optional
            Path or filename to save the plot. If None, plot is not saved. Default is None.
        show_plot : bool, optional
            If True, displays the plot immediately. If False, the plot is created but not shown. Default is True.

        Returns
        -------
        None
            The method produces a plot and optionally saves or shows it, but does not return any value.
        """
        plot_densities(
            data=y,
            covariate=False,
            x_lim=x_lim,
            y_lim=y_lim,
            density_plot=density_plot,
            storage=storage,
            show_plot=show_plot,
        )

    def plot_marginals(
        self,
        y: torch.FloatTensor,
        names: bool | list = False,
        y_lim: bool | tuple[float, float] = False,
        storage: str | None = None,
        show_plot: bool = True,
    ) -> None:
        """
        Plots the marginal distributions of the provided data in a grid.

        This method visualizes the marginals of the dataset `y`. It can optionally display
        variable names on the plot, set y-axis limits, save the plot to a specified location,
        and control whether the plot is displayed immediately.

        Parameters
        ----------
        y : torch.FloatTensor
            Data for which the marginal distributions will be plotted.
        names : bool | list = False,
            Whether to display variable names in the plot. Default is False.
        y_lim : bool or tuple of float, optional
            Y-axis limits as (ymin, ymax). If False, axis limits are determined automatically. Default is False.
        storage : str or None, optional
            Path or filename to save the plot. If None, the plot is not saved. Default is None.
        show_plot : bool, optional
            If True, displays the plot immediately. If False, the plot is created but not shown. Default is True.

        Returns
        -------
        None
            The method generates a plot and optionally saves or displays it, but does not return a value.
        """
        plot_marginals(
            y,
            covariate=False,
            names=names,
            y_lim=y_lim,
            storage=storage,
            show_plot=show_plot,
        )

    def plot_splines(
        self,
        layer_type: Literal["transformation", "decorrelation"] = "transformation",
        decorrelation_layer_number: int = 0,
        storage: str | None = None,
        show_plot: bool = True,
    ) -> None:
        """
        Plots all spline functions for a specified model layer in a grid.

        This method visualizes spline functions associated with either the
        transformation layer or one of the decorrelation layers of the model. It selects
        the layer based on `layer_type` and plots the corresponding splines using an internal
        plotting function.

        Parameters
        ----------
        layer_type : {"transformation", "decorrelation"}, optional
            Specifies which layer's splines to plot.
            - "transformation": plots the transformation layer splines.
            - "decorrelation": plots the splines from one of the decorrelation layers.
            Requires `decorrelation_layer_number` to specify which layer.
            Default is "transformation".
        decorrelation_layer_number : int, optional
            Index of the decorrelation layer to plot when `layer_type` is "decorrelation".
            Must be in the range `[0, self.number_decorrelation_layers - 1]`.
            Default is 0.
        storage : str or None, optional
            Path or filename to save the generated plot. If None, the plot is not saved.
            Default is None.
        show_plot : bool, optional
            Whether to display the plot immediately. Default is True.

        Raises
        ------
        ValueError
            If `layer_type` is not one of {"transformation", "decorrelation"}, or
            if `decorrelation_layer_number` is out of valid range.

        Returns
        -------
        None
            The method generates a plot and optionally saves or displays it but does not return a value.
        """

        if layer_type == "transformation":
            layer = self.transformation
        elif layer_type == "decorrelation":
            if decorrelation_layer_number >= self.number_decorrelation_layers:
                raise ValueError(
                    "decorrelation_layer_number exceeds the number of decorrelation layers."
                )
            layer = self.decorrelation_layers[decorrelation_layer_number]
        else:
            raise ValueError(
                "layer_type must be either 'transformation' or 'decorrelation'."
            )

        plot_splines(
            layer,
            covariate_exists=False,
            affine=False,
            storage=storage,
            show_plot=show_plot,
        )

    def plot_conditional_dependence_structure(
        self,
        data,
        conditional_independence_table: bool | None = False,
        dependence_metric_threshholding: bool | str = False,
        minimum_dependence_threshold: float = 0,
        after_marginal_transformation: bool = False,
        show_colorbar: bool = True,
        hide_axis_info: bool = False,
        sub_title_fontsize: int = 10,
        x_lim: tuple | None = None,
        y_lim: tuple | None = None,
        storage: str | None = None,
        show_plot: bool = True,
    ) -> None:
        """
        Plots the conditional dependence structure of each pair of variables in a grid based on pseudo conditional correlations.
        It optionally filters pairs by a threshold applied to a dependence metric, which can be
        specified by a conditional independence table. Thus this metric can be likelihood based as well.

        Parameters
        ----------
        data : array-like or tensor
            The dataset on which to compute and plot the dependence structure.
        conditional_independence_table : bool | pandas.DataFrame | None, optional
            Table indicating conditional independence between variable pairs.
            If False, the method tries to use a stored table `self.conditional_independence_table`.
            Default is False.
        dependence_metric_threshholding : bool | str, optional
            Name of the column in the conditional independence table to threshold on:
            - precision_abs_mean: mean absolute entry in the pseudo-precision matrix
            - precision_square_mean: mean squared entry in the pseudo-precision matrix
            - cond_correlation_abs_mean: mean absolute pseudo-conditional correlation
            - cond_correlation_square_mean: mean squared pseudo-conditional correlation
            - kld: Kullback-Leibler divergence (only if likelihood_based_metrics is True)
            - iae: Integrated Absolute Error (only if likelihood_based_metrics is True)
            If False, no thresholding is applied.
            Default is False.
        minimum_dependence_threshold : float, optional
            Minimum threshold for the dependence metric to include a pair in the plot.
            Default is 0.
        after_marginal_transformation : bool, optional
            If True, plots the dependence structure after applying the marginal transformation.
            Default is False.
        show_colorbar : bool, optional
            Whether to display a colorbar alongside the plot.
            Default is True.
        hide_axis_info : bool, optional
            Whether to hide axis labels and ticks in the plot.
            Default is False.
        sub_title_fontsize : int, optional
            Font size for the plot subtitle.
            Default is 10.
        x_lim : tuple | None, optional
            Limits for the x-axis in the plot (min, max). If None, defaults are used.
            Default is None.
        y_lim : tuple | None, optional
            Limits for the y-axis in the plot (min, max). If None, defaults are used.
            Default is None.
        storage : str | None, optional
            File path or name to save the generated plot. If None, the plot is not saved.
            Default is None.
        show_plot : bool, optional
            Whether to display the plot or not.
            Default is True.

        Returns
        -------
        None
            The method produces a plot and optionally saves or displays it but does not return a value.
        """

        dependence_metric_plotting = "pseudo_conditional_correlation"

        # Taking internally stored one from last run
        if conditional_independence_table is False:
            if self.conditional_independence_table is None:
                conditional_independence_table = False
                if minimum_dependence_threshold > 0:
                    raise ValueError(
                        "No conditional independence table found. Please compute it first using compute_conditional_independence_table() in order to be able to set a minimum_dependence_threshold > 0."
                    )
            else:
                # Use the stored conditional independence table
                conditional_independence_table = self.conditional_independence_table

        if dependence_metric_plotting == "pseudo_conditional_correlation":
            metric = self.compute_pseudo_conditional_correlation_matrix(data)
            metric_type = "matrix"
            label_metric = "Pseudo Conditional Correlation"
        elif dependence_metric_plotting == "offdiagonal_precision_matrix":
            metric = self.compute_pseudo_precision_matrix(data)
            metric_type = "matrix"
            label_metric = "Off-Diagonal Precision Matrix"
        else:
            raise ValueError(
                "Unknown dependence metric. Please use 'pseudo_conditional_correlation' or 'offdiagonal_precision_matrix'."
            )

        if (
            minimum_dependence_threshold > 0
            and conditional_independence_table is not False
            and dependence_metric_threshholding is not False
        ):
            significant_dependence_pairs = []
            threshholding_metric_values = []

            significant_subset = conditional_independence_table[
                conditional_independence_table[dependence_metric_threshholding]
                > minimum_dependence_threshold
            ]
            significant_subset = significant_subset.sort_values(
                dependence_metric_threshholding, ascending=False
            )

            for index, row in significant_subset.iterrows():
                significant_dependence_pairs.append(
                    [int(row["var_col"]), int(row["var_row"])]
                )
                threshholding_metric_values.append(row[dependence_metric_threshholding])
            strength_name = dependence_metric_threshholding
        else:
            significant_dependence_pairs = False
            threshholding_metric_values = False
            strength_name = ""

        if after_marginal_transformation == True:
            data_plotting = self.after_transformation(data).detach()
        else:
            data_plotting = data

        plot_metric_scatter(
            data=data_plotting,
            metric=metric,
            covariate=False,
            x_lim=x_lim,
            y_lim=y_lim,
            metric_type=metric_type,
            pairs=significant_dependence_pairs,
            strength_value=threshholding_metric_values,
            strength_name=strength_name,
            show_colorbar=show_colorbar,
            hide_axis_info=hide_axis_info,
            sub_title_fontsize=sub_title_fontsize,
            after_marginal_transformation=after_marginal_transformation,
            label_metric=label_metric,
            storage=storage,
            show_plot=show_plot,
        )

    def plot_conditional_dependence_graph(
        self,
        conditional_independence_table: bool | None = False,
        dependence_metric: str = "iae",
        minimum_dependence_threshold: float = 0,
        pair_plots: bool = False,
        data: bool | None = False,
        after_marginal_transformation: bool = False,
        names: bool = False,
        lim_axis_pairplots: list[int] = [-18, 18],
        variables_move: list[int] | None = None,
        variables_move_positions: list[tuple[float, float]] | None = None,
        k: float = 1.5,
        seed_graph: int = 42,
        storage: str | None = None,
        show_plot: bool = True,
        scatter_plot_size: float = 1,
    ) -> None:
        """
        Creates and plots a full conditional independence graph using the network package.

        Nodes ("bubbles") represent variables, and undirected edges represent conditional dependencies
        based on the specified dependence metric. Optionally, pair plots of pseudo-conditional correlations
        can be displayed on top of the graph.

        Parameters
        ----------
        conditional_independence_table : bool | pandas.DataFrame | None, optional
            Table providing conditional independence information with dependence metrics as columns.
            If False or None, the internally stored `self.conditional_independence_table` is used.
            Default is False.
        dependence_metric : str, optional
            Column name in the conditional independence table used to threshold edges in the graph.
            Possible values include:
            - "precision_abs_mean": Mean absolute entry in the pseudo-precision matrix.
            - "precision_square_mean": Mean squared entry in the pseudo-precision matrix.
            - "cond_correlation_abs_mean": Mean absolute pseudo-conditional correlation.
            - "cond_correlation_square_mean": Mean squared pseudo-conditional correlation.
            - "kld": Kullback-Leibler divergence (requires likelihood-based metrics).
            - "iae": Integrated Absolute Error (requires likelihood-based metrics).
            Default is "iae".
        minimum_dependence_threshold : float, optional
            Minimum threshold for the dependence metric to include edges in the graph.
            Default is 0.
        pair_plots : bool, optional
            If True, plots of pseudo-conditional correlations for variable pairs are shown above the graph.
            Requires `data` to be provided.
            Default is False.
        data : bool | array-like | None, optional
            Dataset used to compute pseudo-conditional correlations for pair plots.
            Ignored if `pair_plots` is False.
            Default is False.
        after_marginal_transformation : bool, optional
            If True, plots pairplots in space after marginal transformation, hence $\tilde{Z}$ instead of Y.
            Default is False.
        names : bool, optional
            If True, variable names are shown on the graph nodes.
            Default is False.
        lim_axis_pairplots : list[int], optional
            Axis limits [min, max] for the pair plots.
            Default is [-18, 18].
        variables_move : list[int] | None, optional
            List of variable indices ("bubbles") to reposition manually on the graph.
            Default is None.
        variables_move_positions : list[tuple[float, float]] | None, optional
            List of (x, y) coordinates to place the variables in `variables_move`.
            Must be the same length as `variables_move`.
            Default is None.
        k : float, optional
            Repulsion parameter controlling node spacing in the network layout.
            Default is 1.5.
        seed_graph : int, optional
            Random seed for reproducibility of the graph layout.
            Default is 42.
        storage : str | None, optional
            File path to save the generated plot. If None, the plot is not saved.
            Default is None.
        show_plot : bool, optional
            Whether to display the plot interactively.
            Default is True.
        scatter_plot_size : float, optional
            Determiens the size of the overlayed scatterplots, default is 1.
            Depends on then umber of scatterplots and dimensionality of the data.

        Raises
        ------
        ValueError
            If `minimum_dependence_threshold > 0` but no conditional independence table is available,
            or if `variables_move` and `variables_move_positions` lengths differ.

        Returns
        -------
        None
            The function generates and optionally saves and/or displays the conditional dependence graph.
        """
        # Function body here...

        dependence_metric_plotting = "pseudo_conditional_correlation"

        # Taking internally stored one from last run
        if conditional_independence_table is False:
            if self.conditional_independence_table is None:
                raise ValueError(
                    "No conditional independence table found. Please compute it first using compute_conditional_independence_table()."
                )
            else:
                # Use the stored conditional independence table
                conditional_independence_table = self.conditional_independence_table

        if names is False:
            names = list(range(self.number_variables))

        ci_matrix = torch.zeros([self.number_variables, self.number_variables])
        for row in conditional_independence_table.iterrows():
            row = row[1]
            ci_matrix[int(row["var_row"]), int(row["var_col"])] = row[dependence_metric]

        if pair_plots is False:

            plot_graph_conditional_independencies(
                ci_matrix,
                gene_names=names,
                min_abs_mean=minimum_dependence_threshold,
                pos_list=variables_move,
                pos_tuple_list=variables_move_positions,
                k=k,
                seed_graph=seed_graph,
                storage=storage,
                show_plot=show_plot,
            )

        elif pair_plots is True:

            if after_marginal_transformation == True:
                data_plotting = self.after_transformation(data).detach().numpy()
            else:
                data_plotting = data

            if dependence_metric_plotting == "pseudo_conditional_correlation":
                metric = self.compute_pseudo_conditional_correlation_matrix(data)
            elif dependence_metric_plotting == "offdiagonal_precision_matrix":
                metric = self.compute_pseudo_precision_matrix(data)
            else:
                raise ValueError(
                    "Unknown dependence metric. Please use 'pseudo_conditional_correlation' or 'offdiagonal_precision_matrix'."
                )

            plot_graph_conditional_independencies_with_pairplots(
                ci_matrix,
                gene_names=names,
                data=data_plotting,
                metric=metric.numpy(),
                min_abs_mean=minimum_dependence_threshold,
                lim_axis=lim_axis_pairplots,
                pos_list=variables_move,
                pos_tuple_list=variables_move_positions,
                k=k,
                seed_graph=seed_graph,
                storage=storage,
                show_plot=show_plot,
                scatter_plot_size=scatter_plot_size,
            )

    def plot_conditional_dependence_pair(
        self,
        sample_indices: list[int, int],
        resampled_samples: torch.FloatTensor,
        show_colorbar: bool = True,
        title: str | None = None,
        show_ticks: bool = False,
        storage: str | None = None,
        show_plot: bool = True,
    ) -> None:
        """
        Plots conditional dependence between two variables as a 2x2 grid of plots.

        The top row shows the joint density and pseudo conditional correlation
        for the original resampled samples, while the bottom row shows the
        same plots after marginal transformation.

        Parameters
        ----------
        sample_indices : list[int, int]
            Indices of the two variables to plot.
        resampled_samples : Tensor | np.ndarray
            Samples to use for plotting.
        show_colorbar : bool, optional
            Whether to display the colorbar for conditional correlation plots.
            Default is True.
        title : str | None, optional
            Optional title for the figure.
        show_ticks : bool, optional
            Whether to show axis ticks and labels on the plots.
            Default is False (ticks are hidden).
        storage : str | None, optional
            File path to save the figure. If None, the figure is not saved.
            Default is None.
        show_plot : bool, optional
            Whether to display the plot interactively.
            Default is True.

        Returns
        -------
        None
        """

        plot_conditional_dependence_pair(
            loaded_model=self,
            sample_indices=sample_indices,
            resampled_samples=resampled_samples,
            show_colorbar=show_colorbar,
            title=title,
            storage=storage,
            show_ticks=show_ticks,
            show_plot=show_plot,
        )
