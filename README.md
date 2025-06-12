# Graphical Transformation Model

Graphical Transformation Models (GTMs) are a class of probabilistic models designed for multivariate data, capable of:

1. Capturing complex nonlinear interdependencies and arbitrary marginals.

2. Being lasso-regularized towards pairwise conditional independencies, similar to Gaussian graphical models for nonlinear dependencies.

Read the full paper on [arXiv](https://arxiv.org/abs/2503.17845).

## 🎬 Demos 

In this repository, we provide demo scripts to replicate the results from our work. These scripts serve as tutorials and offer a good introduction to applying the model. They can also be used as a starting point for your own data.

#### Simulations

For the simulation studies, we provide the [2D copula demo](demos/demo_2D_copula.ipynb) and the [10D copula demo](demos/demo_10D_rvine.ipynb). These demos illustrate how to train and evaluate the model using a simple 2D copula example, as well as on 10D R-vine data, where we also show how to identify the full conditional independence structure.

#### Application to the MAGIC Dataset

We also provide all the files needed to replicate our analysis on the MAGIC dataset. The [training script](demos/train_gtm_magic.py) is a Python script that runs all models used in the analysis and works on both CPU and CUDA. The choice of marginal degrees, based on a simple heuristic approach, is explained in this [notebook](demos/demo_magic_marginal_transformation_degrees.ipynb). The trained models are already stored, allowing you to directly access the analysis and run the respective notebooks.

The [classification notebook](demos/demo_magic_classification.ipynb) demonstrates how we selected the optimal number of decorrelation layers and how we evaluated model prediction performance. It can also serve as a baseline for creating a classifier based on GTMs. The [Identify Conditional Independence notebook](demos/demo_magic_conditional_independence.ipynb) shows how to analyze conditional dependencies using the GTM and how to utilize the plotting functions to create the graphs featured in the paper. Finally, the [synthetic samples notebook](demos/demo_magic_simulate_synthetic_data.ipynb) illustrates how to sample synthetically from the model and visually compare the learned distribution to the actual training data.

## 📥 Installation

To clone the repository run:
```bash
git clone https://github.com/MatthiasHerp/gtm.git
cd your_model_repo

# (Optional) Create a virtual environment
python -m venv gtm_environment
source gtm_environment/bin/activate  # On Windows: gtm_environment\Scripts\activate
```

Run the following command in the cloned repository root and in your python environment to install the package and its dependencies:

```python
pip install -e .
```


## 🏋️‍♂️ Training

The package is built so that it exports the class `GTM` which contains all required functions as methods.
By using `help(GTM)`on the class or any method such as `help(GTM.train)` you can see the documentation.

Create toy example with a Vine where variable pairs "0,1" as well as "1,2" are connected via Joe copulas and 0 is independent of 2 conditioned on 1.

```python
from gtm import *
import pyvinecopulib as pv
from demos.dataset_helpers import Generic_Dataset

# Define GTM Model
model = GTM(
    number_variables = 2,
    degree_transformations = 10,
    degree_decorrelation = 40,
    spline_transformation = "bspline",
    spline_decorrelation = "bspline",
    transformation_spline_range = (-10, 10),
    device = "cpu") 
# help(GTM)

# Toy Data
D = 3  # dimension
rvine_matrix = np.array([[2, 2, 2],
                         [1, 1, 0],
                         [3, 0, 0]], dtype=np.uint64)
pair_copulas = [[pv.Bicop(family=pv.BicopFamily.joe, parameters=np.array([[3.]]), rotation=0),
                 pv.Bicop(family=pv.BicopFamily.joe, parameters=np.array([[5.]]), rotation=0)],
                [pv.Bicop(family=pv.BicopFamily.indep)]]
vine_model = pv.Vinecop.from_structure(matrix=rvine_matrix, pair_copulas=pair_copulas)
simulated_data_uniform_train = vine_model.simulate(n=2000)
simulated_data_train = torch.distributions.Normal(0,1).icdf(torch.tensor(simulated_data_uniform_train)).float()
simulated_data_uniform_validate = vine_model.simulate(n=2000)
simulated_data_validate = torch.distributions.Normal(0,1).icdf(torch.tensor(simulated_data_uniform_validate)).float()

# Create dataset and DataLoader
dataset_train = Generic_Dataset(simulated_data_train)
dataloader_train = DataLoader(dataset_train, batch_size=N_train)
dataset_validate = Generic_Dataset(simulated_data_validate)
dataloader_validate = DataLoader(dataset_validate, batch_size=N_validate)
```

Run Hyperparameter Tuning for the spline penalties.

```python
study = model.hyperparameter_tune_penalties( 
        train_dataloader = dataloader_train,
        validate_dataloader = dataloader_validate,
        penalty_decorrelation_ridge_param = None,
        penalty_decorrelation_ridge_first_difference = "sample",
        penalty_decorrelation_ridge_second_difference = "sample",
        penalty_transformation_ridge_second_difference = None,
        penalty_lasso_conditional_independence = "sample",
        optimizer="LBFGS",
        n_trials=30,
        temp_folder=".",
        study_name=None)
        
penalty_splines_params=torch.FloatTensor([
                            0, #study.best_params["penalty_decorrelation_ridge_param"],
                            study.best_params["penalty_decorrelation_ridge_first_difference"],
                            study.best_params["penalty_decorrelation_ridge_second_difference"],
                            0 #study.best_params["penalty_transformation_ridge_second_difference"]
                              ])
penalty_lasso_conditional_independence=torch.FloatTensor([study.best_params["penalty_lasso_conditional_independence"]])
```

With the optimal penalties, train the final model with pretraining and then joint training.

```python
# pretrain the marginal transformations
_ = model.pretrain_transformation_layer(dataloader_train)

# train the joint model
_ = model.train(train_dataloader=dataloader_train, validate_dataloader=dataloader_validate, penalty_splines_params=penalty_splines_params,
penalty_lasso_conditional_independence=penalty_lasso_conditional_independence)
```

## 📈 Evaluation

Generate synthetic samples, compute the conditional independence table and use them to plot the local conditional pseudo-correlations to visualize the dependence structure.
The model identifies the correct conditional independence as well as the order of the joe copula strengths.

```python
synthetic_samples = model.sample(10000)

conditional_independence_table = model.compute_conditional_independence_table(
                                        y = None,
                                        evaluation_data_type = "samples_from_model",
                                        num_processes=3,
                                        sample_size = 5000,
                                        num_points_quad=15,
                                        copula_only=False,
                                        min_val=-6,
                                        max_val=6)

model.plot_conditional_dependence_structure(data=synthetic_samples,
                          dependence_metric_threshholding="iae",
                          minimum_dependence_threshold=0.0001, 
                          )
```
![Example Plot](demos/synthetic_plots/readme_condcorr.png)

Plot the resulting conditional independence graph.
You can also compare it to the true structure by plotting the first vine tree.

```python
model.plot_conditional_dependence_graph(
                                        dependence_metric="iae", 
                                        minimum_dependence_threshold=0.05, 
                                        pair_plots=True,
                                        data=synthetic_samples.detach(),
                                        lim_axis_pairplots = [-6, 6], 
                                        scatter_plot_size = 2.5)
# vine_model.plot(tree=[0])
```
![Example Plot](demos/synthetic_plots/readme_graph.png)

## Scalability /Runtime Benchmark

We benchmarked the GTM on a 100-dimensional R-Vine model with 10,000 observations in both the training and validation sets, using a single NVIDIA A40 GPU. Two spline basis types Bernstein polynomials and B-splines were tested under two optimizers (Adam and LBFGS). The table below reports the average runtime per iteration for both the pretraining and joint training phases.

Note: Due to the de Boor implementation, B-spline training is less sensitive to the spline degree compared to Bernstein polynomials.

#### Average Runtime per Iteration

| Optimizer | Basis Type | Degree (Transf. / Decorrel.) | Pretraining Time (s) | Joint Training Time (s) |
|-----------|------------|-------------------------------|-----------------------|--------------------------|
| Adam      | Bernstein  | 10 / 10                       | 0.16                  | 1.05                     |
| Adam      | B-spline   | 15 / 20                       | 0.12                  | 2.50                     |
| LBFGS     | Bernstein  | 10 / 10                       | 0.33                  | 2.95                     |
| LBFGS     | B-spline   | 15 / 20                       | 0.33                  | 6.80                     |



## Contact and Contributing

For questions or collaboration inquiries, please contact: Matthias Herp [matthias.herp@bioinf.med.uni-goettingen.de]

## Citation

If you use this code, please cite:

```bibtex
@misc{gtm,
      title={Graphical Transformation Models}, 
      author={Matthias Herp and Johannes Brachem and Michael Altenbuchinger and Thomas Kneib},
      year={2025},
      eprint={2503.17845},
      archivePrefix={arXiv},
      primaryClass={stat.ME},
      url={https://arxiv.org/abs/2503.17845}, 
}
```

## 🛡️ License

This code is released under a custom Non-Commercial Use License. You are free to use, modify, and distribute the code for non-commercial research and educational purposes.

For commercial use, please contact: Matthias Herp [matthias.herp@bioinf.med.uni-goettingen.de]

See the [License](License.md) file for details.