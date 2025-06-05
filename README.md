# Graphical Transformation Model

Graphical Transformation Models (GTMs) model multivariate data with intricate marginals and complex dependency structures non-parametrically, while maintaining interpretability through the identification of conditional independencies. GTMs offering two major advantages. Firstly, GTMs can capture complex nonlinear interdependencies as well as arbitrary marginals. Secondly, GTMs can be lasso regularized towards pairwise conditional independencies, akin to Gaussian graphical models. 

Read the full paper on [arXiv](https://arxiv.org/abs/2503.17845).

## 🚀 Overview

- What is the purpose of this repository?
- What is the model doing or solving?
- What are the key features or contributions?

## 🧠 Background

- Short summary of the research context (problem, hypothesis, related work).
- Link to the related paper or preprint (if available).
- Any useful conceptual diagrams (optional).

## 📦 Repository Structure


## 📥 Installation

```bash
# Clone the repository
git clone https://github.com/your_username/your_model_repo.git
cd your_model_repo

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🏋️‍♂️ Training

## 📈 Evaluation

## 🎬 Demos

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

See the [LICENSE](LICENSE) file for details.

## ToDos:
- clean code with automatic software for that

## Scalability

We run our model training with an Adam optimizer on 100 Dimensional R-Vine with 10000 observations in both the training and validation sets on one hydra GPU.
Using Bernstein Polynomials of degree 10 for both the transformation and the decorrelation layers we take around 1/6 seconds per iteration in the pretraining and 1.05 second per iteration in the joint training.
Using Bsplines Polynomials of degree 15 for the transformation and of degree 20 for the decorrelation layers we take around 0.12 seconds per iteration in the pretraining and 2.5 seconds per iteration in the joint training. 
Due to the deBoor implementation the Bspline training is not so much affected by the degree of the splines compared to the bernstein polynomials.
Using LBFGS for Bernstein Polynomials of degree 10 for both the transformation and the decorrelation layers we take around 1/3 seconds per iteration in the pretraining and 2.95 second per iteration in the joint training.
Using LBFGS for Bsplines Polynomials of degree 15 for the transformation and of degree 20 for the decorrelation layers we take around 1/3 seconds per iteration in the pretraining and 6.8 seconds per iteration in the joint training. 