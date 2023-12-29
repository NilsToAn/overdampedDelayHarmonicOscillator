### Code to reproduce results from "Matrix numerical method evaluating probability densities for stochastic delay differential equations"

by Nils Antary (1,2), Klaus Kroy (1), and Viktor Holubec (3)

(1) Institute for Theoretical Physics, University of Leipzig, 04081 Leipzig, Germany

(2) Potsdam Institute for Climate Impact Research (PIK), Member of the Leibniz
Association, 14473 Potsdam, Germany

(3) Charles University, Faculty of Mathematics and Physics, Department of
Macromolecular Physics, CZ-180 00 Praha, Czech Republic

E-mail: nantary@protonmail.com, viktor.holubec@mff.cuni.cz


This directory contains all the necessary code to reproduce all figures and results from the paper "Matrix numerical method evaluating probability densities for stochastic delay differential equations".
The core of the newly proposed method is stored in a separate Python file called `algorithm.py`.
The main logic needed for the analyses is contained in `function.py`.
The code creates the results and generates the figures inside the different notebooks.

To use the algorithm, make sure you have installed the following python3 packages:
- scipy
- NumPy

To run the notebooks, you need to install:
- Jupyter lab or Jupyter Notebook or any other environment that is able to run `.ipynb` files.
- matplotlib
- scipy
- NumPy

We ran the code on a Lenovo IdeaPad 330S Laptop running Ubuntu (18.4 GiB RAM, AMD Ryzen 5 2500U with Radeon Vega Mobile Gfx processor, 4 core, 2GHz), where it consumed up to 16 GiB RAM. If you have less available, you may need to decrease the number of trajectories for the simulation (N_p or N_loop) or the spatial discretization (N_x) for the numerics.

The code is written so that already contained results are saved in the database folder and automatically used when the same analysis is rerun.
