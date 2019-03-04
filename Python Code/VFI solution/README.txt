# Code to solve the household problem and simulate the economy with method of discrete Value Function Iteration.

model_class.py contains a class to store the main parameters, grids and functions to use in the different scenarios of the model.
VFI_norisk.py solves the hh problem and simulates the economy when agents do not face risk.
VFI_inputs.py solves the hh problem and simulates the economy in the autarky scenario (no access to assets or insurance market).
VFI_insurance.py solves the hh problem and simulates the economy when a index-based insurance market is added to the autarky scenario.
VFI_assets.py solves the hh problem and simulates the economy when a risk-free asset market with borrowing constraint is added to the autarky scenario.
VFI_assets_insurance.py solves the hh problem and simulates the economy when a risk-free asset market and index-based insurance market are added.
