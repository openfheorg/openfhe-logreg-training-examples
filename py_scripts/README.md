# Contents

## `step_by_step_training_debugger.ipynb`

The code for our debugger

- A minimal python implementation of the NAG algorithm (verified against the R gold-standard script). Used primarily to
  facilitate step-by-step result checking

- This allows us to run quickly debug the outputs. This isn't meant to plot the results, just as an output checker

## `parameter_search.ipynb`

- Used to run a plaintext search of our hyperparameter space.
  Because we validated our code on the `step_by_step_training_debugger`, we can just
  run plaintext search in-the-clear.

- The generated plots are stored in the `py_scripts/plots` folder (uncomment the corresponding plt.savefig lines). We plot the affect our hyperparameters have on

    - our `AUC-ROC` across the test and train set.

    - convergence (how fast, and if it converged at all)

    - final loss
