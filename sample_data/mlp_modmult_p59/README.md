# mod-mult (MLP, p=59) — processed trajectories

Place per-norm-budget npz here as `metrics/longtrain_*rho*.npz`.
Keys required: rho, steps, eff (seeds×T), acc (seeds×T).

Populate from your run dir (PowerShell):

    Copy-Item "<your_run_dir>\metrics\*.npz" "sample_data\mlp_modmult_p59\metrics\"

Then `python reproduce_all.py` analyzes this task and rebuilds its dashboard.
