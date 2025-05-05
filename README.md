To setup the project please follow these steps:
<h2>
Requirements:
</h2>

<b>Python</b> (3.10 used during development)

pip package installer for python (24.0 used during development outside of venv)

<h2>
Setup and evaluation
</h2>

1. run the setup script: `source setup_aver.sh` (source is important to keep venv active outside of the process) IMPORTANT: in case you are working on a machine that requires everyone to use a venv by default, such as apollo.fi.muni.cz, comment out the 5th and 7th lines in this script
2. when you are ready, run the evaluation: `./run_eval.sh [-m <model>] [-o <output>] [-d <device>] [-b <batch_size>]`

`<model>` is the huggingface path of the model you wish to evaluate eg. `google/mt5-base`

`<output>` is the path to where you wish to receive complete results.

`<device>` denotes the device which will be used for evaluation `[cuda,cpu,mps]`. Anything other than `cuda` is not recommended for full aver evaluation due to very long evaluation

`<batch_size>` determines batch_size used when evaluating (we used `batch_size=16` on the `NVIDIA A40` GPU on the `apollo` server, running locally we used `batch_size=2`), this greatly influences the evaluating speed