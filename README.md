To setup this project please follow these steps:

1. run the setup script: `source setup_aver.sh` (source is important to keep venv active outside of the process)
2. when you are ready, run the evaluation: `./run_eval.sh [-m <model>] [-o <output>] [-d <device>] [-b <batch_size>]`

`<model>` is the huggingface path of the model you wish to evaluate eg. `google/mt5-base`

`<output>` is the path to where you wish to receive complete results.

`<device>` denotes the device which will be used for evaluation `[cuda,cpu,mps]`. Anything other than `cuda` is not recommended for full aver evaluation due to very long evaluation

`<batch_size>` determines batch_size used when evaluating (we used `batch_size=16` on the `NVIDIA A40` GPU on the `apollo` server, running locally we used `batch_size=2`), this greatly influences the evaluating speed