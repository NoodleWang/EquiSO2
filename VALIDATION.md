# Release validation

Validated locally on 2026-10-13 using Python 3.10, PyTorch 2.8.0+cu126,
e3nn 0.6.0 and pytest 8.4.2. Computation used CPU with one thread.

- 104 tests passed from the release source tree.
- A wheel built successfully and installed into a temporary target directory
  without installing or modifying the research environment's dependencies.
- All 104 tests also passed against the independently installed package,
  with the source-tree Python path disabled.
- The default model contains exactly 508,640 trainable parameters.
- The inference example returned energy [1], force [3,3] and Hessian [9,9].
- The synthetic EH-soft-F example completed one AdamW step.
- Hessian row batches of 1, 3 and 63 agreed in the small FP64 derivative test.
- Selected original source and test snapshots are recorded in PROVENANCE.json.

These checks validate source extraction and basic numerical behavior. They do
not reproduce full-dataset training, GPU performance, trained-checkpoint
predictions or transition-state searches. The repository does not include
pretrained weights or electronic-structure reference datasets.
