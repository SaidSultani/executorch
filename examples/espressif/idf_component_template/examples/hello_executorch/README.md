<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-->

# hello_executorch

Runs a small model through the `executorch` component's `et_model` C API
and checks its outputs. `main/smoke.pte` returns the sum and the product
of two tensors, `main/smoke.io` holds its test inputs and expected
outputs, and `main/smoke_kernels` is its kernel bundle. The app prints
`ET_COMPONENT_SMOKE: PASS` when every output matches.

```sh
idf.py set-target esp32s3
idf.py build flash monitor
```

## How the files were made

The generator made them from the ExecuTorch repository at the commit in
the component's `VERSION`, with the `executorch` package from that commit:

```sh
python examples/espressif/export_smoke_model.py --output smoke.bpte
python examples/espressif/idf_component_smoke/split_bundled_program.py \
    smoke.bpte --atol 0 --rtol 0
python <component>/tools/gen_kernels.py --pte smoke.pte --out main/smoke_kernels
```

The first exports a bundled program, a `.pte` with its test cases. The
second splits it into `smoke.pte` and the `smoke.io` file this app reads.

## Using your own model

Export a `.pte`, make its kernel bundle with `tools/gen_kernels.py`, and
declare both in `main/CMakeLists.txt` with `executorch_add_model` and
`executorch_add_kernels`, in place of `smoke.pte` and `smoke_kernels`.
The `.io` file only feeds this app's self-check; an application passes
its own inputs to `et_model`.
