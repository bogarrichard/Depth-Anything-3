# Third-party notices

This project is licensed under Apache-2.0 (see `LICENSE`). It also contains
code adapted from other open-source projects under their own licenses,
reproduced below as required by those licenses.

## PyTorch3D (BSD License)

`src/depth_anything_3/utils/geometry.py` -- the functions `quat_to_mat`,
`mat_to_quat`, `_sqrt_positive_part` and `standardize_quaternion` -- is
adapted from PyTorch3D's `pytorch3d/transforms/rotation_conversions.py`.

```
BSD License

For PyTorch3D software

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Meta nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## DINOv2 (Apache License 2.0)

`src/depth_anything_3/model/dinov2/` is adapted from Meta's DINOv2, also
under Apache License 2.0 (compatible with, but a separate copyright holder
from, this project's own Apache-2.0 license). Per-file headers already carry
`Copyright (c) Meta Platforms, Inc. and affiliates.` notices; listed here for
completeness.
