# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Schedule for conv2d"""

import tvm


def schedule_conv2d_nhwc(outs):
    """Schedule for conv2d NHWC operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv2d in the format
        of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.te.tensor.Tensor) else outs
    s = tvm.te.create_schedule([x.op for x in outs])
    tvm.te.schedule.AutoInlineInjective(s)
    return s


def schedule_conv2d_nchw(outs):
    return schedule_conv2d_nhwc(outs)


def schedule_conv2d(outs, layout="NHWC"):
    layout_uncase = layout.casefold()
    if layout_uncase == "NHWC".casefold():
        return schedule_conv2d_nhwc(outs)
    if layout_uncase == "NCHW".casefold():
        return schedule_conv2d_nchw(outs)

    raise ValueError(f"Unexpected layout={layout}")


def schedule_depthwise_conv2d_nchw(outs):
    return schedule_conv2d_nchw(outs)


def schedule_depthwise_conv2d_nhwc(out):
    return schedule_conv2d_nhwc(out)
