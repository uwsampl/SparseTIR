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

from typing import Tuple, Optional, List


class Axis:
    def __init__(
        self, name: str, sparse: bool, variable: bool, parent: Optional["Axis"] = None
    ) -> None:
        self.name = name
        self.sparse = sparse
        self.variable = variable
        self.parent = parent

    def __repr__(self) -> str:
        if self.parent is None:
            return "Axis({}, sparse={}, variable={})".format(self.name, self.sparse, self.variable)
        else:
            return "Axis({}, sparse={}, variable={}, parent={})".format(
                self.name, self.sparse, self.variable, self.parent.name
            )


def dense_fixed(name: str) -> Axis:
    return Axis(name, False, False)


def dense_variable(name: str, parent: Axis) -> Axis:
    return Axis(name, False, True, parent)


def sparse_fixed(name: str, parent: Axis) -> Axis:
    return Axis(name, True, False, parent)


def sparse_variable(name: str, parent: Axis) -> Axis:
    return Axis(name, True, True, parent)
