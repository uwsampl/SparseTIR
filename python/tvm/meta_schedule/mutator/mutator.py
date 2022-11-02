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
"""Meta Schedule Mutator."""
from typing import Callable, Optional, TYPE_CHECKING

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.tir.schedule import Trace

from .. import _ffi_api
from ..utils import _get_default_str

if TYPE_CHECKING:
    from ..tune_context import TuneContext


class Mutator(Object):
    """Mutator is designed to mutate the trace to explore the design space."""

    def initialize_with_tune_context(self, context: "TuneContext") -> None:
        """Initialize the mutator with a tune context.

        Parameters
        ----------
        context : TuneContext
            The tuning context for initializing the mutator.
        """
        _ffi_api.MutatorInitializeWithTuneContext(  # type: ignore # pylint: disable=no-member
            self, context
        )

    def apply(self, trace: Trace) -> Optional[Trace]:
        """Apply the mutator function to the given trace.

        Parameters
        ----------
        trace : Trace
            The given trace for mutation.

        Returns
        -------
        trace : Optional[Trace]
            None if mutator failed, otherwise return the mutated trace.
        """
        return _ffi_api.MutatorApply(self, trace, -1)  # type: ignore # pylint: disable=no-member


@register_object("meta_schedule.PyMutator")
class _PyMutator(Mutator):
    """
    A TVM object mutator to support customization on the python side.
    This is NOT the user facing class for function overloading inheritance.

    See also: PyMutator
    """

    def __init__(
        self,
        f_initialize_with_tune_context: Callable = None,
        f_apply: Callable = None,
        f_as_string: Callable = None,
    ):
        """Constructor."""

        self.__init_handle_by_constructor__(
            _ffi_api.MutatorPyMutator,  # type: ignore # pylint: disable=no-member
            f_initialize_with_tune_context,
            f_apply,
            f_as_string,
        )


class PyMutator:
    """
    An abstract mutator with customized methods on the python-side.
    This is the user facing class for function overloading inheritance.

    Note: @derived_object is required for proper usage of any inherited class.
    """

    _tvm_metadata = {
        "cls": _PyMutator,
        "methods": ["initialize_with_tune_context", "apply", "__str__"],
    }

    def initialize_with_tune_context(self, context: "TuneContext") -> None:
        """Initialize the mutator with a tune context.

        Parameters
        ----------
        context : TuneContext
            The tuning context for initializing the mutator.
        """
        raise NotImplementedError

    def apply(self, trace: Trace, _) -> Optional[Trace]:
        """Apply the mutator function to the given trace.

        Parameters
        ----------
        trace : Trace
            The given trace for mutation.

        Returns
        -------
        trace : Optional[Trace]
            None if mutator failed, otherwise return the mutated trace.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """Get the mutator as string with name.

        Return
        ------
        result : str
            Get the mutator as string with name.
        """
        return _get_default_str(self)