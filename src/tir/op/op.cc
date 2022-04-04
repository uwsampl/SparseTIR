/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tir/op/op.cc
 *
 *  Common operator definitions for ops in tir/op.h
 */

#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include <cmath>
// Centralized header for constant folders.
#include "../../arith/const_fold.h"
#include "../../target/datatype/registry.h"

namespace tvm {

using namespace tir;

// macro to register an unary op
#define TIR_REGISTER_PURE_UNARY_OP(OpName)                             \
  TVM_REGISTER_OP(OpName).set_num_inputs(1).set_attr<TCallEffectKind>( \
      "TCallEffectKind", Integer(CallEffectKind::kPure))

// macro to register an binary op
#define TIR_REGISTER_PURE_BINARY_OP(OpName)                            \
  TVM_REGISTER_OP(OpName).set_num_inputs(2).set_attr<TCallEffectKind>( \
      "TCallEffectKind", Integer(CallEffectKind::kPure))

runtime::DataType GetRuntimeDataType(const Type& type) {
  if (auto* n = type.as<PrimTypeNode>()) {
    return n->dtype;
  } else if (type.as<PointerTypeNode>()) {
    return DataType::Handle();
  } else if (IsVoidType(type)) {
    return DataType::Void();
  } else {
    LOG(FATAL) << "Type " << type << " does not have a corresponding runtime::DataType";
    return DataType::Handle();
  }
}

Type GetType(const PrimExpr& expr) {
  // TODO(tqchen): add recursive type inference for Call here
  // once we introduced the corresponding fields to the IR.
  if (auto* ptr = expr.as<tir::VarNode>()) {
    // If Var has a more refined type annotation,
    // return the type anotation
    if (ptr->type_annotation.defined()) {
      return ptr->type_annotation;
    }
  }
  // Default: return the type indicated by the dtype.
  runtime::DataType dtype = expr.dtype();
  if (dtype.is_void()) {
    return VoidType();
  }
  return PrimType(dtype);
}

// LargeUIntImm
PrimExpr LargeUIntImm(DataType t, int64_t low, int64_t high, Span span) {
  return tir::Call(
      t, tir::builtin::large_uint_imm(),
      {make_const(DataType::UInt(32), low, span), make_const(DataType::UInt(32), high, span)},
      span);
}

// Q-multiplication
PrimExpr q_multiply_shift(PrimExpr x, PrimExpr y, PrimExpr q, PrimExpr s, Span span) {
  return tir::Call(DataType::Int(32, x.dtype().lanes()), tir::builtin::q_multiply_shift(),
                   {x, y, q, s}, span);
}

// The public function with a quick checking path.
void BinaryOpMatchTypes(PrimExpr& lhs, PrimExpr& rhs, Span span) {  // NOLINT(*)
  if (lhs.dtype() == rhs.dtype()) return;
  DataType ltype = lhs.dtype();
  DataType rtype = rhs.dtype();
  if (ltype.lanes() == 1 && rtype.lanes() != 1) {
    lhs = tir::Broadcast(lhs, rtype.lanes());
  } else if (rtype.lanes() == 1 && ltype.lanes() != 1) {
    rhs = tir::Broadcast(rhs, ltype.lanes());
  } else {
    ICHECK(ltype.lanes() == rtype.lanes()) << "Cannot match type " << ltype << " vs " << rtype;
  }
  if (lhs.dtype() == rhs.dtype()) return;

  ltype = lhs.dtype();
  rtype = rhs.dtype();
  // We keep dtypes conversion to be relatively consistent to reduce the amount code generated by
  // operators. This can be helpful for users to find potential type conversion problems. The
  // following are exceptions:
  if (ltype.is_float() && rtype.is_float()) {
    // Given two dissimilar floats, cast the lower bit version to the higher bit version.
    // E.g. fp16 + fp32 --> fp32 + fp32
    if (ltype.bits() < rtype.bits()) {
      lhs = cast(rtype, lhs);
    } else {
      rhs = cast(ltype, rhs);
    }
  } else if (!ltype.is_float() &&
             (rtype.is_float() || datatype::Registry::Global()->GetTypeRegistered(rtype.code()))) {
    // Cast int->float when the other operand is a float
    lhs = cast(rtype, lhs);
  } else if ((ltype.is_float() || datatype::Registry::Global()->GetTypeRegistered(ltype.code())) &&
             !rtype.is_float()) {
    // Cast int->float when the other operand is a float
    rhs = cast(ltype, rhs);
  } else if (!ltype.is_bfloat16() &&
             (rtype.is_bfloat16() ||
              datatype::Registry::Global()->GetTypeRegistered(rtype.code()))) {
    // Cast int->bfloat16 when the other operand is a bfloat16
    lhs = cast(rtype, lhs);
  } else if ((ltype.is_bfloat16() ||
              datatype::Registry::Global()->GetTypeRegistered(ltype.code())) &&
             !rtype.is_bfloat16()) {
    // Cast int->bfloat16 when the other operand is a bfloat16
    rhs = cast(ltype, rhs);
  } else if ((ltype.is_int() && rtype.is_int()) || (ltype.is_uint() && rtype.is_uint())) {
    // Promote int to higher bits e.g. int8 + int16 --> int16 + int16
    if (ltype.bits() < rtype.bits()) {
      lhs = cast(rtype, lhs);
    } else {
      rhs = cast(ltype, rhs);
    }
  } else if ((ltype.is_int() && rtype.is_uint()) || (ltype.is_uint() && rtype.is_int())) {
    // Handle mixing signed and unsigned integers
    if (ltype.bits() < rtype.bits()) {
      lhs = cast(rtype, lhs);
    } else if (ltype.bits() > rtype.bits()) {
      rhs = cast(ltype, rhs);
    } else {
      // The width of signed and unsigned integers is same.
      if (ltype.is_uint()) {
        rhs = cast(ltype, rhs);
      } else {
        lhs = cast(rtype, lhs);
      }
    }
  } else {
    LOG(FATAL) << "Cannot match type " << ltype << " vs " << rtype;
  }
}

PrimExpr ret(PrimExpr value, Span span) {
  return tir::Call(value.dtype(), tir::builtin::ret(), {value}, span);
}

// maximum and min limits
PrimExpr max_value(const DataType& dtype, Span span) {
  using namespace tir;
  ICHECK_EQ(dtype.lanes(), 1);
  if (dtype.is_int()) {
    if (dtype.bits() == 64) {
      return IntImm(dtype, std::numeric_limits<int64_t>::max(), span);
    } else if (dtype.bits() < 64) {
      int64_t val = 1;
      val = (val << (dtype.bits() - 1)) - 1;
      return IntImm(dtype, val, span);
    }
  } else if (dtype.is_uint()) {
    if (dtype.bits() == 64) {
      return make_const(dtype, std::numeric_limits<uint64_t>::max(), span);
    } else if (dtype.bits() < 64) {
      uint64_t val = 1;
      val = (val << static_cast<uint64_t>(dtype.bits())) - 1;
      return IntImm(dtype, static_cast<int64_t>(val), span);
    }
  } else if (dtype.is_float()) {
    if (dtype.bits() == 64) {
      return FloatImm(dtype, std::numeric_limits<double>::max(), span);
    } else if (dtype.bits() == 32) {
      return FloatImm(dtype, std::numeric_limits<float>::max(), span);
    } else if (dtype.bits() == 16) {
      return FloatImm(dtype, 65504.0, span);
    }
  } else if (dtype.is_bfloat16()) {
    return FloatImm(dtype, std::numeric_limits<float>::max(), span);
  }
  LOG(FATAL) << "Cannot decide max_value for type" << dtype;
  return PrimExpr();
}

PrimExpr min_value(const DataType& dtype, Span span) {
  using namespace tir;
  ICHECK_EQ(dtype.lanes(), 1);
  if (datatype::Registry::Global()->GetTypeRegistered(dtype.code())) {
    // TODO(tkonolige): need to convert all registered min functions to use the span.
    auto f = datatype::GetMinFunc(dtype.code());
    ICHECK(f) << "No minimum function registered for custom dtype " << (unsigned int)dtype.code();
    // TODO(@hypercubestart) Document this change (and others associated with the overflowing
    // floatimm min bug)
    return (*f)(dtype.bits());
  } else if (dtype.is_int()) {
    if (dtype.bits() == 64) {
      return IntImm(dtype, std::numeric_limits<int64_t>::lowest(), span);
    } else if (dtype.bits() < 64) {
      int64_t val = 1;
      val = -(val << (dtype.bits() - 1));
      return IntImm(dtype, val, span);
    }
  } else if (dtype.is_uint()) {
    return IntImm(dtype, 0, span);
  } else if (dtype.is_float()) {
    if (dtype.bits() == 64) {
      return FloatImm(dtype, std::numeric_limits<double>::lowest(), span);
    } else if (dtype.bits() == 32) {
      return FloatImm(dtype, std::numeric_limits<float>::lowest(), span);
    } else if (dtype.bits() == 16) {
      return FloatImm(dtype, -65504.0, span);
    }
  } else if (dtype.is_bfloat16()) {
    return FloatImm(dtype, std::numeric_limits<float>::lowest(), span);
  }
  LOG(FATAL) << "Cannot decide min_value for type" << dtype;
  return PrimExpr();
}

// infinity
PrimExpr infinity(const DataType& dtype, Span span) {
  using namespace tir;
  ICHECK_EQ(dtype.lanes(), 1);
  if (dtype.is_float()) {
    if (dtype.bits() == 64) {
      return FloatImm(dtype, std::numeric_limits<double>::infinity(), span);
    } else if (dtype.bits() == 32 || dtype.bits() == 16) {
      return FloatImm(dtype, std::numeric_limits<float>::infinity(), span);
    }
  }
  LOG(FATAL) << "Cannot decide infinity for type " << dtype;
  return PrimExpr();
}

namespace tir {
template <typename ValueType>
inline bool ConstPowerHelper(ValueType val, int* shift) {
  if (val <= 0) return false;
  shift[0] = 0;
  while (val != 0) {
    if (val & 1) {
      return (val == 1);
    }
    ++shift[0];
    val = val >> 1;
  }
  return true;
}

bool is_const_power_of_two_integer(const PrimExpr& x, int* shift) {
  if (const auto* op = x.as<tir::IntImmNode>()) {
    return ConstPowerHelper(op->value, shift);
  } else {
    return false;
  }
}
}  // namespace tir

PrimExpr cast(const DataType& t, PrimExpr value, Span span) {
  using tir::FloatImmNode;
  if (value.dtype() == t) return value;
  // const fold IntImm as they are used in index computations
  if (t.lanes() == 1) {
    if (const IntImmNode* op = value.as<IntImmNode>()) {
      return make_const(t, op->value, op->span);
    } else if (const FloatImmNode* op = value.as<FloatImmNode>()) {
      return make_const(t, op->value, op->span);
    }
    ICHECK(!value.dtype().is_handle()) << "Can't cast a handle to other types.";
    return tir::Cast(t, value, span);
  } else {
    if (value.dtype().lanes() == 1) {
      // manually unroll cast
      DataType vtype = t.element_of();
      if (value.dtype() != vtype) {
        if (const IntImmNode* op = value.as<IntImmNode>()) {
          value = make_const(vtype, op->value, op->span);
        } else if (const FloatImmNode* op = value.as<FloatImmNode>()) {
          value = make_const(vtype, op->value, op->span);
        } else {
          value = tir::Cast(vtype, value, span);
        }
      }
      return tir::Broadcast(value, t.lanes(), span);
    } else {
      ICHECK(value.dtype().lanes() == t.lanes());
      return tir::Cast(t, value, span);
    }
  }
}

// reinterpret
PrimExpr reinterpret(const DataType& t, PrimExpr value, Span span) {
  if (value.dtype() == t) return value;
  return tir::Call(t, tir::builtin::reinterpret(), {value}, span);
}

// operator+
PrimExpr operator+(PrimExpr a, PrimExpr b) { return add(a, b); }

PrimExpr add(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  PrimExpr ret = arith::TryConstFold<tir::Add>(a, b);
  if (ret.defined()) return ret;
  return tir::Add(a, b, span);
}

// negation
PrimExpr operator-(PrimExpr a) { return neg(a); }

PrimExpr neg(PrimExpr a, Span span) {
  using tir::FloatImmNode;
  using tir::IntImmNode;
  const IntImmNode* pa = a.as<IntImmNode>();
  const FloatImmNode* fa = a.as<FloatImmNode>();
  if (pa) return IntImm(a.dtype(), -pa->value, span);
  if (fa) return FloatImm(a.dtype(), -fa->value, span);
  return make_zero(a.dtype(), span) - a;
}

PrimExpr operator-(PrimExpr a, PrimExpr b) { return sub(a, b); }

PrimExpr sub(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  PrimExpr ret = arith::TryConstFold<tir::Sub>(a, b);
  if (ret.defined()) return ret;
  return tir::Sub(a, b, span);
}

PrimExpr operator*(PrimExpr a, PrimExpr b) { return mul(a, b); }
PrimExpr mul(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  PrimExpr ret = arith::TryConstFold<tir::Mul>(a, b);
  if (ret.defined()) return ret;
  return tir::Mul(a, b, span);
}

PrimExpr div(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  PrimExpr ret = arith::TryConstFold<tir::Div>(a, b);
  if (ret.defined()) return ret;
  return tir::Div(a, b, span);
}

PrimExpr truncdiv(PrimExpr a, PrimExpr b, Span span) {
  ICHECK(a.dtype().is_int() || a.dtype().is_uint()) << a;
  ICHECK(b.dtype().is_int() || b.dtype().is_uint()) << b;
  return div(a, b, span);
}

PrimExpr truncmod(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  PrimExpr ret = arith::TryConstFold<tir::Mod>(a, b);
  if (ret.defined()) return ret;
  return tir::Mod(a, b, span);
}

PrimExpr operator/(PrimExpr a, PrimExpr b) { return div(a, b); }

PrimExpr operator%(PrimExpr a, PrimExpr b) { return truncmod(a, b); }

// TODO(tqchen): switch to floordiv
PrimExpr indexdiv(PrimExpr a, PrimExpr b, Span span) { return floordiv(a, b, span); }

PrimExpr shapediv(PrimExpr a, PrimExpr b, Span span) { return ceildiv(a, b, span); }

PrimExpr indexmod(PrimExpr a, PrimExpr b, Span span) { return floormod(a, b, span); }

PrimExpr floordiv(PrimExpr a, PrimExpr b, Span span) {
  ICHECK(a.dtype().is_int() || a.dtype().is_uint()) << a;
  ICHECK(b.dtype().is_int() || b.dtype().is_uint()) << b;
  BinaryOpMatchTypes(a, b, span);
  PrimExpr ret = arith::TryConstFold<tir::FloorDiv>(a, b);
  if (ret.defined()) return ret;
  return tir::FloorDiv(a, b, span);
}

PrimExpr ceildiv(PrimExpr a, PrimExpr b, Span span) {
  ICHECK(a.dtype().is_int() || a.dtype().is_uint()) << a;
  ICHECK(b.dtype().is_int() || b.dtype().is_uint()) << b;
  BinaryOpMatchTypes(a, b, span);
  PrimExpr ret = arith::TryConstFold<tir::FloorDiv>(a + b - 1, b);
  if (ret.defined()) return ret;
  return tir::FloorDiv(a + b - 1, b, span);
}

PrimExpr floormod(PrimExpr a, PrimExpr b, Span span) {
  ICHECK(a.dtype().is_int() || a.dtype().is_uint()) << a;
  ICHECK(b.dtype().is_int() || b.dtype().is_uint()) << b;
  BinaryOpMatchTypes(a, b, span);
  PrimExpr ret = arith::TryConstFold<tir::FloorMod>(a, b);
  if (ret.defined()) return ret;
  return tir::FloorMod(a, b, span);
}

PrimExpr min(PrimExpr a, PrimExpr b, Span span) {
  // inf-aware simplificaiton
  using arith::is_neg_inf;
  using arith::is_pos_inf;
  if (is_pos_inf(a)) return b;
  if (is_neg_inf(a)) return a;
  if (is_pos_inf(b)) return a;
  if (is_neg_inf(b)) return b;
  BinaryOpMatchTypes(a, b, span);
  PrimExpr ret = arith::TryConstFold<tir::Min>(a, b);
  if (ret.defined()) return ret;
  return tir::Min(a, b, span);
}

PrimExpr max(PrimExpr a, PrimExpr b, Span span) {
  // inf-aware simplificaiton
  using arith::is_neg_inf;
  using arith::is_pos_inf;
  if (is_pos_inf(a)) return a;
  if (is_neg_inf(a)) return b;
  if (is_pos_inf(b)) return b;
  if (is_neg_inf(b)) return a;
  BinaryOpMatchTypes(a, b, span);
  PrimExpr ret = arith::TryConstFold<tir::Max>(a, b);
  if (ret.defined()) return ret;
  return tir::Max(a, b, span);
}

// if_then_else
PrimExpr if_then_else(PrimExpr cond, PrimExpr true_value, PrimExpr false_value, Span span) {
  ICHECK(cond.dtype() == DataType::Bool(1))
      << "if_then_else only accept the condition to be boolean type.";
  BinaryOpMatchTypes(true_value, false_value, span);
  if (const IntImmNode* op = cond.as<IntImmNode>()) {
    if (op->value != 0) {
      return true_value;
    } else {
      return false_value;
    }
  }

  return tir::Call(true_value.dtype(), tir::builtin::if_then_else(),
                   {cond, true_value, false_value}, span);
}

// likely
PrimExpr likely(PrimExpr cond, Span span) {
  if (is_const_int(cond)) return cond;
  return tir::Call(cond.dtype(), tir::builtin::likely(), {cond}, span);
}

// operator>
PrimExpr operator>(PrimExpr a, PrimExpr b) { return greater(a, b); }
PrimExpr greater(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  PrimExpr ret = arith::TryConstFold<tir::GT>(a, b);
  if (ret.defined()) return ret;
  return tir::GT(a, b, span);
}

PrimExpr operator>=(PrimExpr a, PrimExpr b) { return greater_equal(a, b); }
PrimExpr greater_equal(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  PrimExpr ret = arith::TryConstFold<tir::GE>(a, b);
  if (ret.defined()) return ret;
  return tir::GE(a, b, span);
}

PrimExpr operator<(PrimExpr a, PrimExpr b) { return less(a, b); }
PrimExpr less(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  PrimExpr ret = arith::TryConstFold<tir::LT>(a, b);
  if (ret.defined()) return ret;
  return tir::LT(a, b, span);
}

PrimExpr operator<=(PrimExpr a, PrimExpr b) { return less_equal(a, b); }
PrimExpr less_equal(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  PrimExpr ret = arith::TryConstFold<tir::LE>(a, b);
  if (ret.defined()) return ret;
  return tir::LE(a, b, span);
}

PrimExpr operator==(PrimExpr a, PrimExpr b) { return equal(a, b); }
PrimExpr equal(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  PrimExpr ret = arith::TryConstFold<tir::EQ>(a, b);
  if (ret.defined()) return ret;
  return tir::EQ(a, b, span);
}

PrimExpr operator!=(PrimExpr a, PrimExpr b) { return not_equal(a, b); }
PrimExpr not_equal(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  PrimExpr ret = arith::TryConstFold<tir::NE>(a, b);
  if (ret.defined()) return ret;
  return tir::NE(a, b, span);
}

PrimExpr operator&&(PrimExpr a, PrimExpr b) { return logical_and(a, b); }
PrimExpr logical_and(PrimExpr a, PrimExpr b, Span span) {
  ICHECK(a.dtype().is_bool());
  ICHECK(b.dtype().is_bool());
  PrimExpr ret = arith::TryConstFold<tir::And>(a, b);
  if (ret.defined()) return ret;
  return tir::And(a, b, span);
}

PrimExpr operator||(PrimExpr a, PrimExpr b) { return logical_or(a, b); }
PrimExpr logical_or(PrimExpr a, PrimExpr b, Span span) {
  ICHECK(a.dtype().is_bool());
  ICHECK(b.dtype().is_bool());
  PrimExpr ret = arith::TryConstFold<tir::Or>(a, b);
  if (ret.defined()) return ret;
  return tir::Or(a, b, span);
}

PrimExpr operator!(PrimExpr a) { return logical_not(a); }
PrimExpr logical_not(PrimExpr a, Span span) {
  ICHECK(a.dtype().is_bool());
  PrimExpr ret = arith::TryConstFold<tir::Not>(a);
  if (ret.defined()) return ret;
  return tir::Not(a, span);
}

// shift right
PrimExpr operator>>(PrimExpr a, PrimExpr b) { return right_shift(a, b); }

PrimExpr right_shift(PrimExpr a, PrimExpr b, Span span) {
  ICHECK(a.dtype().is_int() || a.dtype().is_uint());
  ICHECK(b.dtype().is_int() || b.dtype().is_uint());
  BinaryOpMatchTypes(a, b, span);
  TVM_INDEX_CONST_PROPAGATION({
    const DataType& rtype = a.dtype();
    if (pb)
      ICHECK(pb->value >= 0 && pb->value < rtype.bits())
          << "Shift amount must be non-negative and less than " << rtype.bits() << " for type "
          << rtype;
    if (pa && pb) {
      return IntImm(rtype, (pa->value >> pb->value), span);
    }
    if (pb) {
      if (pb->value == 0) return a;
    }
  });

  return tir::Call(a.dtype(), tir::builtin::shift_right(), {a, b}, span);
}

// shift left
PrimExpr operator<<(PrimExpr a, PrimExpr b) { return left_shift(a, b); }
PrimExpr left_shift(PrimExpr a, PrimExpr b, Span span) {
  ICHECK(a.dtype().is_int() || a.dtype().is_uint());
  ICHECK(b.dtype().is_int() || b.dtype().is_uint());
  BinaryOpMatchTypes(a, b, span);
  TVM_INDEX_CONST_PROPAGATION({
    const DataType& rtype = a.dtype();
    if (pb)
      ICHECK(pb->value >= 0 && pb->value < rtype.bits())
          << "Shift amount must be non-negative and less than " << rtype.bits() << " for type "
          << rtype;
    if (pa && pb) return IntImm(rtype, (pa->value << pb->value), span);
    if (pb) {
      if (pb->value == 0) return a;
    }
  });
  return tir::Call(a.dtype(), tir::builtin::shift_left(), {a, b}, span);
}

// bitwise and
PrimExpr operator&(PrimExpr a, PrimExpr b) { return bitwise_and(a, b); }
PrimExpr bitwise_and(PrimExpr a, PrimExpr b, Span span) {
  ICHECK(a.dtype().is_int() || a.dtype().is_uint());
  ICHECK(b.dtype().is_int() || b.dtype().is_uint());
  BinaryOpMatchTypes(a, b, span);
  TVM_INDEX_CONST_PROPAGATION({
    const DataType& rtype = a.dtype();
    if (pa && pb) return IntImm(rtype, (pa->value & pb->value), span);
  });
  return tir::Call(a.dtype(), tir::builtin::bitwise_and(), {a, b}, span);
}

// bitwise_or
PrimExpr operator|(PrimExpr a, PrimExpr b) { return bitwise_or(a, b); }
PrimExpr bitwise_or(PrimExpr a, PrimExpr b, Span span) {
  ICHECK(a.dtype().is_int() || a.dtype().is_uint());
  ICHECK(b.dtype().is_int() || b.dtype().is_uint());
  BinaryOpMatchTypes(a, b, span);
  TVM_INDEX_CONST_PROPAGATION({
    const DataType& rtype = a.dtype();
    if (pa && pb) return IntImm(rtype, (pa->value | pb->value), span);
  });
  return tir::Call(a.dtype(), tir::builtin::bitwise_or(), {a, b}, span);
}

// bitwise_xor
PrimExpr operator^(PrimExpr a, PrimExpr b) { return bitwise_xor(a, b); }
PrimExpr bitwise_xor(PrimExpr a, PrimExpr b, Span span) {
  ICHECK(a.dtype().is_int() || a.dtype().is_uint());
  ICHECK(b.dtype().is_int() || b.dtype().is_uint());
  BinaryOpMatchTypes(a, b, span);
  TVM_INDEX_CONST_PROPAGATION({
    const DataType& rtype = a.dtype();
    if (pa && pb) return IntImm(rtype, (pa->value ^ pb->value), span);
  });
  return tir::Call(a.dtype(), tir::builtin::bitwise_xor(), {a, b}, span);
}

// bitwise_not
PrimExpr operator~(PrimExpr a) { return bitwise_neg(a); }

PrimExpr bitwise_neg(PrimExpr a, Span span) {
  ICHECK(a.dtype().is_int() || a.dtype().is_uint());
  return tir::Call(a.dtype(), tir::builtin::bitwise_not(), {a}, span);
}

TVM_REGISTER_GLOBAL("tir.bitwise_not").set_body_typed([](PrimExpr a, Span span) {
  return bitwise_neg(a, span);
});

// pow
PrimExpr pow(PrimExpr x, PrimExpr y, Span span) {
  BinaryOpMatchTypes(x, y, span);
  ICHECK(x.dtype().is_float()) << "power only applies to float";
  static auto op = Op::Get("tir.pow");
  return tir::Call(x.dtype(), op, {x, y}, span);
}

TIR_REGISTER_PURE_BINARY_OP("tir.pow").set_attr<TVectorizable>("TVectorizable", true);

// abs
PrimExpr abs(PrimExpr x, Span span) {
  if (x.dtype().is_int()) {
    using tir::IntImmNode;
    const IntImmNode* px = x.as<IntImmNode>();
    if (px) {
      return IntImm(x.dtype(), std::abs(px->value), px->span);
    }
    return tir::Select(x >= make_zero(x.dtype()), x, -x, span);
  } else if (x.dtype().is_float()) {
    using tir::FloatImmNode;
    const FloatImmNode* fx = x.as<FloatImmNode>();
    if (fx) {
      return FloatImm(x.dtype(), std::fabs(fx->value), fx->span);
    }
    static auto op = Op::Get("tir.fabs");
    return tir::Call(x.dtype(), op, {x}, span);
  } else if (x.dtype().is_uint()) {
    return x;
  } else {
    LOG(FATAL) << "Data type " << x.dtype()
               << " not supported for absolute op. Skipping absolute op...";
    return x;
  }
}

TIR_REGISTER_PURE_UNARY_OP("tir.fabs").set_attr<TVectorizable>("TVectorizable", true);

// isnan
PrimExpr isnan(PrimExpr x, Span span) {
  DataType t = DataType::Bool(x.dtype().lanes());
  if (x.dtype().is_int() || x.dtype().is_uint()) {
    return make_const(t, false);
  } else if (x.dtype().is_float()) {
    using tir::FloatImmNode;
    const FloatImmNode* fx = x.as<FloatImmNode>();
    if (fx) {
      return make_const(t, std::isnan(fx->value), fx->span);
    }
    static auto op = Op::Get("tir.isnan");
    if (x.dtype().bits() == 16) {
      return tir::Call(t, op, {cast(DataType::Float(32, t.lanes()), std::move(x), span)}, span);
    } else {
      return tir::Call(t, op, {x}, span);
    }
  } else {
    LOG(FATAL) << "Data type " << x.dtype() << " not supported for isnan op. Skipping isnan op...";
    return x;
  }
}

// isinf
PrimExpr isinf(PrimExpr x, Span span) {
  DataType t = DataType::Bool(x.dtype().lanes());
  if (x.dtype().is_int() || x.dtype().is_uint()) {
    return make_const(t, false, span);
  } else if (x.dtype().is_float()) {
    PrimExpr infX = infinity(x.dtype(), span);
    return abs(x, span) == infX && !isnan(x, span);
  } else {
    LOG(FATAL) << "Data type " << x.dtype() << " not supported for finiteness ops. Skipping it...";
    return x;
  }
}

// isfinite
PrimExpr isfinite(PrimExpr x, Span span) { return !isinf(x, span) && !isnan(x, span); }

PrimExpr sum(PrimExpr source, Array<IterVar> rdom, Array<PrimExpr> init, Span span) {
  Var x("x", source.dtype(), span), y("y", source.dtype(), span);
  PrimExpr result = tir::Add(x, y, span);
  PrimExpr identity_element = make_zero(source.dtype(), span);
  tir::CommReducer combiner = tir::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return tir::Reduce(combiner, {source}, rdom, make_const(DataType::Bool(1), true), 0, init, span);
}

PrimExpr all(PrimExpr source, Array<IterVar> rdom, Array<PrimExpr> init, Span span) {
  ICHECK(source.dtype().is_bool());
  Var x("x", source.dtype(), span), y("y", source.dtype());
  PrimExpr result = tir::And(x, y, span);
  PrimExpr identity_element = make_const(source.dtype(), true, span);
  tir::CommReducer combiner = tir::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return tir::Reduce(combiner, {source}, rdom, make_const(DataType::Bool(1), true), 0, init, span);
}

PrimExpr any(PrimExpr source, Array<IterVar> rdom, Array<PrimExpr> init, Span span) {
  ICHECK(source.dtype().is_bool());
  Var x("x", source.dtype(), span), y("y", source.dtype(), span);
  PrimExpr result = tir::Or(x, y, span);
  PrimExpr identity_element = make_const(source.dtype(), false, span);
  tir::CommReducer combiner = tir::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return tir::Reduce(combiner, {source}, rdom, make_const(DataType::Bool(1), true), 0, init, span);
}

PrimExpr max(PrimExpr source, Array<IterVar> rdom, Array<PrimExpr> init, Span span) {
  Var x("x", source.dtype(), span), y("y", source.dtype(), span);
  PrimExpr result = tir::Max(x, y, span);
  PrimExpr identity_element = min_value(source.dtype(), span);
  tir::CommReducer combiner = tir::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return tir::Reduce(combiner, {source}, rdom, make_const(DataType::Bool(1), true), 0, init, span);
}

PrimExpr min(PrimExpr source, Array<IterVar> rdom, Array<PrimExpr> init, Span span) {
  Var x("x", source.dtype(), span), y("y", source.dtype(), span);
  PrimExpr result = tir::Min(x, y, span);
  PrimExpr identity_element = max_value(source.dtype(), span);
  tir::CommReducer combiner = tir::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return tir::Reduce(combiner, {source}, rdom, make_const(DataType::Bool(1), true), 0, init, span);
}

PrimExpr prod(PrimExpr source, Array<IterVar> rdom, Array<PrimExpr> init, Span span) {
  Var x("x", source.dtype(), span), y("y", source.dtype(), span);
  PrimExpr result = tir::Mul(x, y, span);
  PrimExpr identity_element = make_const(source.dtype(), 1, span);
  tir::CommReducer combiner = tir::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return tir::Reduce(combiner, {source}, rdom, make_const(DataType::Bool(1), true), 0, init, span);
}

// fmod
PrimExpr fmod(PrimExpr x, PrimExpr y, Span span) {
  BinaryOpMatchTypes(x, y, span);
  ICHECK(x.dtype().is_float()) << "fmod only applies to float";
  static auto op = Op::Get("tir.fmod");
  return tir::Call(x.dtype(), op, {x, y}, span);
}

TIR_REGISTER_PURE_UNARY_OP("tir.fmod");

// floor
PrimExpr floor(PrimExpr x, Span span) {
  if (x.dtype().is_int() || x.dtype().is_uint()) {
    return x;
  }
  using tir::FloatImmNode;
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.dtype(), std::floor(fx->value), fx->span);
  static auto op = Op::Get("tir.floor");
  return tir::Call(x.dtype(), op, {x}, span);
}

TIR_REGISTER_PURE_UNARY_OP("tir.floor").set_attr<TVectorizable>("TVectorizable", true);

// ceil
PrimExpr ceil(PrimExpr x, Span span) {
  if (x.dtype().is_int() || x.dtype().is_uint()) {
    return x;
  }
  using tir::FloatImmNode;
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.dtype(), std::ceil(fx->value), fx->span);
  static auto op = Op::Get("tir.ceil");
  return tir::Call(x.dtype(), op, {x}, span);
}

TIR_REGISTER_PURE_UNARY_OP("tir.ceil").set_attr<TVectorizable>("TVectorizable", true);

// round
PrimExpr round(PrimExpr x, Span span) {
  if (x.dtype().is_int() || x.dtype().is_uint()) {
    return x;
  }
  using tir::FloatImmNode;
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.dtype(), std::nearbyint(fx->value), fx->span);
  static auto op = Op::Get("tir.round");
  return tir::Call(x.dtype(), op, {x}, span);
}

TIR_REGISTER_PURE_UNARY_OP("tir.round").set_attr<TVectorizable>("TVectorizable", true);

// nearbyint
PrimExpr nearbyint(PrimExpr x, Span span) {
  if (x.dtype().is_int() || x.dtype().is_uint()) {
    return x;
  }
  using tir::FloatImmNode;
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.dtype(), std::nearbyint(fx->value), fx->span);
  static auto op = Op::Get("tir.nearbyint");
  return tir::Call(x.dtype(), op, {x}, span);
}

TIR_REGISTER_PURE_UNARY_OP("tir.nearbyint");

// atomic_add
PrimExpr atomic_add(tir::Var ptr, PrimExpr elem_offset, PrimExpr val, Span span) {
  return tir::Call(val->dtype, builtin::tvm_atomic_add(), {ptr, elem_offset, val}, span);
}

// trunc
PrimExpr trunc(PrimExpr x, Span span) {
  if (x.dtype().is_int() || x.dtype().is_uint()) {
    return x;
  }
  using tir::FloatImmNode;
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) {
    return FloatImm(x.dtype(), (fx->value < 0 ? std::ceil(fx->value) : std::floor(fx->value)),
                    fx->span);
  }
  static auto op = Op::Get("tir.trunc");
  return tir::Call(x.dtype(), op, {x}, span);
}

TIR_REGISTER_PURE_UNARY_OP("tir.trunc").set_attr<TVectorizable>("TVectorizable", true);

// unary op registration.
TIR_REGISTER_PURE_UNARY_OP("tir.exp").set_attr<TVectorizable>("TVectorizable", true);

TIR_REGISTER_PURE_UNARY_OP("tir.exp2").set_attr<TVectorizable>("TVectorizable", true);

TIR_REGISTER_PURE_UNARY_OP("tir.exp10").set_attr<TVectorizable>("TVectorizable", true);

TIR_REGISTER_PURE_UNARY_OP("tir.erf");

TIR_REGISTER_PURE_UNARY_OP("tir.tanh").set_attr<TVectorizable>("TVectorizable", true);

TIR_REGISTER_PURE_UNARY_OP("tir.sigmoid");

TIR_REGISTER_PURE_UNARY_OP("tir.sqrt").set_attr<TVectorizable>("TVectorizable", true);

TIR_REGISTER_PURE_UNARY_OP("tir.rsqrt");

TIR_REGISTER_PURE_UNARY_OP("tir.log").set_attr<TVectorizable>("TVectorizable", true);

TIR_REGISTER_PURE_UNARY_OP("tir.log2").set_attr<TVectorizable>("TVectorizable", true);

TIR_REGISTER_PURE_UNARY_OP("tir.log1p");

TIR_REGISTER_PURE_UNARY_OP("tir.log10").set_attr<TVectorizable>("TVectorizable", true);

TIR_REGISTER_PURE_UNARY_OP("tir.tan").set_attr<TVectorizable>("TVectorizable", true);

TIR_REGISTER_PURE_UNARY_OP("tir.cos").set_attr<TVectorizable>("TVectorizable", true);

TIR_REGISTER_PURE_UNARY_OP("tir.cosh").set_attr<TVectorizable>("TVectorizable", true);

TIR_REGISTER_PURE_UNARY_OP("tir.sin").set_attr<TVectorizable>("TVectorizable", true);

TIR_REGISTER_PURE_UNARY_OP("tir.sinh").set_attr<TVectorizable>("TVectorizable", true);

TIR_REGISTER_PURE_UNARY_OP("tir.asin");

TIR_REGISTER_PURE_UNARY_OP("tir.acos");

TIR_REGISTER_PURE_UNARY_OP("tir.atan");

TIR_REGISTER_PURE_UNARY_OP("tir.acosh");

TIR_REGISTER_PURE_UNARY_OP("tir.asinh");

TIR_REGISTER_PURE_UNARY_OP("tir.atanh");

TIR_REGISTER_PURE_UNARY_OP("tir.clz");

// binary intrinsics
TIR_REGISTER_PURE_BINARY_OP("tir.atan2");

TIR_REGISTER_PURE_BINARY_OP("tir.nextafter");

TIR_REGISTER_PURE_BINARY_OP("tir.hypot");

TIR_REGISTER_PURE_BINARY_OP("tir.copysign");

TIR_REGISTER_PURE_BINARY_OP("tir.ldexp");

// expose basic functions to node namespace
TVM_REGISTER_GLOBAL("node._const").set_body([](TVMArgs args, TVMRetValue* ret) {
  if (args[0].type_code() == kDLInt) {
    *ret = tir::make_const(args[1], args[0].operator int64_t(), args[2]);
  } else if (args[0].type_code() == kDLFloat) {
    *ret = tir::make_const(args[1], args[0].operator double(), args[2]);
  } else {
    LOG(FATAL) << "only accept int or float";  // FIXME
  }
});

TVM_REGISTER_GLOBAL("node.LargeUIntImm").set_body_typed(LargeUIntImm);

TVM_REGISTER_GLOBAL("tir.min_value").set_body_typed(min_value);

TVM_REGISTER_GLOBAL("tir.max_value").set_body_typed(max_value);

TVM_REGISTER_GLOBAL("tir.abs").set_body_typed(tvm::abs);

TVM_REGISTER_GLOBAL("tir.isnan").set_body_typed(tvm::isnan);

TVM_REGISTER_GLOBAL("tir.isfinite").set_body_typed(tvm::isfinite);

TVM_REGISTER_GLOBAL("tir.isinf").set_body_typed(tvm::isinf);

TVM_REGISTER_GLOBAL("tir.floor").set_body_typed(tvm::floor);

TVM_REGISTER_GLOBAL("tir.ceil").set_body_typed(tvm::ceil);

TVM_REGISTER_GLOBAL("tir.round").set_body_typed(tvm::round);

TVM_REGISTER_GLOBAL("tir.nearbyint").set_body_typed(tvm::nearbyint);

TVM_REGISTER_GLOBAL("tir.trunc").set_body_typed(tvm::trunc);

TVM_REGISTER_GLOBAL("tir._cast").set_body_typed(tvm::cast);

TVM_REGISTER_GLOBAL("tir.atomic_add").set_body_typed(tvm::atomic_add);

// operator overloading, smarter than make
#define REGISTER_MAKE_BINARY_OP(Node, Func)                                                \
  TVM_REGISTER_GLOBAL("tir." #Node).set_body_typed([](PrimExpr a, PrimExpr b, Span span) { \
    return (Func(a, b, span));                                                             \
  })

#define REGISTER_MAKE_BIT_OP(Node, Func)                                                \
  TVM_REGISTER_GLOBAL("tir." #Node).set_body([](TVMArgs args, TVMRetValue* ret) {       \
    bool lhs_is_int = args[0].type_code() == kDLInt;                                    \
    bool rhs_is_int = args[1].type_code() == kDLInt;                                    \
    if (lhs_is_int) {                                                                   \
      *ret = (Func(args[0].operator int(), args[1].operator PrimExpr(), args[2]));      \
    } else if (rhs_is_int) {                                                            \
      *ret = (Func(args[0].operator PrimExpr(), args[1].operator int(), args[2]));      \
    } else {                                                                            \
      *ret = (Func(args[0].operator PrimExpr(), args[1].operator PrimExpr(), args[2])); \
    }                                                                                   \
  })

REGISTER_MAKE_BINARY_OP(_OpAdd, add);
REGISTER_MAKE_BINARY_OP(_OpSub, sub);
REGISTER_MAKE_BINARY_OP(_OpMul, mul);
REGISTER_MAKE_BINARY_OP(_OpDiv, div);
REGISTER_MAKE_BINARY_OP(_OpMod, truncmod);
REGISTER_MAKE_BINARY_OP(_OpIndexDiv, indexdiv);
REGISTER_MAKE_BINARY_OP(_OpIndexMod, indexmod);
REGISTER_MAKE_BINARY_OP(_OpFloorDiv, floordiv);
REGISTER_MAKE_BINARY_OP(_OpFloorMod, floormod);
REGISTER_MAKE_BINARY_OP(_OpTruncDiv, truncdiv);
REGISTER_MAKE_BINARY_OP(_OpTruncMod, truncmod);
REGISTER_MAKE_BINARY_OP(_OpPow, pow);
REGISTER_MAKE_BINARY_OP(_OpMin, min);
REGISTER_MAKE_BINARY_OP(_OpMax, max);
REGISTER_MAKE_BINARY_OP(_OpEQ, equal);
REGISTER_MAKE_BINARY_OP(_OpNE, not_equal);
REGISTER_MAKE_BINARY_OP(_OpLT, less);        // NOLINT(*)
REGISTER_MAKE_BINARY_OP(_OpLE, less_equal);  // NOLINT(*)
REGISTER_MAKE_BINARY_OP(_OpGT, greater);     // NOLINT(*)
REGISTER_MAKE_BINARY_OP(_OpGE, greater_equal);
REGISTER_MAKE_BINARY_OP(_OpAnd, logical_and);
REGISTER_MAKE_BINARY_OP(_OpOr, logical_or);
REGISTER_MAKE_BIT_OP(bitwise_and, bitwise_and);
REGISTER_MAKE_BIT_OP(bitwise_or, bitwise_or);
REGISTER_MAKE_BIT_OP(bitwise_xor, bitwise_xor);
REGISTER_MAKE_BIT_OP(left_shift, left_shift);  // NOLINT(*)
REGISTER_MAKE_BIT_OP(right_shift, right_shift);

TVM_REGISTER_GLOBAL("tir._OpIfThenElse")
    .set_body_typed([](PrimExpr cond, PrimExpr true_value, PrimExpr false_value, Span span) {
      return if_then_else(cond, true_value, false_value, span);
    });

TVM_REGISTER_GLOBAL("tir.const_true").set_body_typed([](DataType t, Span span) {
  return const_true(t.lanes(), span);
});

}  // namespace tvm
