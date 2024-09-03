// RUN: mlir_fusions_opt %s -split-input-file -xla-gpu-expand-float-ops="pre-ampere=true" -canonicalize | FileCheck %s
// RUN: mlir_fusions_opt %s -split-input-file -xla-gpu-expand-float-ops="pre-ampere=false" -canonicalize | FileCheck %s
// RUN: mlir_fusions_opt %s -split-input-file -xla-gpu-expand-float-ops="pre-ampere=false has-f8-cvt=true" -canonicalize | FileCheck %s -check-prefixes=CHECK-F8-CVT

module {
  func.func @tanh(%arg0: f32) -> f32 {
    %ret = math.tanh %arg0 : f32
    return %ret : f32
  }
}

// CHECK-LABEL: @tanh
// CHECK-NOT: tanh

// -----

module {
  func.func @erf(%arg0: f32) -> f32 {
    %ret = math.erf %arg0 : f32
    return %ret : f32
  }
}

// CHECK-LABEL: @erf
// CHECK-NOT: erf

// -----

module {
  func.func @maximumf(%arg0: f32, %arg1: f32) -> f32 {
    %ret = arith.maximumf %arg0, %arg1 : f32
    return %ret : f32
  }
}

// CHECK-LABEL: @maximumf
// CHECK: arith.maximumf

// -----

module {
  func.func @minimumf(%arg0: f32, %arg1: f32) -> f32 {
    %ret = arith.minimumf %arg0, %arg1 : f32
    return %ret : f32
  }
}

// CHECK-LABEL: @minimumf
// CHECK: arith.minimumf

// -----

module {
  func.func @minimumf64(%arg0: f64, %arg1: f64) -> f64 {
    %ret = arith.minimumf %arg0, %arg1 : f64
    return %ret : f64
  }
}

// CHECK-LABEL: @minimumf64
// CHECK: arith.minimumf

// -----

module {
  func.func @cmpif8(%arg0: f8E5M2, %arg1: f8E5M2) -> i1 {
    %ret = arith.cmpf une, %arg0, %arg1 : f8E5M2
    return %ret : i1
  }
}

// Just check that this lowers successfully. We have integration tests to verify
// correctness.
// CHECK-LABEL: @cmpif8
// CHECK-NOT: arith.cmpf une{{.*}}f8E5M2

// -----

module {
  func.func @fptoi8(%arg0: f8E5M2) -> i32 {
    %ret = arith.fptosi %arg0 : f8E5M2 to i32
    return %ret : i32
  }
}

// Just check that this lowers successfully. We have integration tests to verify
// correctness.
// CHECK-LABEL: @fptoi8
// CHECK-NOT: arith.fptosi {{.*}}f8E5M2

// -----

module {
  func.func @double_to_f8(%arg0: f64) -> f8E5M2FNUZ {
    %ret = arith.truncf %arg0 : f64 to f8E5M2FNUZ
    return %ret : f8E5M2FNUZ
  }
}

// Just check that this lowers successfully. We have integration tests to verify
// correctness.
// CHECK-LABEL: @double_to_f8
// CHECK-NOT: arith.truncf

// -----

module {
  func.func @bf16_to_f8(%arg0: bf16) -> f8E5M2 {
    %ret = arith.truncf %arg0 : bf16 to f8E5M2
    return %ret : f8E5M2
  }
}

// Verify that we go through f32/f16. We have integration tests to verify
// correctness.
// CHECK-LABEL: @bf16_to_f8
// CHECK: %[[EXT:.*]] = arith.extf {{.*}} : bf16 to f32
// CHECK: arith.truncf %[[EXT]] : f32 to f16
// CHECK-NOT: arith.truncf

// -----

module {
  func.func @intr_f16_to_f8(%arg0: f16) -> (f8E4M3FN, f8E5M2) {
    %a = arith.truncf %arg0 : f16 to f8E4M3FN
    %b = arith.truncf %arg0 : f16 to f8E5M2
    return %a, %b : f8E4M3FN, f8E5M2
  }
}

// CHECK-LABEL: @intr_f16_to_f8
// CHECK-F8-CVT: llvm.nvvm.f16x2.to.e4m3x2.rn
// CHECK-F8-CVT: llvm.nvvm.f16x2.to.e5m2x2.rn

// -----

module {
  func.func @intr_bf16_to_f8(%arg0: bf16) -> (f8E4M3FN, f8E5M2) {
    %a = arith.truncf %arg0 : bf16 to f8E4M3FN
    %b = arith.truncf %arg0 : bf16 to f8E5M2
    return %a, %b : f8E4M3FN, f8E5M2
  }
}

// CHECK-LABEL: @intr_bf16_to_f8
// CHECK-F8-CVT: arith.extf %{{.+}} : bf16 to f32
// CHECK-F8-CVT: llvm.nvvm.ff.to.e4m3x2.rn
// CHECK-F8-CVT: llvm.nvvm.ff.to.e5m2x2.rn

// -----

module {
  func.func @intr_f32_to_f8(%arg0: f32) -> (f8E4M3FN, f8E5M2) {
    %a = arith.truncf %arg0 : f32 to f8E4M3FN
    %b = arith.truncf %arg0 : f32 to f8E5M2
    return %a, %b : f8E4M3FN, f8E5M2
  }
}

// CHECK-LABEL: @intr_f32_to_f8
// CHECK-F8-CVT: llvm.nvvm.ff.to.e4m3x2.rn
// CHECK-F8-CVT: llvm.nvvm.ff.to.e5m2x2.rn

// -----

module {
  func.func @intr_f64_to_f8(%arg0: f64) -> (f8E4M3FN, f8E5M2) {
    %a = arith.truncf %arg0 : f64 to f8E4M3FN
    %b = arith.truncf %arg0 : f64 to f8E5M2
    return %a, %b : f8E4M3FN, f8E5M2
  }
}

// CHECK-LABEL: @intr_f64_to_f8
// CHECK-F8-CVT: arith.truncf %{{.+}} : f64 to f32
// CHECK-F8-CVT: llvm.nvvm.ff.to.e4m3x2.rn
// CHECK-F8-CVT: llvm.nvvm.ff.to.e5m2x2.rn

// -----

module {
  func.func @intr_f8_to_f16(%arg0: f8E4M3FN, %arg1: f8E5M2) -> (f16, f16) {
    %a = arith.extf %arg0 : f8E4M3FN to f16
    %b = arith.extf %arg1 : f8E5M2 to f16
    return %a, %b : f16, f16
  }
}

// CHECK-LABEL: @intr_f8_to_f16
// CHECK-F8-CVT: llvm.nvvm.e4m3x2.to.f16x2.rn
// CHECK-F8-CVT: llvm.nvvm.e5m2x2.to.f16x2.rn

// -----

module {
  func.func @intr_f8_to_bf16(%arg0: f8E4M3FN, %arg1: f8E5M2) -> (bf16, bf16) {
    %a = arith.extf %arg0 : f8E4M3FN to bf16
    %b = arith.extf %arg1 : f8E5M2 to bf16
    return %a, %b : bf16, bf16
  }
}

// CHECK-LABEL: @intr_f8_to_bf16
// CHECK-F8-CVT: llvm.nvvm.e4m3x2.to.f16x2.rn
// CHECK-F8-CVT: llvm.nvvm.e5m2x2.to.f16x2.rn
// CHECK-F8-CVT: arith.extf %{{.+}} : f16 to f32
// CHECK-F8-CVT: arith.truncf %{{.+}} : f32 to bf16

// -----

module {
  func.func @intr_f8_to_f32(%arg0: f8E4M3FN, %arg1: f8E5M2) -> (f32, f32) {
    %a = arith.extf %arg0 : f8E4M3FN to f32
    %b = arith.extf %arg1 : f8E5M2 to f32
    return %a, %b : f32, f32
  }
}

// CHECK-LABEL: @intr_f8_to_f32
// CHECK-F8-CVT: llvm.nvvm.e4m3x2.to.f16x2.rn
// CHECK-F8-CVT: llvm.nvvm.e5m2x2.to.f16x2.rn
// CHECK-F8-CVT: arith.extf %{{.+}} : f16 to f32

// -----

module {
  func.func @intr_f8_to_f8(%arg0: f8E4M3FN) -> f8E5M2 {
    %tmp = arith.extf %arg0 : f8E4M3FN to f16
    %res = arith.truncf %tmp : f16 to f8E5M2
    return %res : f8E5M2
  }
}

// CHECK-LABEL: @intr_f8_to_f8
// CHECK-F8-CVT: llvm.nvvm.e4m3x2.to.f16x2.rn
// CHECK-F8-CVT: llvm.nvvm.f16x2.to.e5m2x2.rn

// -----

module {
  func.func @intr_f16_to_f8_fix_infinity(%arg0: f16) -> f8E5M2 {
    %res = arith.truncf %arg0 : f16 to f8E5M2
    return %res : f8E5M2
  }
}

// CHECK-LABEL: @intr_f16_to_f8_fix_infinity
// CHECK-F8-CVT: %[[PAIR:.*]] = llvm.call_intrinsic "llvm.nvvm.f16x2.to.e5m2x2.rn"
// CHECK-F8-CVT: %[[RES:.*]] = llvm.trunc %[[PAIR]] : i16 to i8
// CHECK-F8-CVT: %[[INT:.*]] = arith.bitcast %arg0 : f16 to i16
// CHECK-F8-CVT: %[[VAL:.*]] = arith.andi %[[INT]], %c32767_i16
// CHECK-F8-CVT: %[[LOWER:.*]] = arith.cmpi ugt, %[[VAL]], %c31615_i16
// CHECK-F8-CVT: %[[UPPER:.*]] = arith.cmpi ule, %[[VAL]], %c31744_i16
// CHECK-F8-CVT: %[[ISINF:.*]] = arith.andi %[[LOWER]], %[[UPPER]]
// CHECK-F8-CVT: arith.select %[[ISINF]], {{.*}}, %[[RES]]

// -----

module {
  func.func @intr_f32_to_f8_fix_infinity(%arg0: f32) -> f8E4M3FN {
    %res = arith.truncf %arg0 : f32 to f8E4M3FN
    return %res : f8E4M3FN
  }
}

// CHECK-LABEL: @intr_f32_to_f8_fix_infinity
// CHECK-F8-CVT: %[[PAIR:.*]] = llvm.call_intrinsic "llvm.nvvm.ff.to.e4m3x2.rn"
// CHECK-F8-CVT: %[[RES:.*]] = llvm.trunc %[[PAIR]] : i16 to i8
// CHECK-F8-CVT: %[[INT:.*]] = arith.bitcast %arg0 : f32 to i32
// CHECK-F8-CVT: %[[VAL:.*]] = arith.andi %[[INT]], %c2147483647_i32
// CHECK-F8-CVT: %[[LOWER:.*]] = arith.cmpi ugt, %[[VAL]], %c1139277824_i32
// CHECK-F8-CVT: %[[UPPER:.*]] = arith.cmpi ule, %[[VAL]], %c2139095040_i32
// CHECK-F8-CVT: %[[ISINF:.*]] = arith.andi %[[LOWER]], %[[UPPER]]
// CHECK-F8-CVT: arith.select %[[ISINF]], {{.*}}, %[[RES]]
