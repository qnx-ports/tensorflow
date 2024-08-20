load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "versioning",
    srcs = [
        "op_version.cc",
        "runtime_version.cc",
    ],
    hdrs = [
        "op_version.h",
        "runtime_version.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":op_signature",
        "//tensorflow/compiler/mlir/lite/core/c:c_api_types",
        "//tensorflow/compiler/mlir/lite/core/c:tflite_common",
        "//tensorflow/compiler/mlir/lite/kernels/internal:compatibility_macros",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs_with_mutable",
        "//tensorflow/compiler/mlir/lite/schema:schema_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "op_signature",
    srcs = [
        "op_signature.cc",
    ],
    hdrs = [
        "op_signature.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/lite/core/api",
        "//tensorflow/compiler/mlir/lite/core/c:c_api_types",
        "//tensorflow/compiler/mlir/lite/core/c:common",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/compiler/mlir/lite/schema:schema_utils",
        "@flatbuffers//:runtime_cc",
    ],
)
