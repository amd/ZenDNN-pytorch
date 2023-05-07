#pragma once

#include <c10/macros/Macros.h>

namespace c10 {
struct StorageImpl;
}; // namespace c10

namespace c10 {
namespace impl {
namespace cow {

auto C10_API materialize(StorageImpl& storage) -> void;

} // namespace cow
} // namespace impl
} // namespace c10
