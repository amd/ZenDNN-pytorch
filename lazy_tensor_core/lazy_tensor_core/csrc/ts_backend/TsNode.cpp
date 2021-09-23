#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"
#include "lazy_tensor_core/lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace ir {

Value::Value(TsNodePtr tsnode, size_t index)
    : node(std::dynamic_pointer_cast<Node>(tsnode)), index(index){};

ShapeCache* GetShapeCache() {
  static lazy_tensors::int64 shape_cache_size =
      lazy_tensors::sys_util::GetEnvInt("LTC_IR_SHAPE_CACHE_SIZE", 4096);
  static ShapeCache* cache = new ShapeCache(shape_cache_size);
  return cache;
}
TsNode::TsNode(OpKind op, OpList operands, AtenShape aten_shape,
               c10::ScalarType aten_type, size_t num_outputs,
               lazy_tensors::hash_t hash_seed)
    : Node(std::move(op), operands, aten_shape, aten_type, num_outputs,
           hash_seed) {}
TsNode::TsNode(OpKind op, OpList operands,
               const std::function<lazy_tensors::Shape()>& shape_fn,
               size_t num_outputs, lazy_tensors::hash_t hash_seed)
    : Node(std::move(op), operands, num_outputs, hash_seed) {
  // Forward the constructor to the one above (with empty shape), so we have the
  // full hash information, then fetch/compute the real shape.
  SetShapeDeferred(shape_fn);
}

TsNode::TsNode(OpKind op, OpList operands, size_t num_outputs,
               lazy_tensors::hash_t hash_seed)
    : Node(std::move(op), operands, num_outputs, hash_seed) {}

TsNode::TsNode(OpKind op, OpList operands, lazy_tensors::Shape shape,
               size_t num_outputs, lazy_tensors::hash_t hash_seed)
    : Node(std::move(op), operands, num_outputs, hash_seed) {
  auto shape_fn = [&]() { return shape; };
  SetShapeDeferred(shape_fn);
}
TsNode::TsNode(OpKind op, lazy_tensors::Shape shape, size_t num_outputs,
               lazy_tensors::hash_t hash_seed)
    : Node(std::move(op), num_outputs, hash_seed) {
  auto shape_fn = [&]() { return shape; };
  SetShapeDeferred(shape_fn);
}
TsNode::~TsNode() {}

void TsNode::SetShapeDeferred(
    const std::function<lazy_tensors::Shape()>& shape_fn) {
  auto shape = GetOpShape(shape_fn);
  if (shape.IsTuple()) {
    std::vector<AtenShape> multi_out_aten_shape;
    multi_out_aten_shape.reserve(shape.tuple_shapes_size());
    for (size_t i = 0; i < shape.tuple_shapes_size(); i++) {
      multi_out_aten_shape.emplace_back(
          shape.tuple_shapes().at(i).dimensions().begin(),
          shape.tuple_shapes().at(i).dimensions().end());
    }
    set_aten_multi_out_shape(multi_out_aten_shape);
  } else {
    // TODO(whc) improve this; do we want to make Aten shape vec<const int> or
    // leave as is?
    set_aten_shape(std::vector<int64_t>(shape.dimensions().begin(),
                                        shape.dimensions().end()));
  }
  // TODO(whc) danger: better to avoid this kind of roundtrip conversion
  set_aten_dtype(PrimitiveToScalarType(shape.element_type()));
}

const lazy_tensors::Shape TsNode::shape() const {
  return lazy_tensors::Shape(aten_type(), aten_shape());
}

// Retrieves the shape of the output at a given index. If the node is not a
// multi-output node, output_index must be zero.
const lazy_tensors::Shape TsNode::shape(size_t output_index) const {
  return lazy_tensors::Shape(aten_type(), aten_shape(output_index));
}

lazy_tensors::Shape TsNode::GetOpShape(
    const std::function<lazy_tensors::Shape()>& shape_fn) const {
  ShapeCache* shape_cache = GetShapeCache();
  auto shape = shape_cache->Get(hash());
  if (shape == nullptr) {
    shape = shape_cache->Add(hash(),
                             std::make_shared<lazy_tensors::Shape>(shape_fn()));
  }
  return *shape;
}
NodePtr TsNode::Clone(OpList operands) const {
  LTC_ERROR() << "Cloning not implemented for tsnode: " << *this;
}

void TsNodeSetShapeDeferred(NodePtr node) {
  if (auto tsnode = std::dynamic_pointer_cast<ir::TsNode>(node)) {
    tsnode->SetShapeDeferred(
        [&]() { return compiler::NodeLowering::Get()->Infer(node.get()); });
  } else {
    throw std::runtime_error("Invalid node");
  }
}

}  // namespace ir
}  // namespace torch_lazy_tensors