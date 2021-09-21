#pragma once

#include <ATen/core/interned_strings.h>

#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "lazy_tensor_core/csrc/python_util.h"
#include "lazy_tensors/computation_client/types.h"
#include "lazy_tensors/shape.h"
#include "lazy_tensors/span.h"

namespace torch_lazy_tensors {
namespace ir {

class Node;

using NodePtr = std::shared_ptr<Node>;

// The base class for user defined metadata which is possible to attach to IR
// nodes.
struct UserMetaData {
  virtual ~UserMetaData() {}
};

struct MetaData {
  std::string scope;
  std::vector<SourceLocation> frame_info;
};

// Represents a use of the output of a given node.
// If use U is within node N, it means that node U.node is using the output
// U.index of the node N.
struct Use {
  Use() = default;
  Use(Node* node, size_t operand_index, size_t index)
      : node(node), operand_index(operand_index), index(index) {}

  bool operator<(const Use& rhs) const;

  std::string ToString() const;

  // The node using the output of the node this use belongs to.
  Node* node = nullptr;
  // The operand index, within node's operands, which this use refers to.
  size_t operand_index = 0;
  // The index within output the user node refers to.
  size_t index = 0;
};

inline std::ostream& operator<<(std::ostream& stream, const Use& use) {
  stream << use.ToString();
  return stream;
}

// Represents a specific output produced by a node. Since the output of a node
// can be composed by multiple outputs, the node+index coordinates fully qualify
// each single output.
struct Output {
  struct Hasher {
    size_t operator()(const Output& output) const;
  };

  Output() = default;
  explicit Output(const Node* node, size_t index = 0)
      : node(node), index(index) {}

  // Retrieves the shape of this output. If the IR Node generating the value is
  // a multi-output node, the shape returned by this API will not be the full
  // tuple shape, but only the shape at index referred by this value.
  // To retrieve the full tuple shape in that case, use the node_shape() API.
  const lazy_tensors::Shape shape() const;
  const lazy_tensors::Shape node_shape() const;

  lazy_tensors::hash_t hash() const;

  bool operator==(const Output& rhs) const {
    return node == rhs.node && index == rhs.index;
  }
  bool operator!=(const Output& rhs) const { return !operator==(rhs); }

  std::string ToString() const;

  // The node providing the output.
  const Node* node = nullptr;
  // The index in the node's output this output refers to.
  size_t index = 0;
};

inline std::ostream& operator<<(std::ostream& stream, const Output& output) {
  stream << output.ToString();
  return stream;
}

using OutputSet = std::unordered_set<Output, Output::Hasher>;

template <typename T>
using OutputMap = std::unordered_map<Output, T, Output::Hasher>;

// Represents an input/operand for a Node object.
struct Value {
  Value() = default;
  Value(NodePtr node, size_t index = 0) : node(std::move(node)), index(index) {}

  // Retrieves the shape of this value. If the IR Node generating the value is a
  // multi-output node, the shape returned by this API will not be the full
  // tuple shape, but only the shape at index referred by this value.
  // To retrieve the full tuple shape in that case, use the node_shape() API.
  const lazy_tensors::Shape shape() const;
  const lazy_tensors::Shape node_shape() const;

  lazy_tensors::hash_t hash() const;

  operator bool() const { return node != nullptr; }

  operator Output() const { return Output(node.get(), index); }

  Node* operator->() const { return node.get(); }

  NodePtr node;
  size_t index = 0;
};

// The Kind of operation a Node can be associated to.
struct OpKind {
  OpKind() = default;
  explicit OpKind(c10::Symbol op) : op(std::move(op)) {}

  bool operator==(const OpKind& rhs) const { return op == rhs.op; }
  bool operator!=(const OpKind& rhs) const { return !operator==(rhs); }
  bool operator<(const OpKind& rhs) const {
    return c10::unique_t(op) < c10::unique_t(rhs.op);
  }

  lazy_tensors::hash_t hash() const;

  std::string ToString() const { return op.toQualString(); }

  // Retrieves an existing operation object, or creates a new one. Operations
  // that are specific to lazy tensors, should live within the 'lazy_tensors::'
  // namespace.
  static OpKind Get(const std::string& name);

  c10::Symbol op;
};

inline std::ostream& operator<<(std::ostream& stream, const OpKind& op) {
  stream << op.ToString();
  return stream;
}

using OpList = lazy_tensors::Span<const Value>;
using AtenShape = std::vector<int64_t>;

// A node in the graph. Nodes for operations which requires extra data to be
// stored for lowering, should inherit from this class and add operation
// specific member there. For example, a constant might create a new
// NodeConstant class (inheriting from Node) with an extra lazy_tensors::Literal
// field, or a tensor value might create a new NodeTensor with computation
// client data handle in it.
class Node {
 public:
  // Creates a new node with the given op name. The op is a unique identifier
  // for the operation. The num_outputs tells how many outputs a given operation
  // generates.
  Node(OpKind op, OpList operands, AtenShape aten_shape,
       c10::ScalarType aten_type, size_t num_outputs = 1,
       lazy_tensors::hash_t hash_seed = 0x5a2d296e9);
  Node(OpKind op, OpList operands, std::vector<AtenShape> multi_out_aten_shape,
       c10::ScalarType aten_type, size_t num_outputs,
       lazy_tensors::hash_t hash_seed = 0x5a2d296e9);
  // // Same as the constructor above, but the shape is generated by a function,
  // // only if needed (shape cache miss).
  // Node(OpKind op, OpList operands,
  //      const std::function<lazy_tensors::Shape()>& shape_fn,
  //      size_t num_outputs = 1, lazy_tensors::hash_t hash_seed = 0x5a2d296e9);

  // // The shape is set later.
  Node(OpKind op, OpList operands, size_t num_outputs = 1,
       lazy_tensors::hash_t hash_seed = 0x5a2d296e9);

  // void SetShapeDeferred(const std::function<lazy_tensors::Shape()>&
  // shape_fn);

  // Contructor used to create leaf nodes.
  Node(OpKind op, std::vector<int64_t> aten_shape, c10::ScalarType aten_type,
       size_t num_outputs, lazy_tensors::hash_t hash_seed);
  Node(OpKind op, size_t num_outputs, lazy_tensors::hash_t hash_seed);
  virtual ~Node();

  const OpKind& op() const { return op_; }

  size_t num_outputs() const { return num_outputs_; }

  // Retrieves the full shape of the IR Node. Note that if this is a
  // multi-output node, the returned shape will be a tuple.
  // const lazy_tensors::Shape& shape() const { return shape_; }

  // Retrieves the shape of the output at a given index. If the node is not a
  // multi-output node, output_index must be zero.
  // const lazy_tensors::Shape& shape(size_t output_index) const;
  const AtenShape& aten_shape(size_t output_index) const;
  const std::vector<int64_t>& aten_shape() const { return aten_shape_; }
  const std::vector<AtenShape>& multi_out_aten_shape() const {
    return multi_out_aten_shape_;
  }
  const c10::ScalarType& aten_type() const { return aten_type_; }

  void set_aten_shape(const AtenShape& aten_shape) { aten_shape_ = aten_shape; }

  void set_aten_dtype(const c10::ScalarType& aten_type) {
    aten_type_ = aten_type;
  }

  void set_aten_multi_out_shape(
      const std::vector<AtenShape>& multi_out_aten_shape) {
    multi_out_aten_shape_ = multi_out_aten_shape;
  }

  const std::vector<Output>& operands() const { return operands_as_outputs_; }

  const Output& operand(size_t i) const { return operands_as_outputs_.at(i); }

  const std::set<Use>& uses() const { return uses_; }

  lazy_tensors::hash_t node_hash() const { return node_hash_; }

  lazy_tensors::hash_t hash() const { return hash_; }

  const MetaData& metadata() const { return metadata_; }

  UserMetaData* user_metadata() const { return user_metadata_.get(); }

  std::shared_ptr<UserMetaData> SetUserMetadata(
      std::shared_ptr<UserMetaData> user_meta) {
    std::swap(user_metadata_, user_meta);
    return user_meta;
  }

  void ReplaceOperand(size_t operand_no, NodePtr node, size_t index = 0);

  void ReplaceAllUsesWith(NodePtr node, size_t index = 0);

  virtual std::string ToString() const;

  virtual NodePtr Clone(OpList operands) const;

 private:
  // Adds node's index output number as operand.
  void AddOperand(NodePtr node, size_t index = 0);

  void AddUse(Use use) { uses_.insert(std::move(use)); }

  void RemoveUse(const Use& use) { uses_.erase(use); }

  // lazy_tensors::Shape GetOpShape(
  //     const std::function<lazy_tensors::Shape()>& shape_fn) const;

  static lazy_tensors::hash_t GetOpHash(OpKind op,
                                        std::vector<int64_t> aten_shape,
                                        c10::ScalarType aten_type,
                                        lazy_tensors::hash_t hash_seed);

  static std::vector<SourceLocation> GetFrameInfo();

  // The ID of the operation captured by this node.
  OpKind op_;
  size_t num_outputs_ = 1;
  // TODO(whc) use new names now to aid refactoring, but may want to rename to
  // 'shape', 'dtype' later
  AtenShape aten_shape_;
  // TODO(whc) combine or keep separate (the multi-out field?) taken from
  // XLA::shape
  std::vector<AtenShape> multi_out_aten_shape_;
  c10::ScalarType aten_type_;
  // A node holds a real reference to its operands.
  std::vector<NodePtr> operands_;
  // Outputs do not hold references on the nodes, and neither do the uses, since
  // otherwise we get into circular reference counting.
  std::vector<Output> operands_as_outputs_;
  // We use a set for uses, as we want deterministic use sequencing.
  std::set<Use> uses_;
  // The hash value of this node.
  lazy_tensors::hash_t node_hash_ = 0;
  // The hash value of the graph rooted at this node.
  lazy_tensors::hash_t hash_ = 0;
  // The IR specific metadata attached to the IR node.
  MetaData metadata_;
  // The IR framework user can attach a user defined metadata object deriving
  // from UserMetaData.
  std::shared_ptr<UserMetaData> user_metadata_;
};

// TODO(whc) this is a helper to use during refactoring.  Ultimately
// things using Node base shouldn't be expecting lazy_tensors::Shape
lazy_tensors::Shape AtenToLazyShapeHelper(const Node& node) {
  if (node.num_outputs() >= 1) {
    return lazy_tensors::Shape(node.aten_type(), node.aten_shape());
  } else {
    std::vector<lazy_tensors::Shape> tuple_shapes;
    tuple_shapes.reserve(node.multi_out_aten_shape().size());
    for (size_t i = 0; i < node.multi_out_aten_shape().size(); i++) {
      tuple_shapes.emplace_back(lazy_tensors::Shape(
          node.aten_type(), node.multi_out_aten_shape().at(i)));
    }
    return lazy_tensors::Shape(tuple_shapes);
  }
}

// RAII data structure to be used a stack variable to enter a new IR scope. IR
// scope names will appear in the IR and will help identifying the source of the
// single IR nodes.
struct ScopePusher {
  explicit ScopePusher(const std::string& name);
  ~ScopePusher();

  static void ResetScopes();
};

inline std::ostream& operator<<(std::ostream& stream, const Node& node) {
  stream << node.ToString();
  return stream;
}

template <typename T, typename... Args>
NodePtr MakeNode(Args&&... args) {
  return std::make_shared<T>(std::forward<Args>(args)...);
}

template <typename T>
T* NodeCast(const Node* node, OpKind op) {
  if (op != node->op()) {
    return nullptr;
  }
  const T* casted;
#ifdef NDEBUG
  casted = static_cast<const T*>(node);
#else
  casted = &dynamic_cast<const T&>(*node);
#endif
  return const_cast<T*>(casted);
}

}  // namespace ir
}  // namespace torch_lazy_tensors
