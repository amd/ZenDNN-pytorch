#include <c10/core/impl/cow/context.h>

#include <c10/core/impl/cow/deleter.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace c10::impl {
namespace {

class DeleteTracker {
 public:
  explicit DeleteTracker(int& delete_count) : delete_count_(delete_count) {}
  ~DeleteTracker() {
    ++delete_count_;
  }

 private:
  int& delete_count_;
};

class ContextTest : public testing::Test {
 protected:
  auto delete_count() const -> int {
    return delete_count_;
  }
  auto new_delete_tracker() -> std::unique_ptr<void, DeleterFnPtr> {
    return {new DeleteTracker(delete_count_), +[](void* ptr) {
              delete static_cast<DeleteTracker*>(ptr);
            }};
  }

 private:
  int delete_count_ = 0;
};

TEST_F(ContextTest, Basic) {
  auto& context = *new cow::Context(new_delete_tracker());
  ASSERT_THAT(delete_count(), testing::Eq(0));

  context.increment_refcount();

  context.decrement_refcount();
  ASSERT_THAT(delete_count(), testing::Eq(0));

  context.decrement_refcount();
  ASSERT_THAT(delete_count(), testing::Eq(1));
}

TEST_F(ContextTest, decrement_refcount) {
  auto& context = *new cow::Context(new_delete_tracker());
  ASSERT_THAT(delete_count(), testing::Eq(0));

  context.decrement_refcount();
  ASSERT_THAT(delete_count(), testing::Eq(1));
}

TEST_F(ContextTest, delete_context) {
  // This is effectively the same thing as decrement_refcount() above.
  auto& context = *new cow::Context(new_delete_tracker());
  ASSERT_THAT(delete_count(), testing::Eq(0));

  cow::delete_context(&context);
  ASSERT_THAT(delete_count(), testing::Eq(1));
}

} // namespace
} // namespace c10::impl
