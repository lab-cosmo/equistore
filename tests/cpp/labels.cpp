#include <catch.hpp>

#include <equistore.hpp>
using namespace equistore;

TEST_CASE("Labels") {
    auto labels = Labels({"foo", "bar"}, {{1, 2}, {3, 4}, {5, 6}});

    CHECK(labels.shape().size() == 2);
    CHECK(labels.shape()[0] == 3);
    CHECK(labels.shape()[1] == 2);

    CHECK(labels.names().size() == 2);
    CHECK(labels.names()[0] == std::string("foo"));
    CHECK(labels.names()[1] == std::string("bar"));

    auto label = std::array<int32_t, 2>{3, 4};
    CHECK_THROWS_WITH(
        labels.position(label),
        "invalid parameter: these labels do not support calling eqs_labels_position"
    );

    CHECK(labels(0, 0) == 1);
    CHECK(labels(0, 1) == 2);

    CHECK(labels(1, 0) == 3);
    CHECK(labels(1, 1) == 4);

    CHECK(labels(2, 0) == 5);
    CHECK(labels(2, 1) == 6);

    CHECK(labels == Labels({"foo", "bar"}, {{1, 2}, {3, 4}, {5, 6}}));
    CHECK(labels != Labels({"bar", "foo"}, {{1, 2}, {3, 4}, {5, 6}}));


    CHECK_THROWS_WITH(labels(1), "expected 2 indexes in Labels::operator(), got 1");
    CHECK_THROWS_WITH(labels(1, 1, 1), "expected 2 indexes in Labels::operator(), got 3");

    CHECK_THROWS_WITH(
        labels.position({3, 4}),
        "invalid parameter: these labels do not support calling eqs_labels_position"
    );

    CHECK_THROWS_WITH(Labels({"foo"}, {{1}, {3, 4}}), "invalid size for row: expected 1 got 2");
}


TEST_CASE("Labels inside blocks") {
    auto block = TensorBlock(
        std::unique_ptr<SimpleDataArray>(new SimpleDataArray({3, 1})),
        Labels({"a", "b", "c"}, {{0, 1, 0}, {1, 1, 1}, {4, 3, 2}}),
        {},
        Labels({"properties"}, {{0}})
    );

    auto samples = block.samples();

    CHECK(samples.position({0, 1, 0}) == 0);
    CHECK(samples.position({4, 3, 2}) == 2);

    CHECK(samples.position({1, 3, 2}) == -1);

    CHECK_THROWS_WITH(
        samples.position({3, 4}),
        "invalid parameter: expected label of size 3 in eqs_labels_position, got size 2"
    );
}
