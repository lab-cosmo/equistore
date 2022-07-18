#include <catch.hpp>

#include <equistore.hpp>
using namespace equistore;

static TensorMap test_tensor_map();

TEST_CASE("TensorMap") {
    SECTION("keys") {
        auto tensor = test_tensor_map();
        CHECK(tensor.keys() == Labels({"key_1", "key_2"}, {{0, 0}, {1, 0}, {2, 2}, {2, 3}}));
    }

    SECTION("get block") {
        auto tensor = test_tensor_map();

        // block by index
        auto block = tensor.block_by_id(2);
        const auto values = block.values();
        CHECK(values(0, 0, 0) == 3);

        // block by selection
        auto selection = Labels({"key_1", "key_2"}, {{1, 0}});
        auto matching = tensor.blocks_matching(selection);
        CHECK(matching.size() == 1);
        CHECK(matching[0] == 1);

        selection = Labels({"key_2"}, {{0}});
        matching = tensor.blocks_matching(selection);
        CHECK(matching.size() == 2);
        CHECK(matching[0] == 0);
        CHECK(matching[1] == 1);
    }

    SECTION("keys_to_samples") {
        auto tensor = test_tensor_map();
        tensor.keys_to_samples("key_2", /* sort_samples */ true);

        CHECK(tensor.keys() == Labels({"key_1"}, {{0}, {1}, {2}}));

        // The first two blocks are not modified
        auto block = tensor.block_by_id(0);
        CHECK(block.samples() == Labels({"samples", "key_2"}, {{0, 0}, {2, 0}, {4, 0}}));
        auto& values_1 = SimpleDataArray::from_eqs_array(block.eqs_array("values"));
        CHECK(values_1 == SimpleDataArray({3, 1, 1}, 1.0));


        block = tensor.block_by_id(1);
        CHECK(block.samples() == Labels({"samples", "key_2"}, {{0, 0}, {1, 0}, {3, 0}}));

        auto& values_2 = SimpleDataArray::from_eqs_array(block.eqs_array("values"));
        CHECK(values_2 == SimpleDataArray({3, 1, 3}, 2.0));

        // The new third block contains the old third and fourth blocks merged
        block = tensor.block_by_id(2);
        CHECK(block.samples() == Labels({"samples", "key_2"}, {
            {0, 2}, {0, 3}, {1, 3}, {2, 3}, {3, 2}, {5, 3}, {6, 2}, {8, 2}
        }));

        auto expected = SimpleDataArray({8, 3, 1}, {
            3.0, 3.0, 3.0,
            4.0, 4.0, 4.0,
            4.0, 4.0, 4.0,
            4.0, 4.0, 4.0,
            3.0, 3.0, 3.0,
            4.0, 4.0, 4.0,
            3.0, 3.0, 3.0,
            3.0, 3.0, 3.0,
        });

        auto& values_3 = SimpleDataArray::from_eqs_array(block.eqs_array("values"));
        CHECK(values_3 == expected);

        auto gradient = block.gradient("parameter");
        CHECK(gradient.samples() == Labels({"sample", "parameter"}, {
            {1, 1}, {4, -2}, {5, 3},
        }));

        expected = SimpleDataArray({3, 3, 1}, {
            14.0, 14.0, 14.0,
            13.0, 13.0, 13.0,
            14.0, 14.0, 14.0,
        });

        auto gradient_3 = SimpleDataArray::from_eqs_array(block.eqs_array("parameter"));
        CHECK(gradient_3 == expected);

        // unsorted samples
        tensor = test_tensor_map();
        tensor.keys_to_samples("key_2", /*sort_samples*/ false);

        block = tensor.block_by_id(2);
        CHECK(block.samples() == Labels({"samples", "key_2"}, {
            {0, 2}, {3, 2}, {6, 2}, {8, 2}, {0, 3}, {1, 3}, {2, 3}, {5, 3}
        }));
    }

    SECTION("keys_to_properties") {
        auto tensor = test_tensor_map();
        tensor.keys_to_properties("key_1");

        CHECK(tensor.keys() == Labels({"key_2"}, {{0}, {2}, {3}}));

        // The new first block contains the old first two blocks merged
        auto block = tensor.block_by_id(0);
        CHECK(block.samples() == Labels({"samples"}, {{0}, {1}, {2}, {3}, {4}}));

        auto components = block.components();
        CHECK(components.size() == 1);
        CHECK(components[0] == Labels({"component"}, {{0}}));

        CHECK(block.properties() == Labels({"key_1", "properties"}, {
            {0, 0}, {1, 3}, {1, 4}, {1, 5}
        }));

        auto expected = SimpleDataArray({5, 1, 4}, {
            1.0, 2.0, 2.0, 2.0,
            0.0, 2.0, 2.0, 2.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 2.0, 2.0,
            1.0, 0.0, 0.0, 0.0,
        });

        auto& values_1 = SimpleDataArray::from_eqs_array(block.eqs_array("values"));
        CHECK(values_1 == expected);

        auto gradient = block.gradient("parameter");
        CHECK(gradient.samples() == Labels({"sample", "parameter"}, {{0, -2}, {0, 3}, {3, -2}, {4, 3}}));

        expected = SimpleDataArray({4, 1, 4}, {
            11.0, 12.0, 12.0, 12.0,
            0.0, 12.0, 12.0, 12.0,
            0.0, 12.0, 12.0, 12.0,
            11.0, 0.0, 0.0, 0.0,
        });

        auto gradient_1 = SimpleDataArray::from_eqs_array(block.eqs_array("parameter"));
        CHECK(gradient_1 == expected);

        // The new second block contains the old third block
        block = tensor.block_by_id(1);
        CHECK(block.properties() == Labels({"key_1", "properties"}, {{2, 0}}));

        auto values_2 = SimpleDataArray::from_eqs_array(block.eqs_array("values"));
        CHECK(values_2 == SimpleDataArray({4, 3, 1}, 3.0));

        // the new third block contains the old fourth block
        block = tensor.block_by_id(2);
        CHECK(block.properties() == Labels({"key_1", "properties"}, {{2, 0}}));

        auto values_3 = SimpleDataArray::from_eqs_array(block.eqs_array("values"));
        CHECK(values_3 == SimpleDataArray({4, 3, 1}, 4.0));
    }

    SECTION("component_to_properties") {
        auto tensor = test_tensor_map();
        tensor.components_to_properties("component");

        auto block = tensor.block_by_id(0);

        CHECK(block.samples() == Labels({"samples"}, {{0}, {2}, {4}}));

        auto components = block.components();
        CHECK(components.size() == 0);

        CHECK(block.properties() == Labels({"component", "properties"}, {{0, 0}}));
    }
}


TEST_CASE("TensorMap serialization") {
    SECTION("loading file") {
        // DATA_NPZ is defined by cmake and expand to the path of tests/data.npz
        auto tensor = TensorMap::load(DATA_NPZ);

        auto keys = tensor.keys();
        CHECK(keys.names().size() == 3);
        CHECK(keys.names()[0] == std::string("spherical_harmonics_l"));
        CHECK(keys.names()[1] == std::string("center_species"));
        CHECK(keys.names()[2] == std::string("neighbor_species"));
        CHECK(keys.shape()[0] == 27);

        auto block = tensor.block_by_id(21);

        auto samples = block.samples();
        CHECK(samples.names().size() == 2);
        CHECK(samples.names()[0] == std::string("structure"));
        CHECK(samples.names()[1] == std::string("center"));

        CHECK(block.values().shape() == std::vector<size_t>{9, 5, 3});

        auto gradient = block.gradient("positions");
        samples = gradient.samples();
        CHECK(samples.names().size() == 3);
        CHECK(samples.names()[0] == std::string("sample"));
        CHECK(samples.names()[1] == std::string("structure"));
        CHECK(samples.names()[2] == std::string("atom"));

        CHECK(gradient.data().shape() == std::vector<size_t>{59, 3, 5, 3});
    }
}


TensorMap test_tensor_map() {
    auto blocks = std::vector<TensorBlock>();

    auto components = std::vector<Labels>();
    components.emplace_back(Labels({"component"}, {{0}}));

    auto block_1 = TensorBlock(
        std::unique_ptr<SimpleDataArray>(new SimpleDataArray({3, 1, 1}, 1.0)),
        Labels({"samples"}, {{0}, {2}, {4}}),
        components,
        Labels({"properties"}, {{0}})
    );
    block_1.add_gradient(
        "parameter",
        std::unique_ptr<SimpleDataArray>(new SimpleDataArray({2, 1, 1}, 11.0)),
        Labels({"sample", "parameter"}, {{0, -2}, {2, 3}}),
        components
    );

    blocks.emplace_back(std::move(block_1));

    auto block_2 = TensorBlock(
        std::unique_ptr<SimpleDataArray>(new SimpleDataArray({3, 1, 3}, 2.0)),
        Labels({"samples"}, {{0}, {1}, {3}}),
        components,
        Labels({"properties"}, {{3}, {4}, {5}})
    );
    block_2.add_gradient(
        "parameter",
        std::unique_ptr<SimpleDataArray>(new SimpleDataArray({3, 1, 3}, 12.0)),
        Labels({"sample", "parameter"}, {{0, -2}, {0, 3}, {2, -2}}),
        components
    );

    blocks.emplace_back(std::move(block_2));

    components = std::vector<Labels>();
    components.emplace_back(Labels({"component"}, {{0}, {1}, {2}}));
    auto block_3 = TensorBlock(
        std::unique_ptr<SimpleDataArray>(new SimpleDataArray({4, 3, 1}, 3.0)),
        Labels({"samples"}, {{0}, {3}, {6}, {8}}),
        components,
        Labels({"properties"}, {{0}})
    );
    block_3.add_gradient(
        "parameter",
        std::unique_ptr<SimpleDataArray>(new SimpleDataArray({1, 3, 1}, 13.0)),
        Labels({"sample", "parameter"}, {{1, -2}}),
        components
    );

    blocks.emplace_back(std::move(block_3));

    auto block_4 = TensorBlock(
        std::unique_ptr<SimpleDataArray>(new SimpleDataArray({4, 3, 1}, 4.0)),
        Labels({"samples"}, {{0}, {1}, {2}, {5}}),
        components,
        Labels({"properties"}, {{0}})
    );
    block_4.add_gradient(
        "parameter",
        std::unique_ptr<SimpleDataArray>(new SimpleDataArray({2, 3, 1}, 14.0)),
        Labels({"sample", "parameter"}, {{0, 1}, {3, 3}}),
        components
    );

    blocks.emplace_back(std::move(block_4));

    auto keys = Labels(
        {"key_1", "key_2"},
        {{0, 0}, {1, 0}, {2, 2}, {2, 3}}
    );

    return TensorMap(std::move(keys), std::move(blocks));
}
