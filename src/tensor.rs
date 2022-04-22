use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use indexmap::IndexSet;

use crate::{TensorBlock, Error, BasicBlock};
use crate::{Labels, LabelsBuilder, LabelValue};

/// A tensor map is the main user-facing struct of this library, and can store
/// any kind of data used in atomistic machine learning.
///
/// A tensor map contains a list of `TensorBlock`s, each one associated with a
/// key in the form of a single `Labels` entry.
///
/// It provides functions to merge blocks together by moving some of these keys
/// to the samples or properties labels of the blocks, transforming the sparse
/// representation of the data to a dense one.
#[derive(Debug)]
pub struct TensorMap {
    keys: Labels,
    blocks: Vec<TensorBlock>,
    // TODO: arbitrary tensor-level metadata? e.g. using `HashMap<String, String>`
}

#[allow(clippy::needless_pass_by_value)]
fn check_labels_names(
    block: &BasicBlock,
    sample_names: &[&str],
    components_names: &[Vec<&str>],
    context: String,
) -> Result<(), Error> {
    if block.samples().names() != sample_names {
        return Err(Error::InvalidParameter(format!(
            "all blocks must have the same sample label names, got [{}] and [{}]{}",
            block.samples().names().join(", "),
            sample_names.join(", "),
            context,
        )));
    }

    if block.components().len() != components_names.len() {
        return Err(Error::InvalidParameter(format!(
            "all blocks must contains the same set of components, the current \
            block has {} components while the first block has {}{}",
            block.components().len(),
            components_names.len(),
            context,
        )));
    }

    for (component_i, component) in block.components().iter().enumerate() {
        if component.names() != components_names[component_i] {
            return Err(Error::InvalidParameter(format!(
                "all blocks must have the same component label names, got [{}] and [{}]{}",
                component.names().join(", "),
                components_names[component_i].join(", "),
                context,
            )));
        }
    }

    Ok(())
}

impl TensorMap {
    /// TODO: doc
    #[allow(clippy::similar_names)]
    pub fn new(keys: Labels, blocks: Vec<TensorBlock>) -> Result<TensorMap, Error> {
        if blocks.len() != keys.count() {
            return Err(Error::InvalidParameter(format!(
                "expected the same number of blocks ({}) as the number of \
                entries in the keys when creating a tensor, got {}",
                keys.count(), blocks.len()
            )))
        }

        if !blocks.is_empty() {
            // make sure all blocks have the same kind of samples, components &
            // properties labels
            let sample_names = blocks[0].values.samples().names();
            let components_names = blocks[0].values.components()
                .iter()
                .map(|c| c.names())
                .collect::<Vec<_>>();
            let properties_names = blocks[0].values.properties().names();

            let gradients_data = blocks[0].gradients().iter()
                .map(|(name, gradient)| {
                    let components_names = gradient.components()
                        .iter()
                        .map(|c| c.names())
                        .collect::<Vec<_>>();
                    (&**name, (gradient.samples().names(), components_names))
                })
                .collect::<HashMap<_, _>>();


            for block in &blocks {
                check_labels_names(&block.values, &sample_names, &components_names, "".into())?;

                if block.values.properties().names() != properties_names {
                    return Err(Error::InvalidParameter(format!(
                        "all blocks must have the same property label names, got [{}] and [{}]",
                        block.values.properties().names().join(", "),
                        properties_names.join(", "),
                    )));
                }

                if block.gradients().len() != gradients_data.len() {
                    return Err(Error::InvalidParameter(
                        "all blocks must contains the same set of gradients".into(),
                    ));
                }

                for (parameter, gradient) in block.gradients() {
                    match gradients_data.get(&**parameter) {
                        None => {
                            return Err(Error::InvalidParameter(format!(
                                "missing gradient with respect to {} in one of the blocks",
                                parameter
                            )));
                        },
                        Some((sample_names, components_names)) => {
                            check_labels_names(
                                gradient,
                                sample_names,
                                components_names,
                                format!(" for gradients with respect to {}", parameter)
                            )?;
                        }
                    }
                }
            }
        }

        Ok(TensorMap {
            keys,
            blocks,
        })
    }

    /// Get the list of blocks in this `TensorMap`
    pub fn blocks(&self) -> &[TensorBlock] {
        &self.blocks
    }

    /// Get the keys defined in this `TensorMap`
    pub fn keys(&self) -> &Labels {
        &self.keys
    }

    /// Get an iterator over the keys and associated block
    pub fn iter(&self) -> impl Iterator<Item=(&[LabelValue], &TensorBlock)> + '_ {
        self.keys.iter().zip(&self.blocks)
    }

    /// Get the list of blocks matching the given selection. The selection must
    /// contains a single entry, defining the requested key. The selection can
    /// contain only a subset of the variables defined in the keys, in which
    /// case there can be multiple matching blocks.
    pub fn blocks_matching(&self, selection: &Labels) -> Result<Vec<&TensorBlock>, Error> {
        let matching = self.find_matching_blocks(selection)?;

        return Ok(matching.into_iter().map(|i| &self.blocks[i]).collect());
    }

    /// Get a reference to the block matching the given selection.
    ///
    /// The selection behaves similarly to `blocks_matching`, with the exception
    /// that this function returns an error if there is more than one matching
    /// block.
    pub fn block(&self, selection: &Labels) -> Result<&TensorBlock, Error> {
        let matching = self.find_matching_blocks(selection)?;
        if matching.len() != 1 {
            let selection_str = selection.names()
                .iter().zip(&selection[0])
                .map(|(name, value)| format!("{} = {}", name, value))
                .collect::<Vec<_>>()
                .join(", ");


            return Err(Error::InvalidParameter(format!(
                "{} blocks matched the selection ({}), expected only one",
                matching.len(), selection_str
            )));
        }

        return Ok(&self.blocks[matching[0]]);
    }

    /// Actual implementation of `blocks_matching` and related functions, this
    /// function finds the matching blocks & return their index in the
    /// `self.blocks` vector.
    fn find_matching_blocks(&self, selection: &Labels) -> Result<Vec<usize>, Error> {
        if selection.size() == 0 {
            return Ok((0..self.blocks().len()).collect());
        }

        if selection.count() != 1 {
            return Err(Error::InvalidParameter(format!(
                "block selection labels must contain a single row, got {}",
                selection.count()
            )));
        }

        let mut variables = Vec::new();
        'outer: for requested in selection.names() {
            for (i, &name) in self.keys.names().iter().enumerate() {
                if requested == name {
                    variables.push(i);
                    continue 'outer;
                }
            }

            return Err(Error::InvalidParameter(format!(
                "'{}' is not part of the keys for this tensor",
                requested
            )));
        }

        let mut matching = Vec::new();
        let selection = selection.iter().next().expect("empty selection");

        for (block_i, labels) in self.keys.iter().enumerate() {
            let mut selected = true;
            for (&requested_i, &value) in variables.iter().zip(selection) {
                if labels[requested_i] != value {
                    selected = false;
                    break;
                }
            }

            if selected {
                matching.push(block_i);
            }
        }

        return Ok(matching);
    }

    /// Move the given variables from the keys to the property labels of the
    /// blocks.
    ///
    /// Blocks containing the same values in the keys for the `variables` will
    /// be merged together. The resulting merged blocks will have `variables` as
    /// the first property variables, followed by the current properties. The
    /// new sample labels will contains all of the merged blocks sample labels.
    ///
    /// The order of the samples is controlled by `sort_samples`. If
    /// `sort_samples` is true, samples are re-ordered to keep them
    /// lexicographically sorted. Otherwise they are kept in the order in which
    /// they appear in the blocks.
    pub fn keys_to_properties(&mut self, variables: &[&str], sort_samples: bool) -> Result<(), Error> {
        // TODO: requested values
        // TODO: keys_to_properties_no_gradients?

        if variables.is_empty() {
            return Ok(());
        }

        let (new_keys, new_properties) = split_keys(&self.keys, variables)?;

        let mut new_blocks = Vec::new();
        if new_keys.count() == 1 {
            // create a single block with everything
            let mut matching = Vec::new();
            for i in 0..self.blocks.len() {
                matching.push(i);
            }

            let block = self.merge_blocks_along_properties(
                &matching, &new_properties, sort_samples
            )?;
            new_blocks.push(block);
        } else {
            for entry in new_keys.iter() {
                let mut selection = LabelsBuilder::new(new_keys.names());
                selection.add(entry.to_vec());

                let matching = self.find_matching_blocks(&selection.finish())?;
                let block = self.merge_blocks_along_properties(
                    &matching, &new_properties, sort_samples
                )?;
                new_blocks.push(block);
            }
        }


        self.keys = new_keys;
        self.blocks = new_blocks;

        return Ok(());
    }

    /// Merge the blocks with the given `block_idx` along the property axis. The
    /// property names & values to add to the property axis are passed in
    /// `new_property_labels`.
    #[allow(clippy::too_many_lines)]
    fn merge_blocks_along_properties(
        &self,
        block_idx: &[usize],
        new_property_labels: &Labels,
        sort_samples: bool,
    ) -> Result<TensorBlock, Error> {
        assert!(!block_idx.is_empty());

        let blocks_to_merge = block_idx.iter().map(|&i| &self.blocks[i]).collect::<Vec<_>>();

        let first_block = &self.blocks[block_idx[0]];
        let first_components_label = first_block.values.components();
        for block in &blocks_to_merge {
            if block.values.components() != first_components_label {
                return Err(Error::InvalidParameter(
                    "can not move keys to properties if the blocks have \
                    different components labels, call components_to_properties first".into()
                ))
            }
        }

        let new_property_names = new_property_labels.names().iter()
            .chain(first_block.values.properties().names().iter())
            .copied()
            .collect();
        let mut new_properties_builder = LabelsBuilder::new(new_property_names);
        let mut old_property_sizes = Vec::new();

        // Collect samples in an IndexSet to keep them in the same order as they
        // were in the blocks, and then optionally sort them later below
        let mut merged_samples = IndexSet::new();
        for (block, new_property) in blocks_to_merge.iter().zip(new_property_labels) {
            for sample in block.values.samples().iter() {
                merged_samples.insert(sample.to_vec());
            }

            let old_properties = block.values.properties();
            old_property_sizes.push(old_properties.count());
            for old_property in old_properties.iter() {
                let mut property = new_property.to_vec();
                property.extend_from_slice(old_property);
                new_properties_builder.add(property);
            }
        }

        if sort_samples {
            merged_samples.sort_unstable();
        }

        let mut merged_samples_builder = LabelsBuilder::new(first_block.values.samples().names());
        for sample in merged_samples {
            merged_samples_builder.add(sample);
        }
        let merged_samples = merged_samples_builder.finish();

        // Vec<Vec<usize>> mapping from old values sample index (per block) to
        // the new sample index
        let mut samples_mapping = Vec::new();
        for block in &blocks_to_merge {
            let mut mapping_for_block = Vec::new();
            for sample in block.values.samples().iter() {
                let new_sample_i = merged_samples.position(sample).expect("missing entry in merged samples");
                mapping_for_block.push(new_sample_i);
            }
            samples_mapping.push(mapping_for_block);
        }

        let new_components = first_block.values.components().to_vec();
        let new_properties = Arc::new(new_properties_builder.finish());
        let new_properties_count = new_properties.count();

        let mut new_shape = first_block.values.data.shape()?.to_vec();
        new_shape[0] = merged_samples.count();
        let property_axis = new_shape.len() - 1;
        new_shape[property_axis] = new_properties_count;
        let mut new_data = first_block.values.data.create(&new_shape)?;

        let mut property_ranges = Vec::new();
        let mut start = 0;
        for size in old_property_sizes {
            let stop = start + size;
            property_ranges.push(start..stop);
            start = stop;
        }

        for ((block_i, block), property_range) in blocks_to_merge.iter().enumerate().zip(&property_ranges) {
            for sample_i in 0..block.values.samples().count() {
                let new_sample_i = samples_mapping[block_i][sample_i];
                new_data.move_sample(
                    new_sample_i,
                    property_range.clone(),
                    &block.values.data,
                    sample_i
                )?;
            }
        }

        let mut new_block = TensorBlock::new(
            new_data,
            merged_samples,
            new_components,
            new_properties
        ).expect("constructed an invalid block");

        // now collect & merge the different gradients
        for (parameter, first_gradient) in first_block.gradients() {
            let new_gradient_samples = merge_gradient_samples(
                &blocks_to_merge, parameter, &samples_mapping
            );

            let mut new_shape = first_gradient.data.shape()?.to_vec();
            new_shape[0] = new_gradient_samples.count();
            let property_axis = new_shape.len() - 1;
            new_shape[property_axis] = new_properties_count;

            let mut new_gradient = first_block.values.data.create(&new_shape)?;
            let new_components = first_gradient.components().to_vec();

            for ((block_i, block), property_range) in blocks_to_merge.iter().enumerate().zip(&property_ranges) {
                let gradient = block.get_gradient(parameter).expect("missing gradient");
                debug_assert!(gradient.components() == new_components);

                for (sample_i, grad_sample) in gradient.samples().iter().enumerate() {
                    // translate from the old sample id in gradients to the new ones
                    let mut grad_sample = grad_sample.to_vec();
                    let old_sample_i = grad_sample[0].usize();
                    grad_sample[0] = LabelValue::from(samples_mapping[block_i][old_sample_i]);

                    let new_sample_i = new_gradient_samples.position(&grad_sample).expect("missing entry in merged samples");
                    new_gradient.move_sample(
                        new_sample_i,
                        property_range.clone(),
                        &gradient.data,
                        sample_i
                    )?;
                }
            }

            new_block.add_gradient(
                parameter, new_gradient, new_gradient_samples, new_components
            ).expect("created invalid gradients");
        }

        return Ok(new_block);
    }

    /// Move the given variables from the component labels to the property labels
    /// for each block in this `TensorMap`.
    pub fn components_to_properties(&mut self, variables: &[&str]) -> Result<(), Error> {
        // TODO: requested values
        if variables.is_empty() {
            return Ok(());
        }

        for block in &mut self.blocks {
            block.components_to_properties(variables)?;
        }

        Ok(())
    }

    /// Move the given variables from the keys to the sample labels of the
    /// blocks.
    ///
    /// Blocks containing the same values in the keys for the `variables` will
    /// be merged together. The resulting merged blocks will have `variables` as
    /// the last sample variables, preceded by the current samples.
    ///
    /// This function is only implemented if all merged block have the same
    /// property labels.
    pub fn keys_to_samples(&mut self, variables: &[&str]) -> Result<(), Error> {
        // TODO: requested values
        // TODO: keys_to_samples_no_gradients?

        if variables.is_empty() {
            return Ok(());
        }

        let (new_keys, new_samples) = split_keys(&self.keys, variables)?;

        let mut new_blocks = Vec::new();
        if new_keys.count() == 1 {
            // create a single block with everything
            let mut matching = Vec::new();
            for i in 0..self.blocks.len() {
                matching.push(i);
            }

            let block = self.merge_blocks_along_samples(&matching, &new_samples)?;
            new_blocks.push(block);
        } else {
            for entry in new_keys.iter() {
                let mut selection = LabelsBuilder::new(new_keys.names());
                selection.add(entry.to_vec());

                let matching = self.find_matching_blocks(&selection.finish())?;
                new_blocks.push(self.merge_blocks_along_samples(&matching, &new_samples)?);
            }
        }


        self.keys = new_keys;
        self.blocks = new_blocks;

        return Ok(());
    }

    /// Merge the blocks with the given `block_idx` along the sample axis. The
    /// new sample names & values to add to the sample axis are passed in
    /// `new_sample_labels`.
    fn merge_blocks_along_samples(
        &self,
        block_idx: &[usize],
        new_sample_labels: &Labels,
    ) -> Result<TensorBlock, Error> {
        assert!(!block_idx.is_empty());

        let first_block = &self.blocks[block_idx[0]];
        let first_components_label = first_block.values.components();
        let first_properties_label = first_block.values.properties();

        let blocks_to_merge = block_idx.iter().map(|&i| &self.blocks[i]).collect::<Vec<_>>();
        for block in &blocks_to_merge {
            if block.values.components() != first_components_label {
                return Err(Error::InvalidParameter(
                    "can not move keys to samples if the blocks have \
                    different components labels, call components_to_properties first".into()
                ))
            }

            if block.values.properties() != first_properties_label {
                return Err(Error::InvalidParameter(
                    "can not move keys to samples if the blocks have \
                    different property labels".into() // TODO: this might be possible
                ))
            }
        }

        // we need to collect the new samples in a BTree set to ensure they stay
        // lexicographically ordered
        let mut merged_samples = BTreeSet::new();
        for (block, new_sample_label) in blocks_to_merge.iter().zip(new_sample_labels) {
            for old_sample in block.values.samples().iter() {
                let mut sample = old_sample.to_vec();
                sample.extend_from_slice(new_sample_label);
                merged_samples.insert(sample);
            }
        }

        let new_samples_names = first_block.values.samples().names().iter().copied()
            .chain(new_sample_labels.names())
            .collect();

        let mut merged_samples_builder = LabelsBuilder::new(new_samples_names);
        for sample in merged_samples {
            merged_samples_builder.add(sample);
        }
        let merged_samples = merged_samples_builder.finish();
        let new_components = first_block.values.components().to_vec();
        let new_properties = Arc::clone(first_block.values.properties());

        let mut new_shape = first_block.values.data.shape()?.to_vec();
        new_shape[0] = merged_samples.count();
        let mut new_data = first_block.values.data.create(&new_shape)?;

        let mut samples_mapping = Vec::new();
        for (block, new_sample_label) in blocks_to_merge.iter().zip(new_sample_labels) {
            let mut mapping_for_block = Vec::new();
            for sample in block.values.samples().iter() {
                let mut new_sample = sample.to_vec();
                new_sample.extend_from_slice(new_sample_label);

                let new_sample_i = merged_samples.position(&new_sample).expect("missing entry in merged samples");
                mapping_for_block.push(new_sample_i);
            }
            samples_mapping.push(mapping_for_block);
        }

        let property_range = 0..new_properties.count();

        for (block_i, block) in blocks_to_merge.iter().enumerate() {
            for sample_i in 0..block.values.samples().count() {

                new_data.move_sample(
                    samples_mapping[block_i][sample_i],
                    property_range.clone(),
                    &block.values.data,
                    sample_i
                )?;
            }
        }

        let mut new_block = TensorBlock::new(
            new_data,
            merged_samples,
            new_components,
            new_properties
        ).expect("invalid block");

        // now collect & merge the different gradients
        for (parameter, first_gradient) in first_block.gradients() {
            let new_gradient_samples = merge_gradient_samples(
                &blocks_to_merge, parameter, &samples_mapping
            );

            let mut new_shape = first_gradient.data.shape()?.to_vec();
            new_shape[0] = new_gradient_samples.count();
            let mut new_gradient = first_block.values.data.create(&new_shape)?;
            let new_components = first_gradient.components().to_vec();

            for (block_i, block) in blocks_to_merge.iter().enumerate() {
                let gradient = block.get_gradient(parameter).expect("missing gradient");
                debug_assert!(gradient.components() == new_components);

                for (sample_i, grad_sample) in gradient.samples().iter().enumerate() {
                    // translate from the old sample id in gradients to the new ones
                    let mut grad_sample = grad_sample.to_vec();
                    let old_sample_i = grad_sample[0].usize();
                    grad_sample[0] = LabelValue::from(samples_mapping[block_i][old_sample_i]);

                    let new_sample_i = new_gradient_samples.position(&grad_sample).expect("missing entry in merged samples");
                    new_gradient.move_sample(
                        new_sample_i,
                        property_range.clone(),
                        &gradient.data,
                        sample_i
                    )?;
                }
            }

            new_block.add_gradient(
                parameter, new_gradient, new_gradient_samples, new_components
            ).expect("created invalid gradients");
        }

        return Ok(new_block);
    }
}

fn merge_gradient_samples(blocks: &[&TensorBlock], gradient_name: &str, mapping: &[Vec<usize>]) -> Labels {
    let mut new_gradient_samples = BTreeSet::new();
    let mut new_gradient_samples_names = None;
    for (block_i, block) in blocks.iter().enumerate() {
        let gradient = block.get_gradient(gradient_name).expect("missing gradient");

        if new_gradient_samples_names.is_none() {
            new_gradient_samples_names = Some(gradient.samples().names());
        }

        for grad_sample in gradient.samples().iter() {
            // translate from the old sample id in gradients to the new ones
            let mut grad_sample = grad_sample.to_vec();
            let old_sample_i = grad_sample[0].usize();
            grad_sample[0] = LabelValue::from(mapping[block_i][old_sample_i]);

            new_gradient_samples.insert(grad_sample);
        }
    }

    let mut new_gradient_samples_builder = LabelsBuilder::new(new_gradient_samples_names.expect("missing gradient samples names"));
    for sample in new_gradient_samples {
        new_gradient_samples_builder.add(sample);
    }
    return new_gradient_samples_builder.finish();
}

/// Split the `keys` into a new set of label without the `variables`; and Labels
/// containing the values taken by `variables`
fn split_keys(keys: &Labels, variables: &[&str]) -> Result<(Labels, Labels), Error> {
    let names = keys.names();
    for variable in variables {
        if !names.contains(variable) {
            return Err(Error::InvalidParameter(format!(
                "'{}' is not part of the keys for this tensor map",
                variable
            )));
        }
    }

    // TODO: use Labels instead of Vec<&str> for variables to ensure
    // uniqueness of variables names & pass 'requested' values around

    let mut remaining = Vec::new();
    let mut remaining_i = Vec::new();
    let mut extracted_i = Vec::new();

    'outer: for (i, &name) in names.iter().enumerate() {
        for &variable in variables {
            if variable == name {
                extracted_i.push(i);
                continue 'outer;
            }
        }
        remaining.push(name);
        remaining_i.push(i);
    }

    let mut extracted_keys = IndexSet::new();
    let mut remaining_keys = IndexSet::new();
    for key in keys.iter() {
        let mut label = Vec::new();
        for &i in &extracted_i {
            label.push(key[i]);
        }
        extracted_keys.insert(label);

        if !remaining_i.is_empty() {
            let mut label = Vec::new();
            for &i in &remaining_i {
                label.push(key[i]);
            }
            remaining_keys.insert(label);
        }
    }

    let remaining_keys = if remaining_keys.is_empty() {
        Labels::single()
    } else {
        let mut remaining_keys_builder = LabelsBuilder::new(remaining);
        for entry in remaining_keys {
            remaining_keys_builder.add(entry);
        }
        remaining_keys_builder.finish()
    };

    assert!(!extracted_keys.is_empty());
    let mut extracted_keys_builder = LabelsBuilder::new(variables.to_vec());
    for entry in extracted_keys {
        extracted_keys_builder.add(entry);
    }

    return Ok((remaining_keys, extracted_keys_builder.finish()));
}


#[cfg(test)]
mod tests {
    use super::*;

    use crate::eqs_array_t;
    use crate::data::TestArray;

    fn example_labels(name: &str, count: usize) -> Labels {
        let mut labels = LabelsBuilder::new(vec![name]);
        for i in 0..count {
            labels.add(vec![LabelValue::from(i)]);
        }
        return labels.finish();
    }

    #[test]
    fn blocks_validation() {
        let block_1 = TensorBlock::new(
            eqs_array_t::new(Box::new(TestArray::new(vec![1, 1, 1]))),
            example_labels("samples", 1),
            vec![Arc::new(example_labels("components", 1))],
            Arc::new(example_labels("properties", 1)),
        ).unwrap();

        let block_2 = TensorBlock::new(
            eqs_array_t::new(Box::new(TestArray::new(vec![2, 3, 1]))),
            example_labels("samples", 2),
            vec![Arc::new(example_labels("components", 3))],
            Arc::new(example_labels("properties", 1)),
        ).unwrap();

        let result = TensorMap::new(example_labels("keys", 2), vec![block_1, block_2]);
        assert!(result.is_ok());

        /**********************************************************************/
        let block_1 = TensorBlock::new(
            eqs_array_t::new(Box::new(TestArray::new(vec![1, 1]))),
            example_labels("samples", 1),
            vec![],
            Arc::new(example_labels("properties", 1)),
        ).unwrap();

        let block_2 = TensorBlock::new(
            eqs_array_t::new(Box::new(TestArray::new(vec![2, 1]))),
            example_labels("something_else", 2),
            vec![],
            Arc::new(example_labels("properties", 1)),
        ).unwrap();

        let result = TensorMap::new(example_labels("keys", 2), vec![block_1, block_2]);
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: all blocks must have the same sample label \
            names, got [something_else] and [samples]"
        );

        /**********************************************************************/
        let block_1 = TensorBlock::new(
            eqs_array_t::new(Box::new(TestArray::new(vec![1, 1, 1]))),
            example_labels("samples", 1),
            vec![Arc::new(example_labels("components", 1))],
            Arc::new(example_labels("properties", 1)),
        ).unwrap();

        let block_2 = TensorBlock::new(
            eqs_array_t::new(Box::new(TestArray::new(vec![2, 1]))),
            example_labels("samples", 2),
            vec![],
            Arc::new(example_labels("properties", 1)),
        ).unwrap();

        let result = TensorMap::new(example_labels("keys", 2), vec![block_1, block_2]);
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: all blocks must contains the same set of \
            components, the current block has 0 components while the first \
            block has 1"
        );

        /**********************************************************************/
        let block_1 = TensorBlock::new(
            eqs_array_t::new(Box::new(TestArray::new(vec![1, 1, 1]))),
            example_labels("samples", 1),
            vec![Arc::new(example_labels("components", 1))],
            Arc::new(example_labels("properties", 1)),
        ).unwrap();

        let block_2 = TensorBlock::new(
            eqs_array_t::new(Box::new(TestArray::new(vec![2, 3, 1]))),
            example_labels("samples", 2),
            vec![Arc::new(example_labels("something_else", 3))],
            Arc::new(example_labels("properties", 1)),
        ).unwrap();

        let result = TensorMap::new(example_labels("keys", 2), vec![block_1, block_2]);
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: all blocks must have the same component label \
            names, got [something_else] and [components]"
        );

        /**********************************************************************/
        let block_1 = TensorBlock::new(
            eqs_array_t::new(Box::new(TestArray::new(vec![1, 1]))),
            example_labels("samples", 1),
            vec![],
            Arc::new(example_labels("properties", 1)),
        ).unwrap();

        let block_2 = TensorBlock::new(
            eqs_array_t::new(Box::new(TestArray::new(vec![2, 1]))),
            example_labels("samples", 2),
            vec![],
            Arc::new(example_labels("something_else", 1)),
        ).unwrap();

        let result = TensorMap::new(example_labels("keys", 2), vec![block_1, block_2]);
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: all blocks must have the same property label \
            names, got [something_else] and [properties]"
        );

        // TODO: check error messages for gradients
    }

    #[cfg(feature = "ndarray")]
    mod moving_labels {
        use super::*;
        use ndarray::ArrayD;

        fn example_labels(name: &str, values: Vec<i32>) -> Labels {
            let mut labels = LabelsBuilder::new(vec![name]);
            for i in values {
                labels.add(vec![LabelValue::from(i)]);
            }
            return labels.finish();
        }

        fn example_tensor() -> TensorMap {
            let mut block_1 = TensorBlock::new(
                eqs_array_t::new(Box::new(ArrayD::from_elem(vec![3, 1, 1], 1.0))),
                example_labels("samples", vec![0, 2, 4]),
                vec![Arc::new(example_labels("components", vec![0]))],
                Arc::new(example_labels("properties", vec![0])),
            ).unwrap();

            let mut gradient_samples_1 = LabelsBuilder::new(vec!["sample", "parameter"]);
            gradient_samples_1.add(vec![LabelValue::new(0), LabelValue::new(-2)]);
            gradient_samples_1.add(vec![LabelValue::new(2), LabelValue::new(3)]);
            let gradient_samples_1 = gradient_samples_1.finish();

            block_1.add_gradient(
                "parameter",
                eqs_array_t::new(Box::new(ArrayD::from_elem(vec![2, 1, 1], 11.0))),
                gradient_samples_1,
                vec![Arc::new(example_labels("components", vec![0]))],
            ).unwrap();

            /******************************************************************/

            let mut block_2 = TensorBlock::new(
                eqs_array_t::new(Box::new(ArrayD::from_elem(vec![3, 1, 3], 2.0))),
                example_labels("samples", vec![0, 1, 3]),
                vec![Arc::new(example_labels("components", vec![0]))],
                // different property size
                Arc::new(example_labels("properties", vec![3, 4, 5])),
            ).unwrap();

            let mut gradient_samples_2 = LabelsBuilder::new(vec!["sample", "parameter"]);
            gradient_samples_2.add(vec![LabelValue::new(0), LabelValue::new(-2)]);
            gradient_samples_2.add(vec![LabelValue::new(0), LabelValue::new(3)]);
            gradient_samples_2.add(vec![LabelValue::new(2), LabelValue::new(-2)]);
            let gradient_samples_2 = gradient_samples_2.finish();

            block_2.add_gradient(
                "parameter",
                eqs_array_t::new(Box::new(ArrayD::from_elem(vec![3, 1, 3], 12.0))),
                gradient_samples_2,
                vec![Arc::new(example_labels("components", vec![0]))],
            ).unwrap();

            /******************************************************************/

            let mut block_3 = TensorBlock::new(
                eqs_array_t::new(Box::new(ArrayD::from_elem(vec![4, 3, 1], 3.0))),
                example_labels("samples", vec![0, 3, 6, 8]),
                // different component size
                vec![Arc::new(example_labels("components", vec![0, 1, 2]))],
                Arc::new(example_labels("properties", vec![0])),
            ).unwrap();

            let mut gradient_samples_3 = LabelsBuilder::new(vec!["sample", "parameter"]);
            gradient_samples_3.add(vec![LabelValue::new(1), LabelValue::new(-2)]);
            let gradient_samples_3 = gradient_samples_3.finish();

            block_3.add_gradient(
                "parameter",
                eqs_array_t::new(Box::new(ArrayD::from_elem(vec![1, 3, 1], 13.0))),
                gradient_samples_3,
                vec![Arc::new(example_labels("components", vec![0, 1, 2]))],
            ).unwrap();

            /******************************************************************/

            let mut block_4 = TensorBlock::new(
                eqs_array_t::new(Box::new(ArrayD::from_elem(vec![4, 3, 1], 4.0))),
                example_labels("samples", vec![0, 1, 2, 5]),
                vec![Arc::new(example_labels("components", vec![0, 1, 2]))],
                Arc::new(example_labels("properties", vec![0])),
            ).unwrap();

            let mut gradient_samples_4 = LabelsBuilder::new(vec!["sample", "parameter"]);
            gradient_samples_4.add(vec![LabelValue::new(0), LabelValue::new(1)]);
            gradient_samples_4.add(vec![LabelValue::new(3), LabelValue::new(3)]);
            let gradient_samples_4 = gradient_samples_4.finish();

            block_4.add_gradient(
                "parameter",
                eqs_array_t::new(Box::new(ArrayD::from_elem(vec![2, 3, 1], 14.0))),
                gradient_samples_4,
                vec![Arc::new(example_labels("components", vec![0, 1, 2]))],
            ).unwrap();

            /******************************************************************/

            let mut keys = LabelsBuilder::new(vec!["key_1", "key_2"]);
            keys.add(vec![LabelValue::new(0), LabelValue::new(0)]);
            keys.add(vec![LabelValue::new(1), LabelValue::new(0)]);
            keys.add(vec![LabelValue::new(2), LabelValue::new(2)]);
            keys.add(vec![LabelValue::new(2), LabelValue::new(3)]);
            let keys = keys.finish();

            return TensorMap::new(keys, vec![block_1, block_2, block_3, block_4]).unwrap();
        }

        #[test]
        fn keys_to_properties() {
            let mut tensor = example_tensor();
            tensor.keys_to_properties(&["key_1"], true).unwrap();

            assert_eq!(tensor.keys().count(), 3);
            assert_eq!(tensor.keys().names(), ["key_2"]);
            assert_eq!(tensor.keys()[0], [LabelValue::new(0)]);
            assert_eq!(tensor.keys()[1], [LabelValue::new(2)]);
            assert_eq!(tensor.keys()[2], [LabelValue::new(3)]);

            assert_eq!(tensor.blocks().len(), 3);

            // The new first block contains the old first two blocks merged
            let block_1 = &tensor.blocks()[0];
            assert_eq!(block_1.values.samples().names(), ["samples"]);
            assert_eq!(block_1.values.samples().count(), 5);
            assert_eq!(block_1.values.samples()[0], [LabelValue::new(0)]);
            assert_eq!(block_1.values.samples()[1], [LabelValue::new(1)]);
            assert_eq!(block_1.values.samples()[2], [LabelValue::new(2)]);
            assert_eq!(block_1.values.samples()[3], [LabelValue::new(3)]);
            assert_eq!(block_1.values.samples()[4], [LabelValue::new(4)]);

            assert_eq!(block_1.values.components().len(), 1);
            assert_eq!(block_1.values.components()[0].names(), ["components"]);
            assert_eq!(block_1.values.components()[0].count(), 1);
            assert_eq!(block_1.values.components()[0][0], [LabelValue::new(0)]);

            assert_eq!(block_1.values.properties().names(), ["key_1", "properties"]);
            assert_eq!(block_1.values.properties().count(), 4);
            assert_eq!(block_1.values.properties()[0], [LabelValue::new(0), LabelValue::new(0)]);
            assert_eq!(block_1.values.properties()[1], [LabelValue::new(1), LabelValue::new(3)]);
            assert_eq!(block_1.values.properties()[2], [LabelValue::new(1), LabelValue::new(4)]);
            assert_eq!(block_1.values.properties()[3], [LabelValue::new(1), LabelValue::new(5)]);

            let expected = ArrayD::from_shape_vec(vec![5, 1, 4], vec![
                1.0, 2.0, 2.0, 2.0,
                0.0, 2.0, 2.0, 2.0,
                1.0, 0.0, 0.0, 0.0,
                0.0, 2.0, 2.0, 2.0,
                1.0, 0.0, 0.0, 0.0,
            ]).unwrap();
            assert_eq!(block_1.values.data.as_array(), expected);

            let gradient_1 = block_1.get_gradient("parameter").unwrap();
            assert_eq!(gradient_1.samples().names(), ["sample", "parameter"]);
            assert_eq!(gradient_1.samples().count(), 4);
            assert_eq!(gradient_1.samples()[0], [LabelValue::new(0), LabelValue::new(-2)]);
            assert_eq!(gradient_1.samples()[1], [LabelValue::new(0), LabelValue::new(3)]);
            assert_eq!(gradient_1.samples()[2], [LabelValue::new(3), LabelValue::new(-2)]);
            assert_eq!(gradient_1.samples()[3], [LabelValue::new(4), LabelValue::new(3)]);

            let expected = ArrayD::from_shape_vec(vec![4, 1, 4], vec![
                11.0, 12.0, 12.0, 12.0,
                0.0, 12.0, 12.0, 12.0,
                0.0, 12.0, 12.0, 12.0,
                11.0, 0.0, 0.0, 0.0,
            ]).unwrap();
            assert_eq!(gradient_1.data.as_array(), expected);

            // The new second block contains the old third block
            let block_2 = &tensor.blocks()[1];
            assert_eq!(block_2.values.data.shape().unwrap(), [4, 3, 1]);
            assert_eq!(block_2.values.data.as_array(), ArrayD::from_elem(vec![4, 3, 1], 3.0));

            // The new third block contains the old second block
            let block_3 = &tensor.blocks()[2];
            assert_eq!(block_3.values.data.as_array(), ArrayD::from_elem(vec![4, 3, 1], 4.0));

            // without sorting the samples
            let mut tensor = example_tensor();
            tensor.keys_to_properties(&["key_1"], false).unwrap();

            assert_eq!(tensor.keys().count(), 3);
            assert_eq!(tensor.blocks().len(), 3);

            let block_1 = &tensor.blocks()[0];
            assert_eq!(block_1.values.samples().names(), ["samples"]);
            assert_eq!(block_1.values.samples().count(), 5);
            assert_eq!(block_1.values.samples()[0], [LabelValue::new(0)]);
            assert_eq!(block_1.values.samples()[1], [LabelValue::new(2)]);
            assert_eq!(block_1.values.samples()[2], [LabelValue::new(4)]);
            assert_eq!(block_1.values.samples()[3], [LabelValue::new(1)]);
            assert_eq!(block_1.values.samples()[4], [LabelValue::new(3)]);
        }

        #[test]
        fn keys_to_samples() {
            let mut tensor = example_tensor();
            tensor.keys_to_samples(&["key_2"]).unwrap();

            assert_eq!(tensor.keys().count(), 3);
            assert_eq!(tensor.keys().names(), ["key_1"]);
            assert_eq!(tensor.keys()[0], [LabelValue::new(0)]);
            assert_eq!(tensor.keys()[1], [LabelValue::new(1)]);
            assert_eq!(tensor.keys()[2], [LabelValue::new(2)]);

            assert_eq!(tensor.blocks().len(), 3);

            // The first two blocks are not modified
            let block_1 = &tensor.blocks()[0];
            assert_eq!(block_1.values.data.as_array(), ArrayD::from_elem(vec![3, 1, 1], 1.0));

            let block_2 = &tensor.blocks()[1];
            assert_eq!(block_2.values.data.as_array(), ArrayD::from_elem(vec![3, 1, 3], 2.0));

            // The new third block contains the old third and fourth blocks merged
            let block_3 = &tensor.blocks()[2];
            assert_eq!(block_3.values.samples().names(), ["samples", "key_2"]);
            assert_eq!(block_3.values.samples().count(), 8);
            assert_eq!(block_3.values.samples()[0], [LabelValue::new(0), LabelValue::new(0)]);
            assert_eq!(block_3.values.samples()[1], [LabelValue::new(0), LabelValue::new(2)]);
            assert_eq!(block_3.values.samples()[2], [LabelValue::new(1), LabelValue::new(2)]);
            assert_eq!(block_3.values.samples()[3], [LabelValue::new(2), LabelValue::new(2)]);
            assert_eq!(block_3.values.samples()[4], [LabelValue::new(3), LabelValue::new(0)]);
            assert_eq!(block_3.values.samples()[5], [LabelValue::new(5), LabelValue::new(2)]);
            assert_eq!(block_3.values.samples()[6], [LabelValue::new(6), LabelValue::new(0)]);
            assert_eq!(block_3.values.samples()[7], [LabelValue::new(8), LabelValue::new(0)]);

            assert_eq!(block_3.values.components().len(), 1);
            assert_eq!(block_3.values.components()[0].names(), ["components"]);
            assert_eq!(block_3.values.components()[0].count(), 3);
            assert_eq!(block_3.values.components()[0][0], [LabelValue::new(0)]);
            assert_eq!(block_3.values.components()[0][1], [LabelValue::new(1)]);
            assert_eq!(block_3.values.components()[0][2], [LabelValue::new(2)]);

            assert_eq!(block_3.values.properties().names(), ["properties"]);
            assert_eq!(block_3.values.properties().count(), 1);
            assert_eq!(block_3.values.properties()[0], [LabelValue::new(0)]);

            let expected = ArrayD::from_shape_vec(vec![8, 3, 1], vec![
                3.0, 3.0, 3.0,
                4.0, 4.0, 4.0,
                4.0, 4.0, 4.0,
                4.0, 4.0, 4.0,
                3.0, 3.0, 3.0,
                4.0, 4.0, 4.0,
                3.0, 3.0, 3.0,
                3.0, 3.0, 3.0,
            ]).unwrap();
            assert_eq!(block_3.values.data.as_array(), expected);

            let gradient_3 = block_3.get_gradient("parameter").unwrap();
            assert_eq!(gradient_3.samples().names(), ["sample", "parameter"]);
            assert_eq!(gradient_3.samples().count(), 3);
            assert_eq!(gradient_3.samples()[0], [LabelValue::new(1), LabelValue::new(1)]);
            assert_eq!(gradient_3.samples()[1], [LabelValue::new(4), LabelValue::new(-2)]);
            assert_eq!(gradient_3.samples()[2], [LabelValue::new(5), LabelValue::new(3)]);

            let expected = ArrayD::from_shape_vec(vec![3, 3, 1], vec![
                14.0, 14.0, 14.0,
                13.0, 13.0, 13.0,
                14.0, 14.0, 14.0,
            ]).unwrap();
            assert_eq!(gradient_3.data.as_array(), expected);
        }
    }
}
