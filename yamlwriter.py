###################################################################################################
#
# Copyright (C) 2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Generate YAML template from PyTorch model

TODO: Implement smart processor and data memory allocator (out_offset, in_offset)
TODO: Implement dilation
TODO: Testing

NOTE: This code somewhat depends on ai8x.py. Weight and bias parameters are expected to be called
'.op.weight' and 'op.bias'. Quantization information is expected in '.weight_bits' and the output
width is inferred from name + '.clamp'.
"""
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union

import distiller

import ai8x
import devices


def allocate_processors(
        _layer_name: str,
        count: int,
        hwc: bool = False,
) -> int:
    """
    Allocate a given count of processors and return the bit map.
    TODO: This function should be expanded to evenly distribute resources, based on weight and
    data memory utilization.
    """
    if hwc:
        # Inner layers are always HWC
        if count > 64:  # Multi-pass processor count
            divider = (count + 63) // 64
            count = (count + divider - 1) // divider  # Rounded up
            remainder = count % 4
            if remainder != 0:
                remainder = 4 - remainder
            count += remainder  # To next multiple of 4
        assert count <= 64

        # Pick "count" processors starting from 0
        return (1 << count) - 1

    assert count <= 16
    # Pick the first for every quadrant (FIFO compatible) or one every 4
    mult = 16 if count <= 4 else 4
    val: int = 0
    for i in range(count):
        val |= 1 << mult * i
    return val


def allocate_offset(
        _layer_name: str,
        _processor_map: int,
        prev_offset: int = 0,
) -> int:
    """
    Find a data memory offset for the given layer. Currently just uses "ping-pong" from 0 to
    half of the data memory.
    TODO: Implement an algorithm that interacts with weight memories and processor selection,
    and understands element-wise data.
    """
    assert ai8x.dev is not None
    HALF_DATA = 0x4000 if ai8x.dev.device == 85 else 0xa000

    return HALF_DATA if prev_offset == 0 else 0


# pylint: disable=too-many-branches, too-many-statements
def create(
        model: Any,
        dataset: str,
        arch: str,
        hwc: bool = False,
        filename: str = 'template.yaml',
        qat_policy: Optional[str] = None,
        verbose=True,
) -> None:
    """
    Create YAML template
    """
    assert ai8x.dev is not None
    if ai8x.dev.device == 85:
        MAX_PIXELS = 8192
    elif ai8x.dev.device == 87:
        MAX_PIXELS = 20480
    else:
        print(f'Unknown device {ai8x.dev.device}')
        return

    model = distiller.make_non_parallel_copy(model)

    # Apply the QAT policy early to set weight_bits
    ai8x.fuse_bn_layers(model)
    if qat_policy is not None:
        ai8x.initiate_qat(model, qat_policy, export=True)
    ai8x.onnx_export_prep(model, simplify=False, remove_clamp=False)

    dummy_input = distiller.get_dummy_input(dataset=None,
                                            device=distiller.model_device(model),
                                            input_shape=None)
    g = distiller.SummaryGraph(model, dummy_input)

    # Get the input/output dimensions
    shapes: Dict[str, Tuple] = {}
    for i, param in g.params.items():
        shape = param['shape']
        if len(shape) > 1:
            shapes[i] = param['shape'][1:]  # Remove batch dimension

    all_ops: OrderedDict[str, Dict[str, Any]] = g.ops

    def canonical_name(s: str) -> str:
        """
        Return the canonical name for this layer
        """
        separator = s.rfind('.')
        if separator > 0:
            suffix = s[separator + 1:]
            if suffix in ('activate', 'calc_out_scale', 'calc_out_shift',
                          'clamp', 'op', 'pool', 'scale') \
               or suffix.startswith('clamp_') or suffix.startswith('pool_'):
                return s[:separator]
        separator = s.rfind('_Div_1')
        if separator > 0:
            return s[:separator]
        return s

    def ignore_layer(_name: str, layer: Dict[str, Any]) -> bool:
        """
        Remove these layers from the graph
        """
        if not layer['inputs'] or not layer['outputs']:
            return True  # Not consuming or producing anything, useless layer
        return layer['type'] not in (
            'Add', 'Sub', 'Xor',  # element-wise
            'Gemm',  # fc
            'Conv', 'ConvTranspose',
            'MaxPool', 'AveragePool',
            'Abs', 'Relu',
        )

    # 1 - Remove inputs with zero dimensions
    remove_list: List[str] = []
    for n in all_ops:
        s: List[str] = all_ops[n]['inputs']
        all_ops[n]['original_inputs'] = s
        remove: bool = False
        new: List[str] = []
        for i in s:
            if i in shapes:
                new.append(i)
            else:
                remove = True
                remove_list.append(i)
        if remove:
            all_ops[n]['inputs'] = new

        s = all_ops[n]['outputs']
        remove = False
        new = []
        for i in s:
            if i in shapes:
                new.append(i)
            else:
                remove = True
                remove_list.append(i)
        if remove:
            all_ops[n]['outputs'] = new

    # 2 - Remove layers
    ignore_layers: List[str] = []
    for n in all_ops:
        if ignore_layer(n, all_ops[n]):
            ignore_layers.append(n)

    def follow_chain(
            layer: str,
            remove: str,
            replacements: List[str],
            which: str,
    ) -> None:
        for n in all_ops:
            if layer == n:
                continue
            for an_output in remove:
                new_inputs: List[str] = []
                # if all_ops[n][which]:
                #     print(which, 'checking layer', n, 'inputs', all_ops[n]['inputs'],
                #           'outputs:', all_ops[n]['outputs'])
                for an_input in all_ops[n][which]:
                    if an_output == an_input:
                        # print('Found output', an_output,
                        #       'as', which, 'in layer', n,
                        #       'replacing with:', replacements)
                        new_inputs += replacements
                    else:
                        # print('checking', check_input, 'is not replaced')
                        new_inputs.append(an_input)

                filtered: List[str] = []
                for ie in new_inputs:
                    if ie not in filtered:
                        filtered.append(ie)
                if filtered != all_ops[n][which]:
                    # print('replacing', which, 'for layer', n, all_ops[n][which], '->',
                    #       new_inputs)
                    all_ops[n][which] = filtered

    # Fix up inputs after removing a layer
    # c = 0
    for layer in ignore_layers:
        ignore_layer_inputs = all_ops[layer]['inputs']
        ignore_layer_outputs = all_ops[layer]['outputs']
        # print(c, '----------------------------', 'working on removing', layer,
        #       ' with inputs', ignore_layer_inputs, 'and outputs', ignore_layer_outputs)
        # c += 1
        # Look for layer's outputs in the inputs of all other layers and replace
        follow_chain(
            layer,
            remove=ignore_layer_outputs,
            replacements=ignore_layer_inputs,
            which='inputs',
        )

    # 3 - Associate inputs and outputs
    input_layer: Dict[str, str] = {}
    output_layer: Dict[str, str] = {}
    final_layer: str = ''

    for name in all_ops:
        for ie in all_ops[name]['inputs']:
            if ie != '':
                input_layer[ie] = name

        for ie in all_ops[name]['outputs']:
            if ie != '':
                output_layer[ie] = name

        final_layer = name

    # 4 - Collect inputs and outputs
    inputs: Dict[str, List[str]] = {}
    outputs: Dict[str, List[str]] = {}

    for name in all_ops:
        if ignore_layer(name, all_ops[name]):
            continue

        # Get input names from first of the layer group
        if name not in inputs:
            inputs[name] = []
        for ie in all_ops[name]['inputs']:
            # Ignore weights and biases (uses hard-coded names from ai8x.py)
            if ie.endswith('.output_shift'):
                print('ALERT: found output_shift in input to layer', name)
            if ie != '' and not ie.endswith('.op.bias') and not ie.endswith('.op.weight') \
               and not ie.endswith('.output_shift'):
                if ie not in output_layer or output_layer[ie] != name:
                    inputs[name].append(ie)

        if name not in outputs:
            outputs[name] = []
        for ie in all_ops[name]['outputs']:
            if ie != '' and (ie not in input_layer or input_layer[ie] != name):
                outputs[name].append(ie)

    # 5 - Collect ops
    prev_op_name: str = ''
    layers: Dict[str, Dict[str, Any]] = {}
    input_hwc: bool = False

    for name in all_ops:
        if ignore_layer(name, all_ops[name]):
            continue
        canonical = canonical_name(name)
        this_layer: Dict[str, Any] = {}

        this_layer['name'] = canonical
        ins = inputs[name]

        # Mark output layers (the final layer is always an output layer)
        if name != final_layer and any(x not in input_layer for x in outputs[name]):
            this_layer['output'] = 'true'

        this_layer['op'] = 'Passthrough'
        main_op = 'Passthrough'
        operands: int = 1

        op = all_ops[name]['type']

        if op in ('Add', 'Sub', 'Xor'):
            operands = len(ins)
            this_layer['eltwise'] = op
            this_layer['operands'] = operands
        elif op == 'Gemm':
            this_layer['op'] = 'Linear'
            main_op = 'Linear'
            this_layer['activate'] = 'None'
        elif op == 'Conv':
            kernel_size = all_ops[name]['attrs']['kernel_shape']
            this_layer['op'] = f'Conv{len(kernel_size)}d'
            this_layer['activate'] = 'None'
            main_op = op
        elif op == 'ConvTranspose':
            kernel_size = all_ops[name]['attrs']['kernel_shape']
            this_layer['op'] = f'ConvTranspose{len(kernel_size)}d'
            this_layer['activate'] = 'None'
            main_op = op
        elif op in ('MaxPool', 'AveragePool'):
            shape = all_ops[name]['attrs']['kernel_shape']
            if len(shape) == 1 or shape[0] == shape[1]:
                shape = shape[0]
            if all_ops[name]['type'] == 'MaxPool':
                this_layer['max_pool'] = shape
            else:
                this_layer['avg_pool'] = shape
            this_layer['pool_stride'] = all_ops[name]['attrs']['strides'][0]
        elif op in ('Abs', 'Relu'):
            this_layer['op'] = op
            this_layer['activate'] = op
        else:
            this_layer['op'] = f'Unknown ({op})'

        this_layer['main_op'] = main_op

        if main_op in ('Conv', 'ConvTranspose'):
            if len(kernel_size) == 1:
                this_layer['kernel_size'] = str(kernel_size[0])
            else:
                this_layer['kernel_size'] = f'{kernel_size[0]}x{kernel_size[1]}'
            pad = all_ops[name]['attrs']['pads']
            this_layer['pad'] = pad[0]
            groups = all_ops[name]['attrs']['group']
            if groups != 1:
                this_layer['groups'] = groups

            # Quantization uses hard-coded name from ai8x.py
            try:
                quantization = int(model.get_parameter(canonical + '.weight_bits'))
                if quantization == 0:
                    quantization = 8
            except AttributeError:
                quantization = 8
            if quantization != 8:
                assert quantization in [1, 2, 4], f'{name}: quantization={quantization}'
                this_layer['quantization'] = quantization

        # Clamping? Using hard-coded name from ai8x.py
        # Also use the hard-coded bias name to record whether a bias is used in the convolution.
        if main_op in ('Conv', 'ConvTranspose', 'Linear'):
            clamp: str = canonical + '.clamp'
            if clamp in all_ops and 'value' in all_ops[clamp]['attrs']:
                clamp_val = int(all_ops[clamp]['attrs']['value'])
                assert clamp_val in [-1, -32768]
                if clamp_val == -32768:
                    this_layer['output_width'] = 32

            have_bias: bool = False
            for e in all_ops[name]['inputs']:
                if e.endswith('.op.bias'):
                    have_bias = True
                    break
            this_layer['have_bias'] = have_bias

        # Check whether inputs need to be flattened (when they are not in 1x1 dimensions)
        flatten: bool = False
        if main_op == 'Linear':
            for ie in inputs[name]:
                if ie in shapes:
                    mult: int = 1
                    for x in shapes[ie]:
                        mult *= x
                    if shapes[ie][0] != mult:
                        flatten = True
            if flatten:
                this_layer['flatten'] = 'true'

        in_dim: Optional[Union[Tuple[int, ...], List[int]]] = None
        prev_dim: Optional[Union[Tuple[int, ...], List[int]]] = None

        # Check whether dimensions need to change
        if not flatten:
            mult = 1
            for n in inputs[name]:
                if n in shapes:
                    for x in shapes[n]:
                        mult *= x
                    if len(shapes[n]) > 1 and mult != shapes[n][0]:
                        in_dim = shapes[n][1:]
                    else:
                        in_dim = shapes[n]
                    break
            mult = 1
            for n in all_ops[name]['original_inputs']:
                if n in shapes:
                    for x in shapes[n]:
                        mult *= x
                    if len(shapes[n]) > 1 and mult != shapes[n][0]:
                        prev_dim = shapes[n][1:]
                    else:
                        prev_dim = shapes[n]
                    break
            if in_dim is not None and prev_dim is not None:
                if in_dim != prev_dim \
                   and (in_dim[0] != prev_dim[0] or len(prev_dim) > 1 or mult != prev_dim[0]):
                    if len(prev_dim) > 0:
                        prev_dim = list(prev_dim)
                    this_layer['in_dim'] = prev_dim

        # Find number of processors needed
        processors: int = 0
        for ie in inputs[name]:
            if ie in shapes:
                processors += shapes[ie][0]
                if operands == 1:
                    break  # Element-wise: data is interleaved, not concatenated
        this_layer['proc_count'] = processors

        # Inner layers and more than 16 channels are always HWC
        hwc = hwc or (processors > 16) or (prev_op_name != '')

        if prev_op_name == '':
            # Show input dimensions and data format for input layers
            this_layer['data_format'] = hwc
            input_hwc = hwc
        else:
            # Don't set in_sequences when using strictly sequential single inputs
            # print('in_sequences candidates for', name, ins, end='')
            ins = [output_layer[x] for x in ins]
            # print(' translated to layers:', ins)
            if len(ins) > 1 or (len(ins) > 0 and ins[0] != prev_op_name):
                if operands > 1:
                    ins.reverse()  # Reverse the list since PyTorch does it backwards
                this_layer['in_sequences'] = ins

        prev_op_name = name
        layers[name] = this_layer

    prev_name: str = ''
    pop_list: List[Tuple[str, str]] = []
    veto: bool = False
    prev: Dict[str, Any] = {}

    # 6a - Merge (fuse) conv and activation layers
    for count, (name, ll) in enumerate(layers.items()):
        if ll['op'] in ('Abs', 'Relu') and prev_name != '':
            prev = layers[prev_name]
            if prev['main_op'] not in ('Conv', 'Linear'):
                continue
            # Check that no layer other than the activation layer uses the intermediate output of
            # the conv layer as an input
            veto = False
            for (other_name, ol) in layers.items():
                if other_name != name and other_name != prev_name and 'in_sequences' in ol:
                    if prev_name in ol['in_sequences']:
                        veto = True
                        break
            if veto:
                continue
            # Combine both layers
            if 'comment' not in prev:
                prev['comment'] = f'{prev_name} fused with {name}'
            else:
                prev['comment'] += f' and {name}'
            pop_list.append((prev_name, name))  # Mark second layer for deletion
            # Copy over convolution operation and keep the element-wise operation in place
            prev['activate'] = ll['activate']
            if 'output' in ll:
                prev['output'] = ll['output']
            if 'quantization' in ll:
                prev['quantization'] = ll['quantization']
            if 'output_width' in ll:
                prev['output_width'] = ll['output_width']
            outputs[prev_name] = outputs[name]
            inputs[name] = inputs[prev_name]
            layers[prev_name] = prev

        prev_name = name

    # Delete the conv layers that were fused into the eltwise layer
    for (prev_name, name) in pop_list:
        # Change any dangling input sequences to the fused layer
        for (other_name, ol) in layers.items():
            if other_name != name and other_name != prev_name and 'in_sequences' in ol:
                for i, e in enumerate(ol['in_sequences']):
                    if e == name:
                        ol['in_sequences'][i] = prev_name

        # Delete the conv portion
        layers.pop(name)

    # 6b - Merge (fuse) pooling and conv layers
    prev_name = ''
    pop_list = []
    veto = False
    prev = {}

    for count, (name, ll) in enumerate(layers.items()):
        if ll['main_op'] in ('Conv', 'Linear') and prev_name != '':
            prev = layers[prev_name]
            if prev['op'] is not 'Passthrough' or (
                'max_pool' not in prev and 'avg_pool' not in prev
            ):
                continue
            # Check that no layer other than the activation layer uses the intermediate output of
            # the conv layer as an input
            veto = False
            for (other_name, ol) in layers.items():
                if other_name != name and other_name != prev_name and 'in_sequences' in ol:
                    if prev_name in ol['in_sequences']:
                        veto = True
                        break
            if veto:
                continue
            # Combine both layers
            if 'comment' not in ll:
                prev['comment'] = f'{prev_name} fused with {name}'
            else:
                prev['comment'] = f'{prev_name} and ' + ll['comment']
            pop_list.append((prev_name, name))  # Mark second layer for deletion
            # Copy over convolution operation
            prev['op'] = ll['op']
            prev['main_op'] = ll['main_op']
            if 'output' in ll:
                prev['output'] = ll['output']
            if 'quantization' in ll:
                prev['quantization'] = ll['quantization']
            if 'output_width' in ll:
                prev['output_width'] = ll['output_width']
            if 'kernel_size' in ll:
                prev['kernel_size'] = ll['kernel_size']
            if 'pad' in ll:
                prev['pad'] = ll['pad']
            if 'groups' in ll:
                prev['groups'] = ll['groups']
            if 'activate' in ll:
                prev['activate'] = ll['activate']
            if 'have_bias' in ll:
                prev['have_bias'] = ll['have_bias']
            outputs[prev_name] = outputs[name]
            inputs[name] = inputs[prev_name]
            layers[prev_name] = prev

        prev_name = name

    # Delete the conv layers that were fused into the eltwise layer
    for (prev_name, name) in pop_list:
        # Change any dangling input sequences to the fused layer
        for (other_name, ol) in layers.items():
            if other_name != name and other_name != prev_name and 'in_sequences' in ol:
                for i, e in enumerate(ol['in_sequences']):
                    if e == name:
                        ol['in_sequences'][i] = prev_name

        # Delete the conv portion
        layers.pop(name)

    # 6c - Merge (fuse) element-wise and convolution layers
    prev_name = ''
    pop_list = []
    for count, (name, ll) in enumerate(layers.items()):
        if ll['main_op'] == 'Conv' and prev_name != '':
            prev = layers[prev_name]
            # Only one pooling operation possible
            pool_count: int = 0
            if 'max_pool' in prev:
                pool_count += 1
            if 'avg_pool' in prev:
                pool_count += 1
            if 'max_pool' in ll:
                pool_count += 1
            if 'avg_pool' in ll:
                pool_count += 1
            # Check that no layer other than the conv layer uses the intermediate output of the
            # element-wise layer as an input
            veto = False
            for (other_name, ol) in layers.items():
                if other_name != name and other_name != prev_name and 'in_sequences' in ol:
                    if prev_name in ol['in_sequences']:
                        veto = True
                        break
            if veto:
                continue
            if 'in_sequences' in ll or 'in_dim' in ll or 'flatten' in ll:
                continue
            if prev['main_op'] != 'Passthrough' or 'operands' not in prev \
               or prev['operands'] == 1 or pool_count > 1:
                continue
            # MAX78002 - avoid element-wise with bias and convolution when using multi-pass
            if ai8x.dev.device == 87 and ll['have_bias'] and prev['proc_count'] > 64:
                continue
            # Combine both layers
            if 'comment' not in ll:
                prev['comment'] = f'{prev_name} fused with {name}'
            else:
                prev['comment'] = f'{prev_name} and ' + ll['comment']
            pop_list.append((prev_name, name))  # Mark second layer for deletion
            # Copy over convolution operation and keep the element-wise operation in place
            prev['op'] = ll['op']
            prev['main_op'] = ll['main_op']
            if 'output' in ll:
                prev['output'] = ll['output']
            if 'quantization' in ll:
                prev['quantization'] = ll['quantization']
            if 'output_width' in ll:
                prev['output_width'] = ll['output_width']
            if 'kernel_size' in ll:
                prev['kernel_size'] = ll['kernel_size']
            if 'pad' in ll:
                prev['pad'] = ll['pad']
            if 'groups' in ll:
                prev['groups'] = ll['groups']
            if 'activate' in ll:
                prev['activate'] = ll['activate']
            if 'max_pool' in ll:
                prev['max_pool'] = ll['max_pool']
                prev['pool_first'] = 'false'
            if 'avg_pool' in ll:
                prev['avg_pool'] = ll['avg_pool']
                prev['pool_first'] = 'false'
            if 'pool_stride' in ll:
                prev['pool_stride'] = ll['pool_stride']
            outputs[prev_name] = outputs[name]
            inputs[name] = inputs[prev_name]

        prev_name = name

    # Delete the conv layers that were fused into the eltwise layer
    for (prev_name, name) in pop_list:
        # Change any dangling input sequences to the fused layer
        for (other_name, ol) in layers.items():
            if other_name != name and other_name != prev_name and 'in_sequences' in ol:
                for i, e in enumerate(ol['in_sequences']):
                    if e == name:
                        ol['in_sequences'][i] = prev_name

        # Delete the conv portion
        layers.pop(name)

    # 7 - Insert passthrough layers for write_gap
    write_gap_list: List[Tuple[str, int]] = []
    insert_list: List[Tuple[str, int]] = []
    source_list: List[Tuple[str, str]] = []
    for (name, ll) in layers.items():
        # There are two cases: element-wise operations ('operands' defined and > 1), and conv
        # operations with multi-pass (> 64 input channels). In either case, in_sequences has
        # more than one member.
        # print('*', name, ll['in_sequences'] if 'in_sequences' in ll else '--')
        if 'in_sequences' not in ll or len(ll['in_sequences']) < 2:
            continue
        operands = ll['operands'] if 'operands' in ll else 1
        # print('still here, have', operands, 'operands', 'and a proc_count of', ll['proc_count'])
        if operands == 1:
            operands = len(ll['in_sequences'])
            # Concat instead of interleave for small channel counts
            if ll['proc_count'] * operands <= 64:
                ll['proc_count'] *= operands
                continue
        # print('checking in_sequences for layer', name, ll['in_sequences'], 'with', operands,
        #       'operands')
        # For each input, check whether anybody else is using the input. If yes, insert a dummy
        # layer that creates a write_gap version of the data. If no, add the write_gap to the
        # producer.
        for source in ll['in_sequences']:
            must_insert: bool = False
            prev_name = ''
            for (other_name, ol) in layers.items():
                if other_name not in (name, source):
                    if 'in_sequences' in ol:
                        for e in ol['in_sequences']:
                            if e == source:
                                must_insert = True
                    elif prev_name == source:
                        must_insert = True
                        # Break the sequence
                        ol['in_sequences'] = [source]
                prev_name = other_name
            # print('must_insert for', source, 'is', must_insert)
            if not must_insert:
                # The source is used only by the element-wise layer, so we can insert the write gap
                # directly
                write_gap_list.append((source, operands))
            else:
                insert_list.append((source, operands))
                # Replace source with source_gap in layers[name]['in_sequences']
                source_list.append((name, source))
                new_name = 'gap_' + source
                # ...and insert shaope information (input and output are both the same as the
                # original layer's output)
                inputs[new_name] = [source + '_data']
                outputs[new_name] = [name + '_data']
                for ie in outputs[source]:
                    if ie in shapes:
                        shapes[name + '_data'] = shapes[ie]
                        shapes[source + '_data'] = shapes[ie]

    # Insert simple write gaps
    for (name, gap) in write_gap_list:
        layers[name]['write_gap'] = gap - 1
    # Break sequence
    for (name, source) in source_list:
        seq = layers[name]['in_sequences']
        for i, s in enumerate(seq):
            if s == source:
                seq[i] = 'gap_' + source
    # Insert additional layers
    for (name, gap) in insert_list:
        new_layer: Dict[str, Any] = {}
        new_name = 'gap_' + name
        new_layer['name'] = new_name
        processors = 0
        for ie in inputs[new_name]:
            if ie in shapes:
                processors += shapes[ie][0]
        new_layer['proc_count'] = processors
        new_layer['op'] = 'Passthrough'
        new_layer['name'] = 'gap_' + name
        new_layer['write_gap'] = gap - 1

        # Insert into dict via list
        insert_pos: int = list(layers.keys()).index(name) + 1
        layers_list: List[Any] = list(layers.items())
        layers_list.insert(insert_pos, (new_name, new_layer))
        layers = dict(layers_list)

    # 8 - TODO: Assign processors and output_offset
    out_offset = 0  # Start at 0 (default input offset)
    for (name, ll) in layers.items():
        processors = ll['proc_count']
        hwc = ll['data_format'] if 'data_format' in ll else True
        if processors == 0:
            ll['processors'] = 0  # Unknown
        else:
            processors = allocate_processors(name, processors, hwc=hwc)
            ll['processors'] = processors

        out_offset = allocate_offset(name, processors, out_offset)
        ll['out_offset'] = out_offset

    # 9 - Print
    with open(filename, mode='w', encoding='utf-8') as f:
        f.write(
            '---\n'
            '# YAML template -- requires manual editing, particularly with regard to out_offset '
            'and processors\n'
            f'# Generated for {devices.partnum(ai8x.dev.device)} with input format '
            f'{"HWC" if input_hwc else "CHW"}\n\n'
            f'arch: {arch}\n'
            f'dataset: {dataset}\n'
            '\n'
            'layers:\n'
        )

        prev_name = ''
        for count, (name, ll) in enumerate(layers.items()):
            f.write('\n'
                    f'  # Layer {count}\n'
                    f'  - name: {canonical_name(name)}')
            if 'comment' in ll:
                f.write(f"  # {ll['comment']}")
            f.write('\n')

            hwc = ll['data_format'] if 'data_format' in ll else True

            # Check input sequences and dimensions
            warn_dim = False
            print_dim = verbose or prev_name == ''
            if print_dim:
                f.write('    # input shape: ')
            i = 0
            max_pixels = MAX_PIXELS if hwc else 4 * MAX_PIXELS
            for ie in inputs[name]:
                if ie in shapes:
                    if i > 0 and print_dim:
                        f.write(', ')
                    if print_dim:
                        f.write(str(shapes[ie]))
                    pixels = 1
                    for x in range(1, len(shapes[ie])):
                        pixels *= shapes[ie][x]
                    if pixels > max_pixels:
                        warn_dim = True
                    i += 1
            if print_dim:
                f.write('\n')
            if warn_dim:
                f.write(f'    # dimensions ({pixels} pixels) may require streaming or folding\n')

            if 'data_format' in ll:
                f.write(f'    data_format: {"HWC" if hwc else "CHW"}\n')
            if 'in_sequences' in ll:
                f.write('    in_sequences: [')
                ins = ll['in_sequences']
                for i, ie in enumerate(ins):
                    if i > 0:
                        f.write(', ')
                    f.write(canonical_name(ie))
                f.write(']\n')
            if 'in_dim' in ll:
                f.write(f"    in_dim: {ll['in_dim']}\n")
            show_output = verbose
            if 'output' in ll:
                f.write(f"    output: {ll['output']}\n")
                show_output = True
            processors = ll['processors']
            if processors == 0:
                f.write('    processors: unknown\n')
            else:
                f.write(f'    processors: 0x{processors:016x}\n')
            f.write(f"    out_offset: 0x{ll['out_offset']:04x}\n")
            if 'quantization' in ll:
                f.write(f"    quantization: {ll['quantization']}\n")
            if 'output_width' in ll:
                f.write(f"    output_width: {ll['output_width']}\n")
            f.write(f"    op: {ll['op']}\n")
            if 'eltwise' in ll:
                f.write(f"    eltwise: {ll['eltwise']}\n")
            if 'operands' in ll:
                f.write(f"    operands: {ll['operands']}\n")
            if 'kernel_size' in ll:
                f.write(f"    kernel_size: {ll['kernel_size']}\n")
            if 'pad' in ll:
                f.write(f"    pad: {ll['pad']}\n")
            if 'groups' in ll:
                f.write(f"    groups: {ll['groups']}\n")
            if 'flatten' in ll:
                f.write(f"    flatten: {ll['flatten']}\n")
            if 'activate' in ll:
                f.write(f"    activate: {ll['activate']}\n")
            if 'pool_first' in ll:
                f.write(f"    pool_first: {ll['pool_first']}\n")
            if 'max_pool' in ll:
                f.write(f"    max_pool: {ll['max_pool']}\n")
            if 'avg_pool' in ll:
                f.write(f"    avg_pool: {ll['avg_pool']}\n")
            if 'pool_stride' in ll:
                f.write(f"    pool_stride: {ll['pool_stride']}\n")
            if 'write_gap' in ll:
                f.write(f"    write_gap: {ll['write_gap']}\n")

            # Show output dimensions for all output layers
            if name == final_layer or show_output:
                f.write('    # output shape: ')
                i = 0
                for ie in outputs[name]:
                    if ie in shapes:
                        if i > 0:
                            f.write(', ')
                        f.write(str(shapes[ie]))
                        i += 1
                f.write('\n')

            prev_name = name
