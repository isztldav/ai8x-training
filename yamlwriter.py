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

TODO: Smart processor and data memory allocator
TODO: write_gap, in_dim, out_offset
TODO: Combine convolution and eltwise passthrough layers
TODO: Test passthrough, eltwise, Abs
"""
from typing import Any, Dict, List, Optional, Tuple

import distiller

import ai8x


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
        if count > 64:
            count = 64
        return (1 << count) - 1

    # Pick the first for every quadrant (FIFO compatible) or one every 4
    mult = 16 if count <= 4 else 4
    val: int = 0
    for i in range(count):
        val |= 1 << (mult * i)
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

    # print('SHAPES', shapes)

    all_ops = g.ops
    # print(all_ops)

    # Create the YAML header
    print(
        '---\n'
        '\n'
        f'arch: {arch}\n'
        f'dataset: {dataset}\n'
        '\n'
        'layers:'
    )

    def canonical_name(s: str) -> str:
        separator = s.rfind('.')
        if separator > 0:
            return s[:separator]
        separator = s.rfind('_Div_1')
        if separator > 0:
            return s[:separator]
        return s

    # Pass 1 - Associate inputs and outputs
    input_layer: Dict[str, str] = {}
    output_layer: Dict[str, str] = {}
    final_layer: str = ''

    for e in all_ops:
        name = canonical_name(e)

        for ie in all_ops[e]['inputs']:
            if ie != '':
                input_layer[ie] = name

        for ie in all_ops[e]['outputs']:
            if ie != '':
                output_layer[ie] = name

        if name.startswith('top_level_op'):
            continue

        final_layer = name

    # Pass 2 - Collect inputs and outputs
    inputs: Dict[str, List[str]] = {}
    outputs: Dict[str, List[str]] = {}

    for e in all_ops:
        name = canonical_name(e)

        # Get input names from first of the layer group
        if name not in inputs:
            inputs[name] = []
        for ie in all_ops[e]['inputs']:
            if ie != '' and not ie.endswith('.op.bias') and not ie.endswith('.op.weight') \
               and not ie.endswith('.output_shift'):
                if ie not in output_layer or output_layer[ie] != name:
                    inputs[name].append(ie)

        if name not in outputs:
            outputs[name] = []
        for ie in all_ops[e]['outputs']:
            if ie != '' and (ie not in input_layer or input_layer[ie] != name):
                outputs[name].append(ie)

    # print('INPUTS', inputs)
    # print('OUTPUTS', outputs)
    # print('INPUT_LAYER', input_layer)
    # print('OUTPUT_LAYER', output_layer)

    # Pass 2 - Collect ops
    layers: List[str] = []

    prev_op_name: str = ''

    def chase_inputs(layer: str, ins: List[str]) -> List[str]:
        ret: List[str] = []

        # print('CHASE_INPUTS', layer, ins)
        for ie in ins:
            if ie == layer:
                continue
            n = ie
            if n in output_layer and output_layer[n].startswith('top_level_op'):
                n = output_layer[n]
            if n in output_layer and output_layer[n] != layer:
                ret.append(output_layer[n])

            if n in inputs and inputs[n]:
                val: List[str] = chase_inputs(layer, inputs[n])
                if val:
                    ret += val

        filtered: List[str] = []
        for ie in ret:
            if ie not in filtered:
                filtered.append(ie)

        # print(ins, '->', ret, '->', filtered)
        return filtered

    out_offset = 0  # Start at 0 (default input offset)

    for e in all_ops:
        name = canonical_name(e)
        if name in layers:
            continue
        layers.append(name)
        if name.startswith('top_level_op'):
            continue

        print(f'  - name: {name}')
        ins = inputs[name]

        # Find number of processors needed
        processors: int = 0
        for ie in inputs[name]:
            if ie in shapes:
                processors += shapes[ie][0]

        # Inner layers and more than 16 channels are always HWC
        hwc = hwc or (processors > 16) or (prev_op_name != '')

        # Check input sequences and dimensions
        warn_dim = False
        print_dim = verbose or prev_op_name == ''
        if print_dim:
            print('    # input shape: ', end='')
        i = 0
        max_pixels = MAX_PIXELS if hwc else 4 * MAX_PIXELS
        for ie in inputs[name]:
            if ie in shapes:
                if i > 0 and print_dim:
                    print(', ', end='')
                if print_dim:
                    print(shapes[ie], end='')
                pixels = 1
                for x in range(1, len(shapes[ie])):
                    pixels *= shapes[ie][x]
                if pixels > max_pixels:
                    warn_dim = True
                i += 1
        if print_dim:
            print('')
        if warn_dim:
            print(f'    # dimensions ({pixels} pixels) may require streaming or folding')

        if prev_op_name == '':
            # Show input dimensions and data format for input layers
            print(f'    data_format: {"HWC" if hwc else "CHW"}')
            flatten = False
        else:
            ins = chase_inputs(name, ins)
            if len(ins) > 1 or (len(ins) > 0 and ins[0] != prev_op_name):
                print('    in_sequences: [', end='')
                for i, ie in enumerate(ins):
                    if i > 0:
                        print(', ', end='')
                    print(ie, end='')
                print(']')

            # Check whether inputs are flattened, and whether they were already flattened
            # previously
            flatten = True
            in_dim = None
            prev_dim = None
            for ie in inputs[name]:
                if ie in shapes:
                    mult: int = 1
                    for x in shapes[ie]:
                        mult *= x
                    if shapes[ie][0] != mult:
                        flatten = False
            prev_flatten = True
            for ie in ins:
                if ie in outputs:
                    for je in outputs[ie]:
                        if je in shapes:
                            mult = 1
                            for x in shapes[je]:
                                mult *= x
                            if shapes[je][0] != mult:
                                prev_flatten = False
            flatten = flatten and not prev_flatten

            # Check whether dimensions need to change
            if not flatten:
                for n in inputs[name]:
                    if n in shapes:
                        in_dim = shapes[n][1:]
                        break
                for n in ins:
                    if n in outputs:
                        for o in outputs[n]:
                            if o in shapes:
                                prev_dim = shapes[o][1:]
                                break
                if in_dim is not None and in_dim != prev_dim:
                    if len(in_dim) > 0:
                        in_dim = list(in_dim)
                    print(f'    in_dim: {in_dim}')

        # Mark output layers (the final layer is always an output layer)
        show_output = verbose
        if name != final_layer and any(x not in input_layer for x in outputs[name]):
            print('    output: true')
            show_output = True
        # Show output dimensions for all output layers
        if name == final_layer or show_output:
            print('    # output shape: ', end='')
            i = 0
            for ie in outputs[name]:
                if ie in shapes:
                    if i > 0:
                        print(', ', end='')
                    print(shapes[ie], end='')
                    i += 1
            print('')

        if processors == 0:
            print('    processors: unknown')
        else:
            processors = allocate_processors(name, processors, hwc=hwc)
            print(f'    processors: 0x{processors:016x}')

        out_offset = allocate_offset(name, processors, out_offset)
        print(f'    out_offset: 0x{out_offset:04x}')

        try:
            quantization = int(model.get_parameter(name + '.weight_bits'))
        except ValueError:
            quantization = 8
        if quantization != 8:
            assert quantization in [1, 2, 4]
            print(f'    quantization: {quantization}')

        clamp: str = name + '.clamp'
        if clamp in all_ops and 'value' in all_ops[clamp]['attrs']:
            clamp_val = int(all_ops[clamp]['attrs']['value'])
            assert clamp_val in [-1, -32768]
            if clamp_val == -32768:
                print('    output_width: 32')

        op_name: str = name + '.op'
        if op_name not in all_ops:
            op_name = name
        if op_name in all_ops:
            op = all_ops[op_name]['type']

            # ('dilations', [1, 1])
            # ('strides', [1, 1])

            main_op = 'Passthrough'
            eltwise: int = 1
            if op in ('Add', 'Sub', 'Xor'):
                eltwise = len(ins)
                print('    op: Passthrough')
                print(f'    eltwise: {op}')
                print(f'    operands: {eltwise}')
            elif op in ('Gemm', 'Transpose'):
                print('    op: Linear')
                main_op = 'Linear'
            elif op == 'Conv':
                kernel_size = all_ops[op_name]['attrs']['kernel_shape']
                print(f'    op: Conv{len(kernel_size)}d')
                main_op = op
            elif op == 'ConvTranspose':
                kernel_size = all_ops[op_name]['attrs']['kernel_shape']
                print(f'    op: ConvTranspose{len(kernel_size)}d')
                main_op = op
            else:
                print(f'    op: Unknown ({op})')

            if main_op in ['Conv', 'ConvTranspose']:
                if len(kernel_size) == 1:
                    print(f'    kernel_size: {kernel_size[0]}')
                else:
                    print(f'    kernel_size: {kernel_size[0]}x{kernel_size[1]}')
                pad = all_ops[op_name]['attrs']['pads']
                print('    pad:', pad[0])
                groups = all_ops[op_name]['attrs']['group']
                if groups != 1:
                    print('    groups:', groups)
        else:
            print('    op: Passthrough')

        if flatten:
            print('    flatten: true')

        activate: str = name + '.activate'
        if activate in all_ops:
            print('    activate:', all_ops[activate]['type'])
        elif main_op in ['Conv', 'Linear']:
            print('    activate: None')

        pool: str = name + '.pool_Pad_1'
        if pool not in all_ops:
            pool = name + '.pool'
        if pool in all_ops:
            if all_ops[pool]['type'] == 'Constant':
                pool += '_Pad_1'
            shape = all_ops[pool]['attrs']['kernel_shape']
            if len(shape) == 1 or shape[0] == shape[1]:
                shape = shape[0]
            if all_ops[pool]['type'] == 'MaxPool':
                print('    max_pool:', shape)
            else:
                print('    avg_pool:', shape)
            print('    pool_stride:', all_ops[pool]['attrs']['strides'][0])

        prev_op_name = name
