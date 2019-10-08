# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument
"""
External compiler related feature registration.

It implements dispatchers that check if an operator should use the external
codegen tool.

Each compiler can customize the support of the operator. For example, they can
check the attribute of an operator and/or the features of the input arguments
to decide if we should use the external compiler.
"""
from __future__ import absolute_import

import logging
import pkgutil
from pathlib import Path
from importlib import import_module

from .. import op as reg

logger = logging.getLogger('ExternOp')

# Load available contrib compilers
compilers = {}
for _, name, _ in pkgutil.iter_modules([Path(__file__).parent]):
    compilers[name] = import_module(
        '.%s' % name, package='.'.join(__name__.split('.')[:-1]))


def get_extern_op(compiler, op_name):
    """Get the extern op function from the registered compiler
    """
    if compiler in compilers:
        if hasattr(compilers[compiler], 'extern_op'):
            extern_op = getattr(compilers[compiler], 'extern_op')
            if hasattr(extern_op, op_name):
                return getattr(extern_op, op_name)

    logger.warning("%s in %s is not registered. Fallback to CPU", op_name,
                   compiler)
    return lambda x, y: False


@reg.register_extern_op("nn.conv2d")
def external_conv2d(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'conv2d')(attrs, args)


@reg.register_extern_op("nn.dense")
def external_dense(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'dense')(attrs, args)


@reg.register_extern_op("nn.relu")
def external_relu(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'relu')(attrs, args)


@reg.register_extern_op("nn.batch_norm")
def external_batch_norm(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'batch_norm')(attrs, args)


@reg.register_extern_op("subtract")
def external_subtract(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'subtract')(attrs, args)


@reg.register_extern_op("add")
def external_add(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'add')(attrs, args)


@reg.register_extern_op("multiply")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'multiply')(attrs, args)

@reg.register_extern_op("sigmoid")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'sigmoid')(attrs, args)

@reg.register_extern_op("tanh")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'tanh')(attrs, args)

@reg.register_extern_op("clip")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'clip')(attrs, args)

@reg.register_extern_op("nn.leaky_relu")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'leaky_relu')(attrs, args)

@reg.register_extern_op("nn.softmax")
def external_batch_norm(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'softmax')(attrs, args)

@reg.register_extern_op("nn.bias_add")
def external_batch_norm(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'bias_add')(attrs, args)

@reg.register_extern_op("subtract")
def external_batch_norm(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'subtract')(attrs, args)

@reg.register_extern_op("divide")
def external_batch_norm(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'divide')(attrs, args)

@reg.register_extern_op("power")
def external_batch_norm(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'power')(attrs, args)

@reg.register_extern_op("nn.max_pool2d")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'max_pool2d')(attrs, args)

@reg.register_extern_op("nn.avg_pool2d")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'avg_pool2d')(attrs, args)

@reg.register_extern_op("nn.global_max_pool2d")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'global_max_pool2d')(attrs, args)

@reg.register_extern_op("nn.global_avg_pool2d")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'global_avg_pool2d')(attrs, args)

@reg.register_extern_op("exp")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'exp')(attrs, args)

@reg.register_extern_op("log")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'log')(attrs, args)

@reg.register_extern_op("sqrt")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'sqrt')(attrs, args)

@reg.register_extern_op("abs")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'abs')(attrs, args)

@reg.register_extern_op("negative")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'negative')(attrs, args)

@reg.register_extern_op("sin")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'sin')(attrs, args)

@reg.register_extern_op("cos")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'cos')(attrs, args)

@reg.register_extern_op("atan")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'atan')(attrs, args)

@reg.register_extern_op("ceil")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'ceil')(attrs, args)

@reg.register_extern_op("floor")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'floor')(attrs, args)

@reg.register_extern_op("nn.batch_flatten")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'batch_flatten')(attrs, args)

@reg.register_extern_op("expand_dims")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'expand_dims')(attrs, args)

@reg.register_extern_op("concatenate")
def external_multiply(attrs, args, compiler):
    """Check if the external compiler should be used.
    """
    return get_extern_op(compiler, 'concatenate')(attrs, args)

