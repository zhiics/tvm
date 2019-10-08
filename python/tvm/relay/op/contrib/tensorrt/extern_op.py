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
"""TensorRT compiler supported operators."""
from __future__ import absolute_import

def relu(attrs, args):
    return True

def sigmoid(attrs, args):
    return True

def tanh(attrs, args):
    return True

def clip(attrs, args):
    return True

def leaky_relu(attrs, args):
    return True

def batch_norm(attrs, args):
    return True

def softmax(attrs, args):
    return True

def conv2d(attrs, args):
    return True

def dense(attrs, args):
    return True

def bias_add(attrs, args):
    return True

def add(attrs, args):
    return True

def subtract(attrs, args):
    return True

def multiply(attrs, args):
    return True

def divide(attrs, args):
    return True

def power(attrs, args):
    return True

def max_pool2d(attrs, args):
    return True

def avg_pool2d(attrs, args):
    return True

def global_max_pool2d(attrs, args):
    return True

def global_avg_pool2d(attrs, args):
    return True

def exp(attrs, args):
    return True

def log(attrs, args):
    return True

def sqrt(attrs, args):
    return True

def abs(attrs, args):
    return True

def negative(attrs, args):
    return True

def sin(attrs, args):
    return True

def cos(attrs, args):
    return True

def atan(attrs, args):
    return True

def ceil(attrs, args):
    return True

def floor(attrs, args):
    return True

def batch_flatten(attrs, args):
    return True

def expand_dims(attrs, args):
    return True

def concatenate(attrs, args):
    return True
