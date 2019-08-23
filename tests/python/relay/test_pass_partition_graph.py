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
"""Unit tests for graph partitioning."""
import numpy as np

import tvm
from tvm import relay
import tvm.relay.testing
import tvm.relay.transform
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.annotation import subgraph_begin, subgraph_end

class MyAnnotator(ExprMutator):
    def visit_call(self, call):
        print(call.op.name)
        if call.op.name == "subtract":
            lhs = subgraph_begin(call.args[0], "gcc")
            rhs = subgraph_begin(call.args[1], "gcc")
            op = relay.subtract(lhs, rhs)
            return subgraph_end(op, "gcc")

        return super().visit_call(call)

    def visit_function(self, func):
        return relay.Function(func.params, self.visit(func.body))

def annotate(expr):
    ann = MyAnnotator()
    return ann.visit(expr)

def test_subgraph():
    x = relay.var('x', shape=(10, 10))
    y = relay.var('y', shape=(10, 10))
    z = x + x
    f = relay.Function([x, y], y - z)
    x_data = np.random.rand(10, 10).astype('float32')
    y_data = np.random.rand(10, 10).astype('float32')
    mod = relay.Module()
    mod["main"] = annotate(f)
    mod = relay.transform.PartitionGraph()(mod)
    mod = relay.transform.InferType()(mod)
    print(mod['main'])
    ex = relay.create_executor("debug", mod=mod, ctx=tvm.cpu(0))
    res = ex.evaluate()(x_data, y_data)
    tvm.testing.assert_allclose(res.asnumpy(), y_data - (x_data + x_data))

def test_extern():
    x = relay.var('x', shape=(10, 10))
    y = relay.var('y', shape=(10, 10))
    z = x + x
    f = relay.Function([x, y], y - z)
    x_data = np.random.rand(10, 10).astype('float32')
    y_data = np.random.rand(10, 10).astype('float32')
    mod = relay.Module()
    mod["main"] = f
    mod = relay.transform.Sequential([relay.transform.ExternOp("gcc"),
                                      relay.transform.PartitionGraph()])(mod)
    print(mod['main'])
    ex = relay.create_executor("debug", mod=mod, ctx=tvm.cpu(0))
    res = ex.evaluate()(x_data, y_data)
    tvm.testing.assert_allclose(res.asnumpy(), y_data - (x_data + x_data))

if __name__ == "__main__":
    test_subgraph()
    test_extern()