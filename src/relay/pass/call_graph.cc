/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/relay/pass/call_graph.cc
 * \brief Get the call graph of a Relay module.
 */

#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/support/logging.h>
#include <tvm/relay/transform.h>
#include <tvm/ir/transform.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace relay {
namespace call_graph {

using GlobalVarSet = std::unordered_set<GlobalVar, ObjectHash, ObjectEqual>;
using CallGraphMap = std::unordered_map<GlobalVar, GlobalVarSet, ObjectHash, ObjectEqual>;
using CallGraphFunctor = ExprFunctor<void(const Expr& e, const GlobalVar& gv)>;

class CallGraphCollector : public CallGraphFunctor {
 public:
  explicit CallGraphCollector(const IRModule& module) : module_(module) {}

  void VisitExpr_(const CallNode* call_node, const GlobalVar& gv) {
    Expr op = call_node->op;

    if (const auto* global = op.as<GlobalVarNode>()) {
      call_graph_[gv].emplace(GetRef<GlobalVar>(global));
    }
    return CallGraphFunctor::VisitExpr(GetRef<Call>(call_node), gv);
  }

  CallGraphMap Collect() {
    auto gvar_funcs = module_->functions;
    for (auto pair : gvar_funcs) {
      auto base_func = pair.second;
      if (auto* n = base_func.as<FunctionNode>()) {
        auto func = GetRef<Function>(n);
        CallGraphFunctor::VisitExpr(func, pair.first);
      }
    }
    return call_graph_;
  }

 private:
  IRModule module_;
  CallGraphMap call_graph_;
};

}  // namespace call_graph

call_graph::CallGraphMap CallGraph(const IRModule& m) {
  return relay::call_graph::CallGraphCollector(m).Collect();
}

}  // namespace relay
}  // namespace tvm
