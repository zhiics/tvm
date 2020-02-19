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
 * \file tvm/relay/pass/remove_unused_funcs.cc
 * \brief Remove unused global relay functions in a relay module.
 */

#include <tvm/relay/expr.h>
#include <tvm/relay/transform.h>
#include <algorithm>
#include <unordered_set>
#include <vector>

#include "call_graph.h"

namespace tvm {
namespace relay {

/*!
 * \brief Remove functions that are not used.
 *
 * \param module The Relay module.
 * \param entry_funcs The set of functions that can be entry function.
 *
 * \return The module with dead functions removed.
 */
IRModule RemoveUnusedFunctions(const IRModule& module,
                               Array<tvm::PrimExpr> entry_funcs) {
  // Create a CallGraph.
  CallGraph cg(module);
  std::unordered_set<CallGraphNode*> called_funcs;
  for (auto entry : entry_funcs) {
    auto* str_name = entry.as<tir::StringImmNode>();
    if (module->ContainGlobalVar(str_name->value)) {
      const GlobalVar& gv = module->GetGlobalVar(str_name->value);
      const CallGraphNode* cgn = cg[gv];
      const std::vector<CallGraphNode*>& topo = cgn->TopologicalOrder();
      called_funcs.insert(topo.begin(), topo.end());
    }
  }

  // Remove the unused functions in the reverse topological order.
  auto topo = cg.TopologicalOrder();
  std::reverse(topo.begin(), topo.end());
  for (auto* cgn : topo) {
    if (called_funcs.find(cgn) == called_funcs.end()) {
      // Cleanup the CallGraphNode entries.
      cgn->CleanCallGraphEntries();
      // Remove the GlobalVar from the IR module and update the CallGraph.
      cg.RemoveGlobalVarFromModule(cgn, /*update_call_graph*/ true);
    }
  }
  return cg.GetModule();
}

namespace transform {

Pass RemoveUnusedFunctions(Array<tvm::PrimExpr> entry_functions) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
    [=](IRModule m, PassContext pc) {
    return relay::RemoveUnusedFunctions(m, entry_functions);
  };
  return CreateModulePass(pass_func, 1, "RemoveUnusedFunctions", {});
}

TVM_REGISTER_GLOBAL("relay._transform.RemoveUnusedFunctions")
.set_body_typed(RemoveUnusedFunctions);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
