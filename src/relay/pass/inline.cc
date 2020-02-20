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
 * \file tvm/relay/pass/inline.cc
 * \brief Global function inliner. It contains the following steps:
 *
 *  - Preprocessing: eligibility checking.
 *
 *  - Inline:
 *
 *  - Postprocessing: remove unused functions.
 */

#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/support/logging.h>
#include <tvm/relay/transform.h>
#include <string>
#include <unordered_set>

#include "call_graph.h"

using namespace tvm::runtime;

namespace tvm {
namespace relay {

class Inliner : ExprMutator {
 public:
  explicit Inliner(CallGraphNode* cur_node, CallGraph* call_graph)
      : cur_node_(cur_node), call_graph_(call_graph) {}

  Expr VisitExpr_(const CallNode* call_node) final {
    Expr op = call_node->op;
    const auto* gvn = op.as<GlobalVarNode>();

    if (gvn) {
      GlobalVar gv = GetRef<GlobalVar>(gvn);
      CallGraphNode* cg_node = (*call_graph_)[gv->name_hint];
      if (CanInline(cg_node)) {
        tvm::Array<Expr> call_args;
        for (auto arg : call_node->args) {
          auto new_arg = VisitExpr(arg);
          call_args.push_back(new_arg);
        }
        cur_node_->RemoveCallTo(gv);
        auto func = MakeLocalFunction(gv);
        return CallNode::make(func, call_args, call_node->attrs, call_node->type_args);
      }
    }
    return ExprMutator::VisitExpr_(call_node);
  }

  Expr VisitExpr_(const GlobalVarNode* gvn) final {
    GlobalVar gv = GetRef<GlobalVar>(gvn);
    CallGraphNode* cg_node = (*call_graph_)[gv->name_hint];
    if (CanInline(cg_node)) {
      cur_node_->RemoveCallTo(gv);
      return MakeLocalFunction(gv);
    }
    return ExprMutator::VisitExpr_(gvn);
  }

  Function MakeLocalFunction(const GlobalVar& global) {
    auto base_func = call_graph_->GetModule()->Lookup(global);
    const auto* func = base_func.as<FunctionNode>();
    CHECK(func) << "Expected to work on a Relay function.";
    return FunctionNode::make(func->params,
                              func->body,
                              func->ret_type,
                              func->type_params,
                              func->attrs);
  }

  Function Inline(const Function& func) {
    return FunctionNode::make(func->params,
                              VisitExpr(func->body),
                              func->ret_type,
                              func->type_params,
                              func->attrs);
  }

 private:
  bool CanInline(const CallGraphNode* cg_node) {
    // The node must be a leaf node and it cannot be recursive.
    if (!cg_node->empty() || cg_node->IsRecursive()) return false;

    auto base_func = call_graph_->GetModule()->Lookup(cg_node->GetNameHint());
    auto func = Downcast<Function>(base_func);
    // The body of a global functions must be defined.
    if (!func->body.defined()) return false;

    // The function must be annotated with the inline attribute.
    ObjectRef res = FunctionGetAttr(func, attr::kCompiler);
    const tir::StringImmNode* pval = res.as<tir::StringImmNode>();
    if (!pval) return false;

    // The function is not abled to be inlined if any callee under the CallGraph
    // of this function cannot be inlined.
    for (const auto& it : *cg_node) {
      if (!CanInline(it.second)) {
        return false;
      }
    }

    return true;
  }

  CallGraphNode* cur_node_;
  CallGraph* call_graph_;
};

IRModule Inline(const IRModule& module) {
  CallGraph cg(module);
  auto topo = cg.TopologicalOrder();
  // Get the reverse topological order of the global functions.
  std::reverse(topo.begin(), topo.end());
  // Cache the functions that are originally entries. This functions will
  // still remain in the module after inlining.
  std::unordered_set<CallGraphNode*> original_entry;

  for (auto* it : topo) {
    if (it->GetRefCount() == 0) original_entry.emplace(it);
    // Skip the leaf calls and the recursive calls that don't call other
    // functions.
    if (it->empty() || (it->IsRecursive() && it->size() == 1)) continue;
    auto base_func = module->Lookup(it->GetNameHint());
    if (const auto* fn = base_func.as<FunctionNode>()) {
      auto func = GetRef<Function>(fn);
      auto new_func = Inliner(it, &cg).Inline(func);
      // TODO(zhiics) Maybe move this to CallGraph, but updating function from
      // CallGraph arbitarily may lead to incorrect CallGraph.
      cg.GetModule()->Update(it->GetGlobalVar(), new_func);
    }
  }

  // Cleanup the functions that are inlined.
  for (auto* cgn : topo) {
    // Skip recursive function and entry functions even if they are marked as
    // `inline`.
    if (cgn->IsRecursive() || original_entry.count(cgn)) continue;
    auto base_func = cg.GetModule()->Lookup(cgn->GetNameHint());
    if (const auto* fn = base_func.as<FunctionNode>()) {
      auto func = GetRef<Function>(fn);
      ObjectRef res = FunctionGetAttr(func, attr::kCompiler);
      const tir::StringImmNode* pval = res.as<tir::StringImmNode>();
      if (pval) {
        CHECK_EQ(cgn->GetRefCount(), 0U)
            << cgn->GetNameHint() << " is marked as inline but not inlined.";
        cgn->CleanCallGraphEntries();
        cg.RemoveGlobalVarFromModule(cgn, /*update_call_graph*/ true);
      }
    }
  }

  return cg.GetModule();
}

namespace transform {

Pass Inline() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
    [=](IRModule m, PassContext pc) {
      return relay::Inline(m);
  };
  return CreateModulePass(pass_func, 1, "Inline", {});
}

TVM_REGISTER_GLOBAL("relay._transform.Inline")
.set_body_typed(Inline);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
