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

using namespace tvm::runtime;

namespace tvm {
namespace relay {

/*!
 * \brief Check if a function is a recursive function. We can simply check if
 * a function contains a call node that calls to itself. If so, this function is
 * recursive. We don't need to check the arguments and return type because Relay
 * does not allow function overloading.
 */
class EligibilityChecker : public ExprVisitor {
 public:
  explicit EligibilityChecker(const std::string& fn_name) : fn_name_(fn_name) {}

  bool CanInline(const Function& func) {
    // Ensure the function is not recursive.
    ExprVisitor::VisitExpr(func);

    // The body of a global functions must be defined.
    is_recursive_ = is_recursive_ && func->body.defined();

    // The function must be annotated with the compiler attribute.
    ObjectRef res = FunctionGetAttr(func, attr::kCompiler);
    const tir::StringImmNode* pval = res.as<tir::StringImmNode>();
    is_recursive_ = is_recursive_ && (pval != nullptr);

    return is_recursive_;
  }

  void VisitExpr_(const CallNode* call_node) final {
    if (auto global = call_node->op.as<GlobalVarNode>()) {
      if (global->name_hint == fn_name_) {
        is_recursive_ = true;
      }
    }
    ExprVisitor::VisitExpr_(call_node);
  }

 private:
  bool is_recursive_{false};
  std::string fn_name_;
};

class Inliner : ExprMutator {
 public:
  explicit Inliner(const IRModule& module) : module_(module) {}

  Expr VisitExpr_(const CallNode* call_node) {
    Expr op = call_node->op;
    const auto* global = op.as<GlobalVarNode>();

    if (global && inline_candidates_.count(global->name_hint)) {
      tvm::Array<Expr> call_args;
      for (auto arg : call_node->args) {
        auto new_arg = VisitExpr(arg);
        call_args.push_back(new_arg);
      }
      auto func = MakeLocalFunction(GetRef<GlobalVar>(global));
      return CallNode::make(func, call_args, call_node->attrs, call_node->type_args);
    } else {
      return ExprMutator::VisitExpr(GetRef<Call>(call_node));
    }
  }

  Function MakeLocalFunction(const GlobalVar& global) {
    auto base_func = module_->Lookup(global);
    const auto* func = base_func.as<FunctionNode>();
    CHECK(func);
    return FunctionNode::make(func->params,
                              func->body,
                              func->ret_type,
                              func->type_params,
                              func->attrs);
  }

  IRModule Inline() {
    auto gvar_funcs = module_->functions;
    for (auto pair : gvar_funcs) {
      const std::string& func_name = pair.first->name_hint;
      EligibilityChecker checker(func_name);
      Function func = Downcast<Function>(pair.second);
      if (checker.CanInline(func)) {
        inline_candidates_.insert(func_name);
      }
    }

    for (auto pair : gvar_funcs) {
      auto global = pair.first;
      auto base_func = pair.second;
      if (auto* n = base_func.as<FunctionNode>()) {
        auto func = GetRef<Function>(n);

        func = FunctionNode::make(func->params,
                                  VisitExpr(func->body),
                                  func->ret_type,
                                  func->type_params,
                                  func->attrs);
        module_->Add(global, func, true);

      }
    }
    return module_;
  }

 private:
  IRModule module_;
  std::unordered_map<Var, Expr, ObjectHash, ObjectEqual> var_map;
  std::unordered_set<std::string> inline_candidates_;
};

namespace transform {

Pass Inline() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
    [=](IRModule m, PassContext pc) {
      return relay::Inliner(m).Inline();
  };
  auto inline_pass = CreateModulePass(pass_func, 1, "Inline", {});
  // Eliminate the unused functions after inlining.
  Array<tvm::PrimExpr> entry_funcs{tir::StringImmNode::make("main")};
  return Sequential({inline_pass, RemoveUnusedFunctions(entry_funcs)}, "Inline");
}

TVM_REGISTER_GLOBAL("relay._transform.Inline")
.set_body_typed(Inline);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
