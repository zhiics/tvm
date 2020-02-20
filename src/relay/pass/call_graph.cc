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
 * \brief Implementation of APIs to handle the call graph of a Relay module.
 */

#include "call_graph.h"

#include <tvm/relay/expr_functor.h>
#include <algorithm>
#include <vector>
#include <unordered_set>

namespace tvm {
namespace relay {

CallGraph::CallGraph(const IRModule& module) : module_(module) {
  auto gvar_funcs = module->functions;
  for (const auto& it : gvar_funcs) {
    if (const auto* fn = it.second.as<FunctionNode>()) {
      auto func = GetRef<Function>(fn);
      // Add the global function to gradually build up the CallGraph.
      AddToCallGraph(it.first, func);
    }
  }
}

void CallGraph::AddToCallGraph(const GlobalVar& gv, const Function& func) {
  CHECK(func.defined() && gv.defined());
  // Add the current global function as an entry to the CallGraph.
  CallGraphNode* cg_node = LookupGlobalVar(gv);

  // Only GlobalVar nodes need to be handled in a function. It indicates that
  // the global function of a callee is called by the function that is being
  // processed. An edge will be added from the current global function, cg_node,
  // to the node that contains the found callee GlobalVarNode.
  //
  // This is the major overhead for constructing a call graph because the
  // post-order visitor will visit each AST node of the current function to
  // figure out the dependencies between functions.
  PostOrderVisit(func, [&](const Expr& expr) {
    if (const GlobalVarNode* gvn = expr.as<GlobalVarNode>()) {
      auto callee = GetRef<GlobalVar>(gvn);
      cg_node->AddCalledGlobal(LookupGlobalVar(callee));
    }
  });
}

const CallGraphNode* CallGraph::operator[](const GlobalVar& gv) const {
  const_iterator cit = call_graph_.find(gv);
  CHECK(cit != call_graph_.end())
      << "GlobalVar " << gv->name_hint << " not found in the call graph!";
  return cit->second.get();
}

CallGraphNode* CallGraph::operator[](const GlobalVar& gv) {
  const_iterator cit = call_graph_.find(gv);
  CHECK(cit != call_graph_.end())
      << "GlobalVar " << gv->name_hint << " not found in the call graph!";
  return cit->second.get();
}

// Query the existence of a GlobalVar in the CallGraph. It creates an entry if
// there is no such a node available.
CallGraphNode* CallGraph::LookupGlobalVar(const GlobalVar& gv) {
  CHECK(gv.defined());

  // This inserts an element to the call graph if it is not there yet.
  auto& call_graph_node = call_graph_[gv];
  if (call_graph_node) return call_graph_node.get();

  CHECK(module_->ContainGlobalVar(gv->name_hint))
      << "GlobalVar " << gv->name_hint << " not found in the current ir module";

  // Create the node for the inserted entry.
  call_graph_node = std::unique_ptr<CallGraphNode>(new CallGraphNode(gv));
  return call_graph_node.get();
}

void CallGraph::Print(std::ostream& os) const {
  // Print the CallGraph in the topological order.
  std::vector<CallGraphNode*> nodes = TopologicalOrder();
  for (const auto* cgn : nodes) {
    cgn->Print(os);
  }
}

GlobalVar CallGraph::RemoveGlobalVarFromModule(CallGraphNode* cg_node,
                                               bool update_call_graph) {
  CHECK(cg_node->empty() || (cg_node->IsRecursive() && cg_node->size() == 1))
      << "Cannot remove global var " << cg_node->GetNameHint()
      << " from call graph, because it still calls "
      << cg_node->size() << " other global functions";

  if (update_call_graph) {
    // Update the CallGraph by removing all edges that point to the node
    // `cg_node`.
    for (auto& it : *this) {
      it.second->RemoveAllCallTo(cg_node);
    }
  }
  GlobalVar gv = cg_node->GetGlobalVar();
  call_graph_.erase(gv);
  // Update the IR module.
  module_->Remove(gv);
  return gv;
}

std::vector<CallGraphNode*> CallGraph::GetEntryGlobals() const {
  std::vector<CallGraphNode*> ret;
  // An entry function in Relay is a function that never called by other
  // functions or only called by itself.
  for (const auto& it : *this) {
    if (it.second->GetRefCount() == 0 || it.second->IsRecursiveEntry()) {
      ret.push_back(it.second.get());
    }
  }
  return ret;
}

std::vector<CallGraphNode*> CallGraph::TopologicalOrder() const {
  std::vector<CallGraphNode*> ret;
  // Collect all entry nodes.
  std::vector<CallGraphNode*> entries = GetEntryGlobals();
  CallGraphNode::CallGraphNodeSet visited;

  for (const auto& it : entries) {
    // Keep tracking the nodes that have been visited.
    auto topo = it->TopologicalOrder(&visited);
    // Preprend the collected items. The intermeidate nodes that are shared by
    // multiple entries are guaranteed to be collected when visiting the
    // previous entries. Therefore, topological order remains.
    ret.insert(ret.begin(), topo.begin(), topo.end());
  }

  // Find out the missing global functions if there are any to help debugging.
  if (ret.size() != module_->functions.size()) {
    for (auto it : module_->functions) {
      if (visited.find((*this)[it.first]) == visited.end()) {
        LOG(WARNING) << "Missing global:" << it.first->name_hint
                     << " with # refs = " << (*this)[it.first]->GetRefCount();
      }
    }
    LOG(FATAL) << "Expected " << module_->functions.size()
               << " globals, but received "
               << ret.size();
  }

  return ret;
}

// A BSF traverser is used to collect the nodes in a CallGraphNode. The nodes
// that are visited by previous CallGraphNode entries can be memoized. This
// helps us to make sure no entry will be visited multiple times when collecting
// the nodes for an entir call graph.
std::vector<CallGraphNode*> CallGraphNode::TopologicalOrder(
    CallGraphNodeSet* visited) const {
  std::vector<CallGraphNode*> ret;
  std::vector<CallGraphNode*> current_nodes;
  if (visited->find(this) == visited->end()) {
    visited->emplace(this);
    current_nodes.emplace_back(const_cast<CallGraphNode*>(this));
  }

  std::vector<CallGraphNode*> next_nodes;
  while (!current_nodes.empty()) {
    for (const auto& node : current_nodes) {
      ret.push_back(node);
      // Iterate through the called entries.
      for (auto git = node->begin(); git != node->end(); ++git) {
        if (visited->find(git->second) == visited->end()) {
          next_nodes.push_back(git->second);
          visited->emplace(git->second);
        }
      }
    }
    // Update the current level and clean the next level.
    current_nodes = next_nodes;
    next_nodes.clear();
  }
  return ret;
}

void CallGraphNode::CleanCallGraphEntries() {
  while (!called_globals_.empty()) {
    // Decrement the reference counter
    called_globals_.back().second->DecRef();
    called_globals_.pop_back();
  }
}

inline void CallGraphNode::AddCalledGlobal(CallGraphNode* cg_node) {
  called_globals_.emplace_back(global_, cg_node);
  // Increment the reference to indicate that another call site is found for
  // the callee in `cg_node`.
  cg_node->IncRef();
  // Mark the global function as recursive if it calls itself.
  if (global_ == cg_node->GetGlobalVar()) {
    cg_node->is_recursive_ = true;
  }
}

// Remove an edge from the current global function to the callee.
void CallGraphNode::RemoveCallTo(const GlobalVar& callee) {
  for (auto it = begin();; ++it) {
    CHECK(it != end()) << "Cannot find global function "
                       << callee->name_hint << " to remove!";
    if (it->second->GetGlobalVar() == callee) {
      // Only remove one occurrence of the call site.
      it->second->DecRef();
      *it = called_globals_.back();
      called_globals_.pop_back();
      return;
    }
  }
}

// Remove all edges from the current global function to the callee.
void CallGraphNode::RemoveAllCallTo(CallGraphNode* callee) {
  for (uint32_t i = 0, e = size(); i != e;) {
    if (called_globals_[i].second == callee) {
      callee->DecRef();
      called_globals_[i] = called_globals_.back();
      called_globals_.pop_back();
      --e;
    } else {
      ++i;
    }
  }
  // Make sure all references to the callee are removed.
  CHECK_EQ(callee->GetRefCount(), 0U)
      << "All references to " << callee->GetNameHint()
      << " should have been removed";
}

void CallGraphNode::Print(std::ostream& os) const {
  if (!global_.defined()) {
    os << "GlobalVar is not defined\n";
    return;
  }

  os << "Call graph node: " << global_->name_hint;
  os << " at: " << this << ",  #refs = " << GetRefCount() << "\n";

  for (const auto& it : *this) {
    os << "  call site: <" << it.first->name_hint << "> calls ";
    os << it.second->GetNameHint() << "\n";
  }
  os << "\n";
}

std::ostream& operator<<(std::ostream& os, const CallGraph& cg) {
  cg.Print(os);
  return os;
}

std::ostream& operator<<(std::ostream& os, const CallGraphNode& cgn) {
  cgn.Print(os);
  return os;
}

}  // namespace relay
}  // namespace tvm
