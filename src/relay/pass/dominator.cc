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
 * Copyright (c) 2019 by Contributors
 * \file tvm/relay/pass/dominator.cc
 * \brief The implementation of dominator analysis for a Relay expr.
 *
 * The algorithm works as the following:
 *
 *  // 1. Save the dominator of the entry expression as itself.
 *  dom(entry) = {entry}
 *
 *  // 2. For each expr, save all other exprs as its dominators.
 *  foreach n in AllNodes - {entry}:
 *    dom(n) = AllNodes;
 *
 *  // 3. Clean the dominator set by iteratively eliminating non-dominators.
 *  do
 *    changed = false;
 *    foreach n in AllNodes - {entry}:
 *      old_dom = dom{n}
 *      dom{n} = {n} U {intersection of dom(p) for all p in pred(n)}
 *      changed |= dom(n) neq old_dom
 *    end
 *  while changed
 */
#include "dominator.h"

#include <tvm/relay/expr_functor.h>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dependency_graph.h"

namespace tvm {
namespace relay {
namespace dominator {

class DominatorTree::ArenaCreator {
 public:
  explicit ArenaCreator(common::Arena* arena) : arena_(arena) {}

  struct DominatorHelper : public ExprVisitor {
    int depth = 0;
    int max_depth = 0;
    std::vector<Expr> all_exprs;
    std::unordered_map<Expr, int, NodeHash, NodeEqual> expr_depth;
    std::unordered_set<const tvm::Node*> visited_;

    void Visit(const Expr& expr) {
      Function func = Downcast<Function>(expr);
      CHECK(func.defined());
      visited_.insert(func.get());
      all_exprs.push_back(expr);
      VisitExpr(func->body);
      expr_depth[expr] = max_depth;
    }

    void VisitExpr(const Expr& expr) final {
      if (visited_.count(expr.get()) != 0) return;
      visited_.insert(expr.get());
      expr_depth[expr] = depth++;
      max_depth = depth > max_depth ? depth : max_depth;
      all_exprs.push_back(expr);
      ExprVisitor::VisitExpr(expr);
      depth--;
    }
  };

  DominatorTree Create(const Expr& expr) {
    dg_ = DependencyGraph::Create(arena_, expr);
    for (const auto& it : dg_.expr_node) {
      dep_expr_[it.second] = it.first;
    }

    // Get all expressions and their corresponding depth.
    DominatorHelper helper;
    helper.Visit(expr);
    all_exprs_ = helper.all_exprs;
    expr_depth_ = helper.expr_depth;

    // 1. Add the entry expr
    SetEntryDom(expr);

    // 2. Set all other nodes for each expr except the entry.
    InitDominators(expr);

    // 3. Eliminating.
    EliminateDominators(expr);
    return std::move(dom_);
  }

 private:
  /* \brief Compute the intersection of two expr sets. */
  std::unordered_set<Node*> Intersection(
      const std::unordered_set<DominatorTree::Node*>& lhs,
      const std::unordered_set<DominatorTree::Node*>& rhs) const {
    std::unordered_set<Node*> ret;
    for (auto* it : lhs) {
      if (rhs.count(it)) {
        ret.emplace(it);
      }
    }
    return ret;
  }

  void SetEntryDom(const Expr& expr) {
    dom_.root_ = AllocNode(expr);
    dom_.expr_node[expr] = dom_.root_;
    dom_.expr_node[expr]->parents.emplace(dom_.expr_node[expr]);
  }

  void InitDominators(const Expr& expr) {
    for (const auto& it : all_exprs_) {
      if (!it.same_as(expr)) {
        dom_.expr_node[it] = AllocNode(it);
        for (const auto& n : all_exprs_) {
          dom_.expr_node[n] = AllocNode(n);
          dom_.expr_node[it]->parents.emplace(dom_.expr_node[n]);
        }
      }
    }
  }

  void EliminateDominators(const Expr& expr) {
    using LinkNode = common::LinkNode<DependencyGraph::Node*>;

    bool changed = false;

    do {
      changed = false;
      std::unordered_set<Node*> cur_dom;
      for (const auto& it : all_exprs_) {
        if (it.same_as(expr)) continue;
        CHECK_GT(dom_.expr_node.count(it), 0);
        auto old_dom = dom_.expr_node[it]->parents;

        if (cur_dom.empty()) {
          cur_dom = dom_.expr_node[it]->parents;
        } else {
          // Iterate over the parent nodes
          CHECK_GT(dg_.expr_node.count(it), 0);
          for (LinkNode* p = dg_.expr_node[it]->parents.head; p != nullptr;
               p = p->next) {
            Expr parent_expr = dep_expr_[p->value];
            // Skip the root node.
            if (parent_expr.operator->() == nullptr) continue;
            CHECK_GT(dom_.expr_node.count(parent_expr), 0);
            cur_dom =
                Intersection(cur_dom, dom_.expr_node[parent_expr]->parents);
          }
        }

        // Check if we've updated the dominators of the current expr in the last
        // iteration.
        if (old_dom.size() != cur_dom.size()) {
          changed = true;
        } else {
          for (auto* node : old_dom) {
            if (cur_dom.find(node) == cur_dom.end()) {
              changed = true;
              break;
            }
          }
        }
        // Update the dominators of a node when needed.
        if (changed) dom_.expr_node[it]->parents = cur_dom;
      }
    } while (changed);
  }

  DominatorTree::Node* AllocNode(const Expr& expr) {
    if (dom_.expr_node.count(expr) && dom_.expr_node.at(expr)) {
      return dom_.expr_node.at(expr);
    } else {
      CHECK(expr_depth_.find(expr) != expr_depth_.end());
      DominatorTree::Node* ret = arena_->make<DominatorTree::Node>();
      ret->expr = expr;
      ret->depth = expr_depth_[expr];
      return ret;
    }
  }

  /*! \brief allocator of all the internal node object */
  common::Arena* arena_;

  /*! \brief The created dominator tree. */
  DominatorTree dom_;

  /*! \brief All exprs included by the input Relay expr. */
  std::vector<Expr> all_exprs_;

  /*! \brief The depth of each expr. */
  std::unordered_map<Expr, int, NodeHash, NodeEqual> expr_depth_;

  /*! \brief The dependency graph node to expr mapping. */
  std::unordered_map<DependencyGraph::Node*, Expr> dep_expr_;

  /*! \brief The dependency graph of the expr. */
  DependencyGraph dg_;
};

DominatorTree DominatorTree::Create(common::Arena* arena, const Expr& expr) {
  return ArenaCreator(arena).Create(expr);
}

DominatorTree::Node* DominatorTree::GetRoot() const {
  CHECK(!expr_node.empty());
  for (const auto& it : expr_node) {
    if (it.second->parents.size() == 1) {
      CHECK_EQ(it.second, root_)
        << "There should be one and only one root for a dominator tree";
    }
  }
  return root_;
}

// The set of nodes that dominate the current expr.
std::unordered_set<DominatorTree::Node*> DominatorTree::Dominators(
    const Expr& expr) const {
  CHECK_GT(expr_node.count(expr), 0);
  return expr_node.at(expr)->parents;
}

// Strict dominators of a node n include all nodes that dominator n except
// itself.
std::unordered_set<DominatorTree::Node*> DominatorTree::StrictDominators(
    const Expr& expr) const {
  CHECK_GT(expr_node.count(expr), 0);
  auto ret = expr_node.at(expr)->parents;
  CHECK_GT(ret.count(expr_node.at(expr)), 0);
  ret.erase(expr_node.at(expr));
  return ret;
}

// The immediate dominator of n is the closest node among the strict dominators
// of n.
DominatorTree::Node* DominatorTree::ImmediateDominator(const Expr& expr) const {
  CHECK_GT(expr_node.count(expr), 0);
  Node* imm = nullptr;
  int min_depth = std::numeric_limits<int>::max();
  for (auto* it : expr_node.at(expr)->parents) {
    if (it->depth < min_depth && it != expr_node.at(expr)) {
      min_depth = it->depth;
      imm = it;
    }
  }
  return imm;
}

bool DominatorTree::Dominate(DominatorTree::Node* a,
                             DominatorTree::Node* b) const {
  return b->parents.find(a) != b->parents.end();
}

bool DominatorTree::Dominate(const Expr& a, const Expr& b) const {
  CHECK_GT(expr_node.count(a), 0);
  CHECK_GT(expr_node.count(b), 0);
  return Dominate(expr_node.at(a), expr_node.at(b));
}

}  // namespace dominator
}  // namespace relay
}  // namespace tvm
