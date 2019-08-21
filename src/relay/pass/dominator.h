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
 *  Copyright (c) 2019 by Contributors.
 * \file tvm/relay/pass/dominator.h
 * \brief Create dominator analysis.
 *
 * https://en.wikipedia.org/wiki/Dominator_(graph_theory)
 */
#ifndef TVM_RELAY_PASS_DOMINATOR_H_
#define TVM_RELAY_PASS_DOMINATOR_H_

#include <tvm/relay/expr.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../common/arena.h"

namespace tvm {
namespace relay {
namespace dominator {

/*!
 * \brief DominatorTree defines the dominator relationships between Relay exprs.
 */
class DominatorTree {
 public:
  /*! \brief A node in the graph. */
  struct Node {
    // The corresponding relay expr.
    Expr expr;
    // The depth of the expr node.
    int depth{0};
    // The nodes that are dominating the current node.
    std::unordered_set<Node*> parents;
  };

  /*!
   * \brief Get the root of the dominator tree.
   */
  Node* GetRoot() const;

  /*!
   * \brief Get the dominators of an expr.
   *
   * \param expr. The Relay expression.
   *
   * \return Dominators of an expression.
   */
  std::unordered_set<Node*> Dominators(const Expr& expr) const;

  /*!
   * \brief Get the immediate dominator of an expr.
   *
   * \param expr. The Relay expression.
   *
   * \return The immediate dominator.
   */
  Node* ImmediateDominator(const Expr& expr) const;

  /*!
   * \brief Get the strict dominators of an expr.
   *
   * \param expr. The Relay expression.
   *
   * \return The strict dominators.
   */
  std::unordered_set<Node*> StrictDominators(const Expr& expr) const;

  /*!
   * \brief Check if the dominator tree node a dominates the other node b.
   *
   * \param a The dominator node.
   * \param b The dominated node.
   *
   * \return true if a dominates b, otherwise false.
   */
  bool Dominate(Node* a, Node* b) const;

  /*!
   * \brief Check if an expr a dominates the other expr b.
   *
   * \param a The dominator expr.
   * \param b The dominated expr.
   *
   * \return true if a dominates b, otherwise false.
   */
  bool Dominate(const Expr& a, const Expr& b) const;

  /*!
   * \brief Create a dominator graph.
   *
   * \param arena The arena used for internal data allocation.
   * \param expr The expr used for dominator analysis.
   */
  static DominatorTree Create(common::Arena* arena, const Expr& expr);

 private:
  /*! \brief Maps a Relay Expr to the corresponding node in the dominator graph. */
  std::unordered_map<Expr, Node*, NodeHash, NodeEqual> expr_node;

  /*! \brief The root of a dominator tree. */
  Node* root_;

  class ArenaCreator;
  friend class ArenaCreator;
};

}  // namespace dominator
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PASS_DOMINATOR_H_
