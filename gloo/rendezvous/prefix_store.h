/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

#include "gloo/rendezvous/store.h"

namespace gloo {
namespace rendezvous {

class PrefixStore : public Store {
 public:
  PrefixStore(const std::string& prefix, Store& store);

  virtual ~PrefixStore() = default;

  virtual void set(const std::string& key, const std::vector<char>& data) override;

  virtual std::vector<char> get(const std::string& key) override;

  virtual void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout = kDefaultTimeout) override;

  virtual bool has_v2_support() override;

  virtual std::vector<std::vector<char>> multi_get(
      const std::vector<std::string>& keys) override;

  virtual void multi_set(
      const std::vector<std::string>& keys,
      const std::vector<std::vector<char>>& values) override;

  virtual void append(
      const std::string& key,
      const std::vector<char>& data) override;

  virtual int64_t add(const std::string& key, int64_t value) override;

 protected:
  const std::string prefix_;
  Store& store_;

  std::string joinKey(const std::string& key);
};

} // namespace rendezvous
} // namespace gloo
