/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include "gloo/common/error.h"
#include "gloo/common/store.h"

// can be used by upstream users to know whether this is available or not.
#define GLOO_STORE_HAS_STORE_V2 1

namespace gloo {
namespace rendezvous {

class Store : public IStore {
 public:
  static constexpr std::chrono::milliseconds kDefaultTimeout =
      std::chrono::milliseconds(30000);

  virtual ~Store() = default;

  virtual void set(const std::string& key, const std::vector<char>& data) override = 0;

  virtual std::vector<char> get(const std::string& key) override = 0;

  virtual void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout = kDefaultTimeout) override = 0;

  // Extended 2.0 API support
  virtual bool has_v2_support() override { return false; }

  virtual std::vector<std::vector<char>> multi_get(
      const std::vector<std::string>& keys) override {
    std::vector<std::vector<char>> result;
    for (const auto& key : keys) {
      result.push_back(get(key));
    }
    return result;
  }

  virtual void multi_set(
      const std::vector<std::string>& keys,
      const std::vector<std::vector<char>>& values) override {
    for (size_t i = 0; i < keys.size(); i++) {
      set(keys[i], values[i]);
    }
  }

  virtual void append(
      const std::string& key,
      const std::vector<char>& value) override {
    auto existing = get(key);
    existing.insert(existing.end(), value.begin(), value.end());
    set(key, existing);
  }

  virtual int64_t add(const std::string& key, int64_t value) override {
    auto data = get(key);
    int64_t current = 0;
    if (data.size() == sizeof(int64_t)) {
      memcpy(&current, data.data(), sizeof(int64_t));
    }
    current += value;
    std::vector<char> newData(sizeof(int64_t));
    memcpy(newData.data(), &current, sizeof(int64_t));
    set(key, newData);
    return current;
  }
};

} // namespace rendezvous
} // namespace gloo
