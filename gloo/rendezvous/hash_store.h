/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <condition_variable>
#include <mutex>
#include <string>
#include <unordered_map>

#include "gloo/rendezvous/store.h"

namespace gloo {
namespace rendezvous {

class HashStore : public Store {
 public:
  virtual ~HashStore() = default;

  virtual void set(const std::string& key, const std::vector<char>& data) override;

  virtual std::vector<char> get(const std::string& key) override;

  virtual void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout = kDefaultTimeout) override;

 protected:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::unordered_map<std::string, std::vector<char>> map_;
};

} // namespace rendezvous
} // namespace gloo
