/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/rendezvous/hash_store.h"

#include <chrono>
#include <stdexcept>

namespace gloo {
namespace rendezvous {

void HashStore::set(const std::string& key, const std::vector<char>& data) {
  std::lock_guard<std::mutex> lock(mutex_);
  map_[key] = data;
  cv_.notify_all();
}

std::vector<char> HashStore::get(const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = map_.find(key);
  if (it == map_.end()) {
    throw std::runtime_error("Key not found: " + key);
  }
  return it->second;
}

void HashStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  auto deadline = std::chrono::steady_clock::now() + timeout;
  std::unique_lock<std::mutex> lock(mutex_);
  while (true) {
    bool allFound = true;
    for (const auto& key : keys) {
      if (map_.find(key) == map_.end()) {
        allFound = false;
        break;
      }
    }
    if (allFound) {
      return;
    }
    if (timeout != kNoTimeout) {
      if (cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
        throw std::runtime_error("Wait timeout");
      }
    } else {
      cv_.wait(lock);
    }
  }
}

} // namespace rendezvous
} // namespace gloo
