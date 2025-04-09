/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/rendezvous/prefix_store.h"

namespace gloo {
namespace rendezvous {

PrefixStore::PrefixStore(const std::string& prefix, Store& store)
    : prefix_(prefix), store_(store) {}

void PrefixStore::set(const std::string& key, const std::vector<char>& data) {
  store_.set(prefix_ + key, data);
}

std::vector<char> PrefixStore::get(const std::string& key) {
  return store_.get(prefix_ + key);
}

void PrefixStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  std::vector<std::string> prefixedKeys;
  for (const auto& key : keys) {
    prefixedKeys.push_back(prefix_ + key);
  }
  store_.wait(prefixedKeys, timeout);
}

bool PrefixStore::has_v2_support() {
  return store_.has_v2_support();
}

std::vector<std::vector<char>> PrefixStore::multi_get(
    const std::vector<std::string>& keys) {
  if (!store_.has_v2_support()) {
    return Store::multi_get(keys);
  }
  std::vector<std::string> prefixed_keys;
  for (const auto& key : keys) {
    prefixed_keys.push_back(prefix_ + key);
  }
  return store_.multi_get(prefixed_keys);
}

void PrefixStore::multi_set(
    const std::vector<std::string>& keys,
    const std::vector<std::vector<char>>& values) {
  if (!store_.has_v2_support()) {
    Store::multi_set(keys, values);
    return;
  }
  std::vector<std::string> prefixed_keys;
  for (const auto& key : keys) {
    prefixed_keys.push_back(prefix_ + key);
  }
  store_.multi_set(prefixed_keys, values);
}

void PrefixStore::append(
    const std::string& key,
    const std::vector<char>& data) {
  if (!store_.has_v2_support()) {
    Store::append(key, data);
    return;
  }
  store_.append(prefix_ + key, data);
}

int64_t PrefixStore::add(const std::string& key, int64_t value) {
  if (!store_.has_v2_support()) {
    return Store::add(key, value);
  }
  return store_.add(prefix_ + key, value);
}

} // namespace rendezvous
} // namespace gloo
