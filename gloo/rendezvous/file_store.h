/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "gloo/rendezvous/store.h"

namespace gloo {
namespace rendezvous {

class FileStore : public Store {
 public:
  explicit FileStore(const std::string& path);

  virtual ~FileStore() = default;

  virtual void set(const std::string& key, const std::vector<char>& data) override;

  virtual std::vector<char> get(const std::string& key) override;

  virtual void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout = kDefaultTimeout) override;

 protected:
  std::string basePath_;

  std::string realPath(const std::string& path);
};

} // namespace rendezvous
} // namespace gloo
