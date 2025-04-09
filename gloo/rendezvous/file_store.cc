/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/rendezvous/file_store.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <fstream>
#include <system_error>
#include <thread>

namespace gloo {
namespace rendezvous {

FileStore::FileStore(const std::string& path) : basePath_(path) {
  if (mkdir(path.c_str(), 0777) != 0 && errno != EEXIST) {
    throw std::system_error(errno, std::system_category());
  }
}

void FileStore::set(const std::string& key, const std::vector<char>& data) {
  auto path = realPath(key);
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::system_error(errno, std::system_category());
  }
  file.write(data.data(), data.size());
}

std::vector<char> FileStore::get(const std::string& key) {
  auto path = realPath(key);
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::system_error(errno, std::system_category());
  }
  
  auto size = file.tellg();
  std::vector<char> data(size);
  file.seekg(0);
  file.read(data.data(), size);
  return data;
}

void FileStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  const auto start = std::chrono::steady_clock::now();
  while (true) {
    bool ready = true;
    for (const auto& key : keys) {
      auto path = realPath(key);
      struct stat st;
      if (stat(path.c_str(), &st) != 0) {
        if (errno == ENOENT) {
          ready = false;
          break;
        }
        throw std::system_error(errno, std::system_category());
      }
    }
    if (ready) {
      break;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    if (elapsed > timeout) {
      throw std::runtime_error("Timeout waiting for keys");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

std::string FileStore::realPath(const std::string& path) {
  return basePath_ + "/" + path;
}

} // namespace rendezvous
} // namespace gloo
