/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include "gloo/transport/tcp/address.h"

namespace gloo {
namespace transport {
namespace tcp {

class Error {
 public:
  Error() : valid_(false), msg_("no error") {}
  explicit Error(bool valid) : valid_(valid), msg_("") {}
  explicit Error(const std::string& msg) : valid_(true), msg_(msg) {}
  virtual ~Error() = default;

  Error(const Error&) = delete;
  Error& operator=(const Error&) = delete;

  operator bool() const {
    return valid_;
  }

  virtual std::string what() const;

  static const Error kSuccess;

 protected:
  bool valid_;
  std::string msg_;
};

class SystemError : public Error {
 public:
  SystemError(const char* syscall, int error, const Address& remote = Address())
      : Error(true),
        syscall_(syscall),
        error_(error),
        remote_(remote) {}

  std::string what() const override;

 private:
  const char* syscall_;
  const int error_;
  const Address remote_;
};

class ShortReadError : public Error {
 public:
  ShortReadError(ssize_t expected, ssize_t actual, const Address& remote = Address())
      : Error(true),
        expected_(expected),
        actual_(actual),
        remote_(remote) {}

  std::string what() const override;

 private:
  const ssize_t expected_;
  const ssize_t actual_;
  const Address remote_;
};

class ShortWriteError : public Error {
 public:
  ShortWriteError(ssize_t expected, ssize_t actual, const Address& remote = Address())
      : Error(true),
        expected_(expected),
        actual_(actual),
        remote_(remote) {}

  std::string what() const override;

 private:
  const ssize_t expected_;
  const ssize_t actual_;
  const Address remote_;
};

class TimeoutError : public Error {
 public:
  explicit TimeoutError(const std::string& msg) : Error(msg) {}
};

class LoopError : public Error {
 public:
  explicit LoopError(const std::string& msg) : Error(msg) {}
};

} // namespace tcp
} // namespace transport
} // namespace gloo
