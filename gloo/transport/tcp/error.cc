/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/transport/tcp/error.h"

#include <cstring>
#include <sstream>

namespace gloo {
namespace transport {
namespace tcp {

// Define static instance instead of copying
const Error Error::kSuccess;

std::string Error::what() const {
  return msg_;
}

std::string SystemError::what() const {
  std::stringstream ss;
  ss << syscall_ << ": " << strerror(error_);
  if (!remote_.str().empty()) {
    ss << " (peer: " << remote_.str() << ")";
  }
  return ss.str();
}

std::string ShortReadError::what() const {
  std::stringstream ss;
  ss << "short read (got " << actual_ << " of " << expected_ << " bytes)";
  if (!remote_.str().empty()) {
    ss << " (peer: " << remote_.str() << ")";
  }
  return ss.str();
}

std::string ShortWriteError::what() const {
  std::stringstream ss;
  ss << "short write (got " << actual_ << " of " << expected_ << " bytes)";
  if (!remote_.str().empty()) {
    ss << " (peer: " << remote_.str() << ")";
  }
  return ss.str();
}

} // namespace tcp
} // namespace transport
} // namespace gloo
