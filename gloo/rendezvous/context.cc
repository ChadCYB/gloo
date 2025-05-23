/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/rendezvous/context.h"

#include <memory>

#include "gloo/common/logging.h"
#include "gloo/rendezvous/store.h"
#include "gloo/transport/address.h"

namespace gloo {
namespace rendezvous {

Context::Context(int rank, int size, int base)
    : ::gloo::Context(rank, size, base) {}

Context::~Context() {}

void Context::connectFullMesh(
    std::shared_ptr<rendezvous::Store> store,
    std::shared_ptr<transport::Device>& dev) {
  auto transportContext = dev->createContext(rank, size);
  transportContext->setTimeout(getTimeout());

  transportContext->createAndConnectAllPairs(std::move(store));

  device_ = dev;
  transportContext_ = std::move(transportContext);
}

ContextFactory::ContextFactory(std::shared_ptr<::gloo::Context> backingContext)
    : backingContext_(backingContext) {
  // We make sure that we have a fully connected context
  for (auto i = 0; i < backingContext_->size; i++) {
    if (i == backingContext_->rank) {
      continue;
    }
    try {
      GLOO_ENFORCE(
          backingContext_->getPair(i) != nullptr,
          "Missing pair in backing context");
    } catch (std::out_of_range&) {
      GLOO_THROW("Backing context not fully connected");
    }
  }

  auto slot = backingContext_->nextSlot();
  auto notificationSlot = backingContext_->nextSlot();

  // Create buffers we'll later use to communicate pair addresses
  recvData_.resize(backingContext_->size);
  sendData_.resize(backingContext_->size);
  recvBuffers_.resize(backingContext_->size);
  sendBuffers_.resize(backingContext_->size);
  recvNotificationData_.resize(backingContext_->size);
  sendNotificationData_.resize(backingContext_->size);
  recvNotificationBuffers_.resize(backingContext_->size);
  sendNotificationBuffers_.resize(backingContext_->size);
  for (auto i = 0; i < backingContext_->size; i++) {
    if (i == backingContext_->rank) {
      continue;
    }

    // Allocate memory for recv/send
    recvData_[i].resize(kMaxAddressSize);
    sendData_[i].resize(kMaxAddressSize);

    // Create pair
    auto& pair = backingContext_->getPair(i);

    // Create payload buffers
    {
      auto recvPtr = recvData_[i].data();
      auto recvSize = recvData_[i].size();
      recvBuffers_[i] = pair->createRecvBuffer(slot, recvPtr, recvSize);
      auto sendPtr = sendData_[i].data();
      auto sendSize = sendData_[i].size();
      sendBuffers_[i] = pair->createSendBuffer(slot, sendPtr, sendSize);
    }

    // Create notification buffers
    {
      auto recvPtr = &recvNotificationData_[i];
      auto recvSize = sizeof(*recvPtr);
      recvNotificationBuffers_[i] =
          pair->createRecvBuffer(notificationSlot, recvPtr, recvSize);
      auto sendPtr = &sendNotificationData_[i];
      auto sendSize = sizeof(*sendPtr);
      sendNotificationBuffers_[i] =
          pair->createSendBuffer(notificationSlot, sendPtr, sendSize);
    }
  }
}

std::shared_ptr<::gloo::Context> ContextFactory::makeContext(
    std::shared_ptr<transport::Device>& dev) {
  auto context =
      std::make_shared<Context>(backingContext_->rank, backingContext_->size);
  context->setTimeout(backingContext_->getTimeout());

  // Assume it's the same for all pairs on a device
  size_t addressSize = 0;

  // Create pairs
  auto transportContext = dev->createContext(context->rank, context->size);
  transportContext->setTimeout(context->getTimeout());
  for (auto i = 0; i < context->size; i++) {
    if (i == context->rank) {
      continue;
    }

    auto& pair = transportContext->createPair(i);
    auto address = pair->address().bytes();
    addressSize = address.size();

    // Send address of new pair to peer
    GLOO_ENFORCE_LE(addressSize, sendData_[i].size());
    sendData_[i].assign(address.begin(), address.end());
    sendBuffers_[i]->send(0, addressSize);
  }

  // Wait for remote addresses and connect peers
  for (auto i = 0; i < context->size; i++) {
    if (i == context->rank) {
      continue;
    }

    recvBuffers_[i]->waitRecv();
    auto& data = recvData_[i];
    auto address = std::vector<char>(data.begin(), data.begin() + addressSize);
    transportContext->getPair(i)->connect(address);

    // Notify peer that we've consumed the payload
    sendNotificationBuffers_[i]->send();
  }

  // Wait for incoming notification from peers
  for (auto i = 0; i < context->size; i++) {
    if (i == context->rank) {
      continue;
    }
    recvNotificationBuffers_[i]->waitRecv();
  }

  // Wait for outgoing notifications to be flushed
  for (auto i = 0; i < context->size; i++) {
    if (i == context->rank) {
      continue;
    }
    sendNotificationBuffers_[i]->waitSend();
  }

  context->device_ = dev;
  context->transportContext_ = std::move(transportContext);
  return std::static_pointer_cast<::gloo::Context>(context);
}

} // namespace rendezvous
} // namespace gloo
