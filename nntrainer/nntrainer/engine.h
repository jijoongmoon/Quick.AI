// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   engine.h
 * @date   27 December 2024
 * @brief  This file contains engine context related functions and classes that
 * manages the engines (NPU, GPU, CPU) of the current environment
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __ENGINE_H__
#define __ENGINE_H__

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <context.h>
#include <mem_allocator.h>
#include <nntrainer_error.h>

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
#include <cl_context.h>
#endif

#include "bs_thread_pool_manager.hpp"
#include "singleton.h"

namespace nntrainer {

extern std::mutex engine_mutex;
namespace {} // namespace

/**
 * @class Engine contains user-dependent configuration
 * @brief App
 */
class Engine : public Singleton<Engine> {
public:
  /**
   * @brief Single global Engine instance shared across every DSO.
   *
   * Hides the inherited Singleton<Engine>::Global() — that base
   * version is a template static member function defined inline in
   * the header, so each DSO that includes singleton.h compiles its
   * own copy of `static Engine instance;`. Under a single Linux
   * linker namespace those weak copies dedupe into a single object,
   * but inside Android app classloader-scoped namespaces the dedupe
   * is unreliable and you end up with one Engine per DSO — plugin
   * Contexts registered through libquick_dot_ai_api.so's copy then
   * disappear when libnntrainer.so's createLayerNode looks them up
   * through its own copy.
   *
   * Declared here, defined exactly once in engine.cpp (which lives in
   * libnntrainer.so). Every other DSO links against that single
   * exported symbol, so all callers see the same Engine instance and
   * the same `engines` map.
   */
  static Engine &Global();

protected:
  static const int RegisterContextMax = 16;
  static nntrainer::Context *nntrainerRegisteredContext[RegisterContextMax];
  /// Valgrind complains memory leaks with context registered because
  /// every context is alive during the whole application lifecycle
  /// and we do not free them. It can be amended by using unique_ptr;
  /// however, as we use container and function calls with context,
  /// let's not bother modify all the related functions, but waste
  /// a few words.

  void initialize() noexcept override;

  void add_default_object();

  void registerContext(std::string name, nntrainer::Context *context,
                       void *library_handle = nullptr,
                       DestroyContextFunc destroy_func = nullptr) {
    const std::lock_guard<std::mutex> lock(engine_mutex);
    static int registerCount = 0;

    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (engines.find(name) != engines.end()) {
      auto old_ctx = engines[name];
      auto old_destroy = destroy_funcs.find(name);
      if (old_destroy != destroy_funcs.end() && old_destroy->second) {
        old_destroy->second(old_ctx);
      }
      engines.erase(name);
      allocator.erase(name);
      library_handles.erase(name);
      destroy_funcs.erase(name);
    }
    engines.insert(std::make_pair(name, context));

    if (registerCount < RegisterContextMax) {
      nntrainerRegisteredContext[registerCount] = context;
      registerCount++;
    }

    auto alloc = context->getMemAllocator();

    allocator.insert(std::make_pair(name, alloc));

    // Store library handle and destroy function for plugin contexts
    if (library_handle) {
      library_handles[name] = library_handle;
    }
    if (destroy_func) {
      destroy_funcs[name] = destroy_func;
    }
  }

public:
  /**
   * @brief   Default constructor
   */
  Engine() = default;

  /**
   * @brief   Destructor - releases all contexts
   */
  ~Engine() { release(); }

  /**
   * @brief   Release resources allocated by Engine
   *
   */
  void release();

  /**
   * @brief register a Context from a shared library
   * plugin must have **extern "C" LayerPluggable *ml_train_context_pluggable**
   * defined else error
   *
   * @param library_path a file name of the library
   * @param base_path    base path to make a full path (optional)
   * @throws std::invalid_parameter if library_path is invalid or library is
   * invalid
   */
  int registerContext(const std::string &library_path,
                      const std::string &base_path = "");

  /**
   * @brief get registered a Context
   *
   * @param name Registered Context Name
   * @throws std::invalid_parameter if no context with name
   * @return Context Pointer : for register Object factory, casting might be
   * needed.
   */
  nntrainer::Context *getRegisteredContext(std::string name) const {

    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (engines.find(name) == engines.end()) {
      throw std::invalid_argument("[Engine] " + name +
                                  " Context is not registered");
    }
    return engines.at(name);
  }

  std::unordered_map<std::string, std::shared_ptr<nntrainer::MemAllocator>>
  getAllocators() {
    return allocator;
  }

  /**
   *
   * @brief   Get pointer to thread pool manager, contruct it if needed
   * @return  Pointer to thread pool manager
   */
  ThreadPoolManager *getThreadPoolManager();

  /**
   *
   * @brief Parse compute Engine keywords in properties : eg) engine = cpu
   *  default is "cpu"
   * @return Context name
   */
  std::string parseComputeEngine(const std::vector<std::string> &props) const;

  /**
   * @brief Create an Layer Object with Layer name
   *
   * @param type layer name
   * @param props property
   * @return unitque_ptr<T> unique pointer to the Layer object
   */
  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const std::string &type,
                    const std::vector<std::string> &properties = {}) const {
    auto ct = getRegisteredContext(parseComputeEngine(properties));
    // Same-DSO re-wrap: on Android app classloader namespaces the
    // plugin context's private typeinfo can fail to match the caller
    // DSO's std::exception typeinfo, making a legitimate throw look
    // like a catch(...) "Unknown exception". Rewrap here as a
    // std::runtime_error whose typeinfo is shared via libc++_shared.so.
    try {
      return ct->createLayerObject(type);
    } catch (const std::exception &e) {
      throw std::runtime_error(std::string("Engine::createLayerObject(") +
                               type + ") failed: " + e.what());
    } catch (...) {
      throw std::runtime_error(std::string("Engine::createLayerObject(") +
                               type + ") failed with a non-std exception");
    }
  }

  /**
   * @brief Create an Layer Object with Layer key
   *
   * @param int_key key
   * @param props property
   * @return unitque_ptr<T> unique pointer to the Layer object
   */
  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const int int_key,
                    const std::vector<std::string> &properties = {}) const {
    auto ct = getRegisteredContext(parseComputeEngine(properties));
    try {
      return ct->createLayerObject(int_key);
    } catch (const std::exception &e) {
      throw std::runtime_error(std::string("Engine::createLayerObject(int_key=") +
                               std::to_string(int_key) + ") failed: " +
                               e.what());
    } catch (...) {
      throw std::runtime_error(std::string("Engine::createLayerObject(int_key=") +
                               std::to_string(int_key) +
                               ") failed with a non-std exception");
    }
  }

  /**
   * @brief Create an Optimizer Object with Optimizer name
   *
   * @param type Optimizer name
   * @param props property
   * @return unitque_ptr<T> unique pointer to the Optimizer object
   */
  std::unique_ptr<nntrainer::Optimizer>
  createOptimizerObject(const std::string &type,
                        const std::vector<std::string> &properties = {}) const {
    auto ct = getRegisteredContext(parseComputeEngine(properties));
    return ct->createOptimizerObject(type);
  }

  /**
   * @brief Create an Optimizer Object with Optimizer key
   *
   * @param int_key key
   * @param props property
   * @return unitque_ptr<T> unique pointer to the Optimizer object
   */
  std::unique_ptr<nntrainer::Optimizer>
  createOptimizerObject(const int int_key,
                        const std::vector<std::string> &properties = {}) const {
    auto ct = getRegisteredContext(parseComputeEngine(properties));
    return ct->createOptimizerObject(int_key);
  }

  /**
   * @brief Create an LearningRateScheduler Object with type
   *
   * @param type type of LearningRateScheduler
   * @param props property
   * @return unitque_ptr<T> unique pointer to the LearningRateScheduler object
   */
  std::unique_ptr<ml::train::LearningRateScheduler>
  createLearningRateSchedulerObject(
    const std::string &type,
    const std::vector<std::string> &properties = {}) const {
    auto ct = getRegisteredContext(parseComputeEngine(properties));
    return ct->createLearningRateSchedulerObject(type, properties);
  }

  /**
   * @brief Create an LearningRateScheduler Object with key
   *
   * @param int_key key
   * @param props property
   * @return unitque_ptr<T> unique pointer to the LearningRateScheduler object
   */
  std::unique_ptr<ml::train::LearningRateScheduler>
  createLearningRateSchedulerObject(
    const int int_key, const std::vector<std::string> &properties = {}) {
    auto ct = getRegisteredContext(parseComputeEngine(properties));
    return ct->createLearningRateSchedulerObject(int_key, properties);
  }

  /**
   * @brief Get Working Path from a relative or representation of a path
   * starting from @a working_path_base.
   * @param[in] path to make full path
   * @return If absolute path is given, returns @a path
   * If relative path is given and working_path_base is not set, return
   * relative path.
   * If relative path is given and working_path_base has set, return absolute
   * path from current working directory
   */
  const std::string getWorkingPath(const std::string &path = "") const;

  /**
   * @brief Set Working Directory for a relative path. working directory is set
   * canonically
   * @param[in] base base directory
   * @throw std::invalid_argument if path is not valid for current system
   */
  void setWorkingDirectory(const std::string &base);

  /**
   * @brief unset working directory
   *
   */
  void unsetWorkingDirectory() { working_path_base = ""; }

  /**
   * @brief query if the appcontext has working directory set
   *
   * @retval true working path base is set
   * @retval false working path base is not set
   */
  bool hasWorkingDirectory() { return !working_path_base.empty(); }

private:
  /**
   * @brief map for Context and Context name
   *
   */
  std::unordered_map<std::string, nntrainer::Context *> engines;

  /**
   * @brief map for library handles (context name -> library handle)
   * These handles must be kept alive until contexts are destroyed
   */
  std::unordered_map<std::string, void *> library_handles;

  /**
   * @brief map for destroy functions (context name -> destroy function)
   * Used to properly destroy contexts created by plugins
   */
  std::unordered_map<std::string, DestroyContextFunc> destroy_funcs;

  std::unordered_map<std::string, std::shared_ptr<nntrainer::MemAllocator>>
    allocator;

  std::string working_path_base;

  std::mutex thread_pool_manager_mutex_ = {};
  std::unique_ptr<ThreadPoolManager> thread_pool_manager_ = {};
};

namespace plugin {}

} // namespace nntrainer

#endif /* __ENGINE_H__ */
