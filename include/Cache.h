// Implements an LRU cache

/* This file has been written and/or modified by the following people:
 *
 * Alex Schutz
 *
 */

#pragma once

#include <iostream>
#include <list>
#include <unordered_map>

namespace DetMCVI {

template <typename T1, typename T2, typename T1Hash = std::hash<T1>,
          typename T1Equal = std::equal_to<T1>>
class LRUCache {
 private:
  size_t capacity;
  mutable std::unordered_map<
      T1, std::pair<T2, typename std::list<T1>::iterator>, T1Hash, T1Equal>
      cacheMap;
  mutable std::list<T1> lruList;

 public:
  LRUCache(size_t capacity) : capacity(capacity) {}

  void put(const T1& key, const T2& value) {
    auto it = cacheMap.find(key);
    if (it != cacheMap.end()) {
      // Key exists, update value and move it to the front of the cache list
      it->second.first = value;
      lruList.erase(it->second.second);
      lruList.push_front(key);
      it->second.second = lruList.begin();
    } else {
      // Key doesn't exist
      if (cacheMap.size() >= capacity) {
        // Cache is full, remove the least recently used element
        T1 lruKey = lruList.back();
        lruList.pop_back();
        cacheMap.erase(lruKey);
      }
      // Add the new key-value pair to the cache
      lruList.push_front(key);
      cacheMap.emplace(key, std::make_pair(value, lruList.begin()));
    }
  }

  bool contains(const T1& key) const {
    return cacheMap.find(key) != cacheMap.end();
  }

  const T2& at(const T1& key) const {
    auto it = cacheMap.find(key);
    if (it != cacheMap.end()) {
      // Key exists, move it to the front of the cache list and return the value
      const T2& value = it->second.first;
      lruList.erase(it->second.second);
      lruList.push_front(key);
      it->second.second = lruList.begin();
      return value;
    }
    throw std::out_of_range("Key not found in cache.");
  }

  T2& operator[](const T1& key) {
    auto it = cacheMap.find(key);
    if (it != cacheMap.end()) {
      // Key exists, move it to the front of the cache list and return the value
      lruList.erase(it->second.second);
      lruList.push_front(key);
      it->second.second = lruList.begin();
      return it->second.first;
    } else {
      // Key doesn't exist, insert default value and return reference to it
      if (cacheMap.size() >= capacity) {
        // Cache is full, remove the least recently used element
        T1 lruKey = lruList.back();
        lruList.pop_back();
        cacheMap.erase(lruKey);
      }
      // Add the new key with a default value
      lruList.push_front(key);
      auto result =
          cacheMap.emplace(key, std::make_pair(T2(), lruList.begin()));
      return result.first->second.first;
    }
  }

  typename std::unordered_map<T1,
                              std::pair<T2, typename std::list<T1>::iterator>,
                              T1Hash, T1Equal>::const_iterator
  find(const T1& key) const {
    return cacheMap.find(key);
  }

  typename std::unordered_map<T1,
                              std::pair<T2, typename std::list<T1>::iterator>,
                              T1Hash, T1Equal>::iterator
  find(const T1& key) {
    return cacheMap.find(key);
  }

  typename std::unordered_map<T1,
                              std::pair<T2, typename std::list<T1>::iterator>,
                              T1Hash, T1Equal>::iterator
  begin() {
    return cacheMap.begin();
  }

  typename std::unordered_map<T1,
                              std::pair<T2, typename std::list<T1>::iterator>,
                              T1Hash, T1Equal>::const_iterator
  cbegin() const {
    return cacheMap.cbegin();
  }

  typename std::unordered_map<T1,
                              std::pair<T2, typename std::list<T1>::iterator>,
                              T1Hash, T1Equal>::iterator
  end() {
    return cacheMap.end();
  }

  typename std::unordered_map<T1,
                              std::pair<T2, typename std::list<T1>::iterator>,
                              T1Hash, T1Equal>::const_iterator
  cend() const {
    return cacheMap.cend();
  }
};

}  // namespace DetMCVI
