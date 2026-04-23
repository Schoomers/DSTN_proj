#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <cassert>
#include <chrono>
#include <atomic>

#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/slice.h"

// Parse one CSV line from the trace
struct TraceEntry {
    long timestamp;
    std::string key;
    int key_size;
    int value_size;
    int client_id;
    std::string operation;
    int ttl;
};

bool parseLine(const std::string& line, TraceEntry& entry) {
    std::stringstream ss(line);
    std::string token;

    try {
        std::getline(ss, token, ','); entry.timestamp  = std::stol(token);
        std::getline(ss, token, ','); entry.key        = token;
        std::getline(ss, token, ','); entry.key_size   = std::stoi(token);
        std::getline(ss, token, ','); entry.value_size = std::stoi(token);
        std::getline(ss, token, ','); entry.client_id  = std::stoi(token);
        std::getline(ss, token, ','); entry.operation  = token;
        std::getline(ss, token, ','); entry.ttl        = std::stoi(token);
    } catch (...) {
        return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <trace_file> [db_path] [max_ops]\n";
        return 1;
    }

    std::string trace_path = argv[1];
    std::string db_path    = argc > 2 ? argv[2] : "/tmp/rocksdb_replay";
    long max_ops           = argc > 3 ? std::stol(argv[3]) : 1000000;

    // Open RocksDB
    rocksdb::Options options;
    options.create_if_missing = true;
    options.write_buffer_size = 64 * 1024 * 1024;      // 64MB memtable
    options.max_write_buffer_number = 3;
    options.compression = rocksdb::kLZ4Compression;

    std::unique_ptr<rocksdb::DB> db;
    rocksdb::Status s = rocksdb::DB::Open(options, db_path, &db);
    assert(s.ok());
    std::cout << "RocksDB opened at " << db_path << "\n";

    // Open trace file
    std::ifstream trace(trace_path);
    if (!trace.is_open()) {
        std::cerr << "Cannot open trace file: " << trace_path << "\n";
        return 1;
    }

    // Stats
    std::atomic<long> ops{0}, gets{0}, sets{0}, hits{0}, misses{0}, errors{0};
    auto start = std::chrono::steady_clock::now();

    std::string line;
    while (std::getline(trace, line) && ops < max_ops) {
        if (line.empty() || line[0] == '#') continue;

        TraceEntry e;
        if (!parseLine(line, e)) { errors++; continue; }

        // Synthesize a value of the right size for writes
        std::string value(e.value_size, 'x');

        rocksdb::Status status;

        if (e.operation == "get" || e.operation == "gets") {
            std::string result;
            status = db->Get(rocksdb::ReadOptions(), e.key, &result);
            gets++;
            if (status.ok())          hits++;
            else if (status.IsNotFound()) misses++;
            else errors++;

        } else if (e.operation == "set" || e.operation == "add" ||
                   e.operation == "replace" || e.operation == "cas") {
            status = db->Put(rocksdb::WriteOptions(), e.key, value);
            sets++;
            if (!status.ok()) errors++;

        } else if (e.operation == "delete") {
            status = db->Delete(rocksdb::WriteOptions(), e.key);
            if (!status.ok() && !status.IsNotFound()) errors++;
        }
        // incr/decr/append/prepend: treat as get for simplicity
        ops++;

        // Print progress every 100k ops
        if (ops % 100000 == 0) {
            auto now = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(now - start).count();
            std::cout << "ops=" << ops
                      << " gets=" << gets << " sets=" << sets
                      << " hits=" << hits << " misses=" << misses
                      << " hit_ratio=" << (gets > 0 ? 100.0 * hits / gets : 0) << "%"
                      << " throughput=" << (long)(ops / secs) << " ops/s\n";
        }
    }

    // Final stats
    auto end = std::chrono::steady_clock::now();
    double total_secs = std::chrono::duration<double>(end - start).count();

    std::cout << "\n=== FINAL STATS ===\n";
    std::cout << "Total ops:    " << ops     << "\n";
    std::cout << "Gets:         " << gets    << "\n";
    std::cout << "Sets:         " << sets    << "\n";
    std::cout << "Hits:         " << hits    << "\n";
    std::cout << "Misses:       " << misses  << "\n";
    std::cout << "Errors:       " << errors  << "\n";
    std::cout << "Hit ratio:    " << (gets > 0 ? 100.0 * hits / gets : 0) << "%\n";
    std::cout << "Total time:   " << total_secs << "s\n";
    std::cout << "Throughput:   " << (long)(ops / total_secs) << " ops/s\n";

    return 0;
}