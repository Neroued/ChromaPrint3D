#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

struct ServerOptions {
    int port         = 8080;
    std::string host = "0.0.0.0";
    std::string data_dir;        // ColorDB directory (required)
    std::string model_pack_path; // Optional model package
    std::string web_dir;         // Optional static files directory
    std::string recipes_dir;     // Optional pre-computed recipes directory
    int max_upload_mb     = 50;
    int max_tasks         = 4;
    int task_ttl_seconds  = 3600;
    std::string log_level = "info";
};

inline void PrintUsage(const char* exe) {
    std::printf(
        "Usage: %s --data <dir> [options]\n"
        "Options:\n"
        "  --port PORT          HTTP port (default: 8080)\n"
        "  --host HOST          Bind address (default: 0.0.0.0)\n"
        "  --data DIR           ColorDB data directory (required)\n"
        "  --model-pack PATH    Model package JSON (optional)\n"
        "  --web DIR            Static web files directory (optional)\n"
        "  --recipes-dir DIR    Pre-computed recipes directory (optional)\n"
        "  --max-upload-mb N    Max upload size in MB (default: 50)\n"
        "  --max-tasks N        Max concurrent tasks (default: 4)\n"
        "  --task-ttl N         Task TTL in seconds (default: 3600)\n"
        "  --log-level LEVEL    Log level: trace/debug/info/warn/error/off (default: info)\n",
        exe);
}

inline bool ParseArgs(int argc, char** argv, ServerOptions& opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--port") && i + 1 < argc) {
            opts.port = std::stoi(argv[++i]);
        } else if ((arg == "--host") && i + 1 < argc) {
            opts.host = argv[++i];
        } else if ((arg == "--data") && i + 1 < argc) {
            opts.data_dir = argv[++i];
        } else if ((arg == "--model-pack") && i + 1 < argc) {
            opts.model_pack_path = argv[++i];
        } else if ((arg == "--web") && i + 1 < argc) {
            opts.web_dir = argv[++i];
        } else if ((arg == "--recipes-dir") && i + 1 < argc) {
            opts.recipes_dir = argv[++i];
        } else if ((arg == "--max-upload-mb") && i + 1 < argc) {
            opts.max_upload_mb = std::stoi(argv[++i]);
        } else if ((arg == "--max-tasks") && i + 1 < argc) {
            opts.max_tasks = std::stoi(argv[++i]);
        } else if ((arg == "--task-ttl") && i + 1 < argc) {
            opts.task_ttl_seconds = std::stoi(argv[++i]);
        } else if ((arg == "--log-level") && i + 1 < argc) {
            opts.log_level = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return false;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            PrintUsage(argv[0]);
            return false;
        }
    }
    if (opts.data_dir.empty()) {
        std::fprintf(stderr, "Error: --data is required\n");
        PrintUsage(argv[0]);
        return false;
    }
    return true;
}
