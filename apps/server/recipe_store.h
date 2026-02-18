#pragma once

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

struct BoardRecipeSet {
    int board_index = 0;
    int grid_rows   = 0;
    int grid_cols   = 0;
    std::vector<std::vector<uint8_t>> recipes;
};

struct EightColorRecipeStore {
    bool loaded = false;
    int base_channel_idx  = 0;
    int base_layers       = 10;
    float layer_height_mm = 0.08f;
    float line_width_mm   = 0.42f;
    int num_channels      = 8;
    int color_layers      = 5;
    std::string layer_order;
    std::vector<BoardRecipeSet> boards; // indexed 0 = board 1, 1 = board 2

    const BoardRecipeSet* FindBoard(int board_index) const {
        for (const auto& b : boards) {
            if (b.board_index == board_index) { return &b; }
        }
        return nullptr;
    }

    static EightColorRecipeStore LoadFromFile(const std::string& path) {
        std::ifstream in(path);
        if (!in.is_open()) { throw std::runtime_error("Cannot open recipe file: " + path); }
        nlohmann::json j;
        in >> j;
        return FromJson(j);
    }

    static EightColorRecipeStore FromJson(const nlohmann::json& j) {
        EightColorRecipeStore store;
        store.loaded           = true;
        store.base_channel_idx = j.value("base_channel_idx", 0);
        store.base_layers      = j.value("base_layers", 10);
        store.layer_height_mm  = j.value("layer_height_mm", 0.08f);
        store.line_width_mm    = j.value("line_width_mm", 0.42f);
        store.layer_order      = j.value("layer_order", "Top2Bottom");

        if (j.contains("meta")) {
            const auto& m       = j.at("meta");
            store.num_channels   = m.value("num_channels", 8);
            store.color_layers   = m.value("color_layers", 5);
        }

        if (!j.contains("boards") || !j["boards"].is_array()) {
            throw std::runtime_error("Recipe JSON missing 'boards' array");
        }
        for (const auto& bj : j["boards"]) {
            BoardRecipeSet brs;
            brs.board_index = bj.value("board_index", 0);
            brs.grid_rows   = bj.value("grid_rows", 0);
            brs.grid_cols   = bj.value("grid_cols", 0);
            if (!bj.contains("recipes") || !bj["recipes"].is_array()) {
                throw std::runtime_error("Board entry missing 'recipes' array");
            }
            for (const auto& rj : bj["recipes"]) {
                std::vector<uint8_t> recipe;
                for (const auto& v : rj) { recipe.push_back(static_cast<uint8_t>(v.get<int>())); }
                brs.recipes.push_back(std::move(recipe));
            }
            store.boards.push_back(std::move(brs));
        }

        spdlog::info("Loaded 8-color recipe store: {} boards", store.boards.size());
        for (const auto& b : store.boards) {
            spdlog::info("  Board {}: {}x{}, {} recipes",
                         b.board_index, b.grid_rows, b.grid_cols, b.recipes.size());
        }
        return store;
    }
};
