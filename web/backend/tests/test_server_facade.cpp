#include <gtest/gtest.h>

#include "application/server_facade.h"
#include "chromaprint3d/calib.h"

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

namespace chromaprint3d::backend {
namespace {

using json = nlohmann::json;

ServerConfig MakeTestConfig() {
    ServerConfig cfg;
    cfg.data_dir            = CHROMAPRINT3D_TEST_DATA_DIR;
    cfg.max_tasks           = 1;
    cfg.max_task_queue      = 8;
    cfg.max_tasks_per_owner = 4;
    cfg.max_task_result_mb  = 256;
    return cfg;
}

ServerFacade& SharedFacade() {
    static auto* facade = new ServerFacade(MakeTestConfig());
    return *facade;
}

std::string ReadFileText(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open test file: " + path.string());
    }
    return std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
}

json MakeGenerateBoardRequest() {
    return {
        {"palette", json::array({
                        {
                            {"color", "White"},
                            {"material", "PLA Basic"},
                        },
                        {
                            {"color", "Black"},
                            {"material", "PLA Basic"},
                        },
                    })},
    };
}

} // namespace

TEST(ServerFacadeTest, DownloadBoardMetaReturnsInlineJsonArtifact) {
    auto& facade = SharedFacade();

    const auto generate = facade.GenerateBoard(MakeGenerateBoardRequest());
    ASSERT_TRUE(generate.ok) << generate.message;
    ASSERT_TRUE(generate.data.contains("board_id"));
    ASSERT_TRUE(generate.data.contains("meta"));

    TaskArtifact artifact;
    const auto download =
        facade.DownloadBoardMeta(generate.data.at("board_id").get<std::string>(), artifact);
    ASSERT_TRUE(download.ok) << download.message;

    EXPECT_FALSE(artifact.is_file_based());
    EXPECT_TRUE(artifact.file_path.empty());
    EXPECT_EQ(artifact.content_type, "application/json");
    EXPECT_FALSE(artifact.filename.empty());
    EXPECT_EQ(std::filesystem::path(artifact.filename).extension(), ".json");
    ASSERT_FALSE(artifact.bytes.empty());

    const std::string body(artifact.bytes.begin(), artifact.bytes.end());
    const json meta_json = json::parse(body);
    EXPECT_EQ(meta_json, generate.data.at("meta"));

    const auto meta = ChromaPrint3D::CalibrationBoardMeta::FromJsonString(body);
    EXPECT_EQ(meta.config.recipe.num_channels, 2);
    ASSERT_EQ(meta.config.palette.size(), 2u);
    EXPECT_EQ(meta.config.palette[0].color, "White");
    EXPECT_EQ(meta.config.palette[1].color, "Black");
}

TEST(ServerFacadeTest, HealthColordbPoolTracksSessionDatabaseBytes) {
    auto& facade = SharedFacade();

    bool created     = false;
    const auto token = facade.EnsureSession(std::nullopt, &created);
    ASSERT_TRUE(created);
    ASSERT_FALSE(token.empty());

    const auto baseline_health = facade.Health();
    ASSERT_TRUE(baseline_health.ok) << baseline_health.message;
    ASSERT_TRUE(baseline_health.data.contains("memory"));
    const auto baseline_pool =
        baseline_health.data.at("memory").at("colordb_pool_bytes").get<std::size_t>();
    EXPECT_FALSE(baseline_health.data.at("memory").at("allocator").get<std::string>().empty());

    const auto db_json = ReadFileText(std::filesystem::path(CHROMAPRINT3D_TEST_DATA_DIR) / "dbs" /
                                      "PLA" / "BambuLab" / "RYBW_008_5L.json");

    constexpr auto kSessionDbName = "session_health_test_db";
    const auto upload =
        facade.UploadSessionDatabase(token, db_json, std::optional<std::string>{kSessionDbName});
    ASSERT_TRUE(upload.ok) << upload.message;

    const auto uploaded_health = facade.Health();
    ASSERT_TRUE(uploaded_health.ok) << uploaded_health.message;
    const auto uploaded_pool =
        uploaded_health.data.at("memory").at("colordb_pool_bytes").get<std::size_t>();
    EXPECT_GT(uploaded_pool, baseline_pool);

    const auto remove = facade.DeleteSessionDatabase(token, kSessionDbName);
    ASSERT_TRUE(remove.ok) << remove.message;

    const auto restored_health = facade.Health();
    ASSERT_TRUE(restored_health.ok) << restored_health.message;
    const auto restored_pool =
        restored_health.data.at("memory").at("colordb_pool_bytes").get<std::size_t>();
    EXPECT_EQ(restored_pool, baseline_pool);
}

} // namespace chromaprint3d::backend
