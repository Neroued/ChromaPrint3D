#include <gtest/gtest.h>

#include "application/server_facade.h"
#include "chromaprint3d/calib.h"

#include <filesystem>
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
    ServerFacade facade(MakeTestConfig());

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

} // namespace chromaprint3d::backend
