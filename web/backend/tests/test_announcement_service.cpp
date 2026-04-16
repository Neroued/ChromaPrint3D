#include <gtest/gtest.h>

#include "application/announcement_service.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace chromaprint3d::backend {
namespace {

using json       = nlohmann::json;
using clock_type = std::chrono::system_clock;

// Use a fresh temporary directory per test instance so PersistLocked() never
// pollutes either CHROMAPRINT3D_TEST_DATA_DIR or another test's state.
class TempDataDir {
public:
    TempDataDir() {
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ =
            std::filesystem::temp_directory_path() /
            ("chromaprint3d_announce_" + std::to_string(now) + "_" + std::to_string(counter_++));
        std::filesystem::create_directories(path_);
    }

    ~TempDataDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    TempDataDir(const TempDataDir&)            = delete;
    TempDataDir& operator=(const TempDataDir&) = delete;

    const std::filesystem::path& path() const { return path_; }

private:
    static inline std::atomic<int> counter_{0};
    std::filesystem::path path_;
};

std::string IsoShift(clock_type::time_point base, std::chrono::seconds delta) {
    return AnnouncementService::FormatIso8601Utc(base + delta);
}

json MakeBaseAnnouncement(const std::string& id, const std::string& ends_at) {
    return json{
        {"id", id},
        {"type", "info"},
        {"severity", "info"},
        {"title", {{"zh", "中文标题"}, {"en", "English Title"}}},
        {"body", {{"zh", "中文正文"}, {"en", "English body"}}},
        {"ends_at", ends_at},
    };
}

} // namespace

TEST(AnnouncementServiceValidation, RejectsInvalidIdFormat) {
    TempDataDir dir;
    AnnouncementService svc(dir.path());

    const auto ends_at = IsoShift(clock_type::now(), std::chrono::hours(1));
    auto body          = MakeBaseAnnouncement("bad id with space", ends_at);

    const auto res = svc.Upsert(body);
    EXPECT_FALSE(res.ok);
    EXPECT_EQ(res.status_code, 400);
    EXPECT_EQ(res.code, "invalid_announcement");
}

TEST(AnnouncementServiceValidation, RejectsUnknownType) {
    TempDataDir dir;
    AnnouncementService svc(dir.path());

    const auto ends_at = IsoShift(clock_type::now(), std::chrono::hours(1));
    auto body          = MakeBaseAnnouncement("hello", ends_at);
    body["type"]       = "promotion";

    const auto res = svc.Upsert(body);
    EXPECT_FALSE(res.ok);
    EXPECT_EQ(res.status_code, 400);
}

TEST(AnnouncementServiceValidation, RejectsEmptyBilingualTitle) {
    TempDataDir dir;
    AnnouncementService svc(dir.path());

    const auto ends_at = IsoShift(clock_type::now(), std::chrono::hours(1));
    auto body          = MakeBaseAnnouncement("t1", ends_at);
    body["title"]      = {{"zh", nullptr}, {"en", ""}};

    const auto res = svc.Upsert(body);
    EXPECT_FALSE(res.ok);
}

TEST(AnnouncementServiceValidation, RejectsPastEndsAt) {
    TempDataDir dir;
    AnnouncementService svc(dir.path());

    const auto past = IsoShift(clock_type::now(), std::chrono::seconds(-10));
    auto body       = MakeBaseAnnouncement("past", past);

    const auto res = svc.Upsert(body);
    EXPECT_FALSE(res.ok);
}

TEST(AnnouncementServiceValidation, RejectsStartsAtAfterEndsAt) {
    TempDataDir dir;
    AnnouncementService svc(dir.path());

    const auto now    = clock_type::now();
    auto body         = MakeBaseAnnouncement("swap", IsoShift(now, std::chrono::hours(1)));
    body["starts_at"] = IsoShift(now, std::chrono::hours(2));

    const auto res = svc.Upsert(body);
    EXPECT_FALSE(res.ok);
}

TEST(AnnouncementServiceValidation, RejectsScheduledUpdateOutsideWindow) {
    TempDataDir dir;
    AnnouncementService svc(dir.path());

    const auto now    = clock_type::now();
    auto body         = MakeBaseAnnouncement("sched", IsoShift(now, std::chrono::hours(2)));
    body["starts_at"] = IsoShift(now, std::chrono::minutes(10));
    body["scheduled_update_at"] = IsoShift(now, std::chrono::minutes(5));

    const auto res = svc.Upsert(body);
    EXPECT_FALSE(res.ok);
}

TEST(AnnouncementServiceWindow, ListActiveRespectsStartsAndEnds) {
    TempDataDir dir;
    AnnouncementService svc(dir.path());

    const auto now = clock_type::now();
    {
        auto body         = MakeBaseAnnouncement("a", IsoShift(now, std::chrono::hours(1)));
        body["starts_at"] = IsoShift(now, std::chrono::hours(-1));
        const auto r      = svc.Upsert(body);
        ASSERT_TRUE(r.ok) << r.message;
    }
    {
        auto body         = MakeBaseAnnouncement("b", IsoShift(now, std::chrono::hours(3)));
        body["starts_at"] = IsoShift(now, std::chrono::hours(2));
        const auto r      = svc.Upsert(body);
        ASSERT_TRUE(r.ok) << r.message;
    }

    const auto visible_now = svc.ListActive(now);
    ASSERT_EQ(visible_now.size(), 1u);
    EXPECT_EQ(visible_now.front().id, "a");

    const auto later         = now + std::chrono::hours(2) + std::chrono::minutes(5);
    const auto visible_later = svc.ListActive(later);
    ASSERT_EQ(visible_later.size(), 1u);
    EXPECT_EQ(visible_later.front().id, "b");

    const auto past         = now + std::chrono::hours(4);
    const auto visible_past = svc.ListActive(past);
    EXPECT_TRUE(visible_past.empty());
}

TEST(AnnouncementServiceWindow, ListActiveSortsBySeverityThenUpdatedAt) {
    TempDataDir dir;
    AnnouncementService svc(dir.path());

    const auto now = clock_type::now();
    for (const auto& [id, sev] : std::vector<std::pair<std::string, std::string>>{
             {"i1", "info"}, {"w1", "warning"}, {"e1", "error"}}) {
        auto body        = MakeBaseAnnouncement(id, IsoShift(now, std::chrono::hours(1)));
        body["severity"] = sev;
        ASSERT_TRUE(svc.Upsert(body).ok);
    }

    const auto list = svc.ListActive(now);
    ASSERT_EQ(list.size(), 3u);
    EXPECT_EQ(list[0].severity, "error");
    EXPECT_EQ(list[1].severity, "warning");
    EXPECT_EQ(list[2].severity, "info");
}

TEST(AnnouncementServicePersistence, UpsertWritesAtomicFileAndReloads) {
    TempDataDir dir;
    const auto ends_at = IsoShift(clock_type::now(), std::chrono::hours(2));

    {
        AnnouncementService svc(dir.path());
        const auto res = svc.Upsert(MakeBaseAnnouncement("persist", ends_at));
        ASSERT_TRUE(res.ok) << res.message;
    }

    const auto file = dir.path() / "announcements.json";
    ASSERT_TRUE(std::filesystem::exists(file));

    std::ifstream in(file);
    ASSERT_TRUE(in.is_open());
    const std::string content((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("\"persist\""), std::string::npos);

    // Reload should find the same entry.
    {
        AnnouncementService svc(dir.path());
        const auto list = svc.ListActive();
        ASSERT_EQ(list.size(), 1u);
        EXPECT_EQ(list.front().id, "persist");
    }
}

TEST(AnnouncementServicePersistence, RemoveDeletesFromFile) {
    TempDataDir dir;
    AnnouncementService svc(dir.path());

    const auto ends_at = IsoShift(clock_type::now(), std::chrono::hours(2));
    ASSERT_TRUE(svc.Upsert(MakeBaseAnnouncement("gone", ends_at)).ok);

    const auto remove = svc.Remove("gone");
    EXPECT_TRUE(remove.ok);

    const auto after = svc.Remove("gone");
    EXPECT_FALSE(after.ok);
    EXPECT_EQ(after.status_code, 404);

    EXPECT_TRUE(svc.ListActive().empty());
}

TEST(AnnouncementServicePersistence, IgnoresCorruptJsonOnLoad) {
    TempDataDir dir;
    const auto file = dir.path() / "announcements.json";

    {
        std::ofstream out(file);
        out << "{ not valid json";
    }

    AnnouncementService svc(dir.path());
    EXPECT_TRUE(svc.ListActive().empty());
}

TEST(AnnouncementServicePersistence, SkipsEntriesFailingSchemaOnLoad) {
    TempDataDir dir;
    const auto file = dir.path() / "announcements.json";

    const auto ends_at = IsoShift(clock_type::now(), std::chrono::hours(1));
    json good          = MakeBaseAnnouncement("good", ends_at);
    good["created_at"] = AnnouncementService::FormatIso8601Utc(clock_type::now());
    good["updated_at"] = good["created_at"];

    json bad = MakeBaseAnnouncement("bad space", ends_at);
    json arr = json::array();
    arr.push_back(good);
    arr.push_back(bad);

    {
        std::ofstream out(file);
        out << arr.dump(2);
    }

    AnnouncementService svc(dir.path());
    const auto list = svc.ListActive();
    ASSERT_EQ(list.size(), 1u);
    EXPECT_EQ(list.front().id, "good");
}

TEST(AnnouncementServiceVersionHash, ChangesOnContentDiffAndResetsOnEmpty) {
    TempDataDir dir;
    AnnouncementService svc(dir.path());

    const auto ends_at = IsoShift(clock_type::now(), std::chrono::hours(1));

    const auto initial_hash = svc.VersionHash();
    EXPECT_EQ(initial_hash.size(), 8u);

    ASSERT_TRUE(svc.Upsert(MakeBaseAnnouncement("h1", ends_at)).ok);
    const auto hash_after_first = svc.VersionHash();
    EXPECT_NE(hash_after_first, initial_hash);

    // Upserting a *different* announcement must change the hash.
    auto other        = MakeBaseAnnouncement("h1", ends_at);
    other["severity"] = "warning";
    ASSERT_TRUE(svc.Upsert(other).ok);
    // Hash must change because severity differs even if updated_at landed
    // in the same 1-second window.
    const auto hash_after_edit = svc.VersionHash();
    EXPECT_NE(hash_after_edit, hash_after_first);

    ASSERT_TRUE(svc.Remove("h1").ok);
    const auto hash_empty = svc.VersionHash();
    EXPECT_EQ(hash_empty, initial_hash);
}

TEST(AnnouncementServiceVersionHash, IdempotentWithinSameSecond) {
    TempDataDir dir;
    AnnouncementService svc(dir.path());

    const auto ends_at = IsoShift(clock_type::now(), std::chrono::hours(1));

    ASSERT_TRUE(svc.Upsert(MakeBaseAnnouncement("same", ends_at)).ok);
    const auto h1 = svc.VersionHash();
    ASSERT_TRUE(svc.Upsert(MakeBaseAnnouncement("same", ends_at)).ok);
    const auto h2 = svc.VersionHash();
    EXPECT_EQ(h1, h2);
}

TEST(AnnouncementServiceVersionHash, ChangesAcrossSecondBoundaryOnReupsert) {
    TempDataDir dir;
    AnnouncementService svc(dir.path());

    const auto ends_at = IsoShift(clock_type::now(), std::chrono::hours(1));
    ASSERT_TRUE(svc.Upsert(MakeBaseAnnouncement("tick", ends_at)).ok);
    const auto h1 = svc.VersionHash();

    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    ASSERT_TRUE(svc.Upsert(MakeBaseAnnouncement("tick", ends_at)).ok);
    const auto h2 = svc.VersionHash();
    EXPECT_NE(h1, h2);
}

TEST(AnnouncementServiceVersionHash, StableAcrossReload) {
    TempDataDir dir;
    const auto ends_at = IsoShift(clock_type::now(), std::chrono::hours(1));
    std::string h1;
    {
        AnnouncementService svc(dir.path());
        ASSERT_TRUE(svc.Upsert(MakeBaseAnnouncement("stable", ends_at)).ok);
        h1 = svc.VersionHash();
    }
    {
        AnnouncementService svc(dir.path());
        EXPECT_EQ(svc.VersionHash(), h1);
    }
}

TEST(AnnouncementServiceUpsert, PreservesCreatedAtOnUpdate) {
    TempDataDir dir;
    AnnouncementService svc(dir.path());

    const auto ends_at = IsoShift(clock_type::now(), std::chrono::hours(2));
    const auto first   = svc.Upsert(MakeBaseAnnouncement("keep", ends_at));
    ASSERT_TRUE(first.ok);
    const auto created_first = first.data.at("announcement").at("created_at").get<std::string>();

    // Small sleep so updated_at differs from created_at
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));

    auto body         = MakeBaseAnnouncement("keep", ends_at);
    body["severity"]  = "warning";
    const auto second = svc.Upsert(body);
    ASSERT_TRUE(second.ok);
    const auto created_second = second.data.at("announcement").at("created_at").get<std::string>();
    const auto updated_second = second.data.at("announcement").at("updated_at").get<std::string>();

    EXPECT_EQ(created_first, created_second);
    EXPECT_NE(updated_second, created_second);
}

TEST(AnnouncementServiceRemove, RejectsInvalidId) {
    TempDataDir dir;
    AnnouncementService svc(dir.path());

    const auto res = svc.Remove("bad id");
    EXPECT_FALSE(res.ok);
    EXPECT_EQ(res.status_code, 400);
}

TEST(AnnouncementServiceIdValidator, AcceptsAndRejectsExpectedForms) {
    EXPECT_TRUE(AnnouncementService::IsValidId("maint-2026-04-18"));
    EXPECT_TRUE(AnnouncementService::IsValidId("A_B_c_1"));
    EXPECT_FALSE(AnnouncementService::IsValidId(""));
    EXPECT_FALSE(AnnouncementService::IsValidId("has space"));
    EXPECT_FALSE(AnnouncementService::IsValidId("with.dot"));
    EXPECT_FALSE(AnnouncementService::IsValidId(std::string(65, 'a')));
}

TEST(AnnouncementServiceIsoValidator, AcceptsAndRejectsExpectedForms) {
    EXPECT_TRUE(AnnouncementService::IsValidIso8601Utc("2026-04-17T03:30:00Z"));
    EXPECT_TRUE(AnnouncementService::IsValidIso8601Utc("2026-04-17T03:30:00.123Z"));
    EXPECT_FALSE(AnnouncementService::IsValidIso8601Utc("2026-04-17 03:30:00Z"));
    EXPECT_FALSE(AnnouncementService::IsValidIso8601Utc("2026-04-17T03:30:00+08:00"));
    EXPECT_FALSE(AnnouncementService::IsValidIso8601Utc(""));
}

// Mirrors the docker-compose + --announcements-dir layout where the
// announcements directory is a separate writable volume mounted outside
// the main data_dir. The service must auto-create the target directory
// on startup and write announcements.json into it.
TEST(AnnouncementServicePersistence, CreatesMissingDirectoryOnStartup) {
    TempDataDir parent;
    const auto target = parent.path() / "nested" / "announcements";
    ASSERT_FALSE(std::filesystem::exists(target));

    AnnouncementService svc(target);
    EXPECT_TRUE(std::filesystem::is_directory(target));

    const auto ends_at = IsoShift(clock_type::now(), std::chrono::hours(1));
    const auto res     = svc.Upsert(MakeBaseAnnouncement("nested", ends_at));
    ASSERT_TRUE(res.ok) << res.message;
    EXPECT_TRUE(std::filesystem::exists(target / "announcements.json"));
}

// Simulates the production layout where announcements_dir is completely
// independent of data_dir. Data persisted via one service instance must
// reload from the same directory on a fresh instance.
TEST(AnnouncementServicePersistence, StandaloneDirectoryReloadsAcrossInstances) {
    TempDataDir dir;
    const auto standalone = dir.path() / "announcements_only";
    const auto ends_at    = IsoShift(clock_type::now(), std::chrono::hours(1));

    {
        AnnouncementService svc(standalone);
        ASSERT_TRUE(svc.Upsert(MakeBaseAnnouncement("iso1", ends_at)).ok);
    }
    {
        AnnouncementService svc(standalone);
        const auto list = svc.ListActive();
        ASSERT_EQ(list.size(), 1u);
        EXPECT_EQ(list.front().id, "iso1");
    }
}

} // namespace chromaprint3d::backend
