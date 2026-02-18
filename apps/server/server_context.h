#pragma once

#include "color_db_cache.h"
#include "session.h"
#include "board_cache.h"
#include "board_geometry_cache.h"
#include "task_manager.h"

#include "chromaprint3d/model_package.h"

#include <httplib/httplib.h>

using namespace ChromaPrint3D;

struct ServerContext {
    httplib::Server& server;
    TaskManager& task_mgr;
    ColorDBCache& db_cache;
    const ModelPackage* model_pack;
    SessionManager& session_mgr;
    BoardCache& board_cache;
    BoardGeometryCache& geometry_cache;
};
