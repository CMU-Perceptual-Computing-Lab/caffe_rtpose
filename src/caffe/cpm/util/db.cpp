#include "caffe/cpm/util/db.hpp"
#include "caffe/util/db_leveldb.hpp"
#include "caffe/util/db_lmdb.hpp"

namespace caffe { namespace db {

DB* GetDB(CPMDataParameter::DB backend) {
  switch (backend) {
#ifdef USE_LEVELDB
  case CPMDataParameter_DB_LEVELDB:
    return new LevelDB();
#endif  // USE_LEVELDB
#ifdef USE_LMDB
  case CPMDataParameter_DB_LMDB:
    return new LMDB();
#endif  // USE_LMDB
  default:
    LOG(FATAL) << "Unknown database backend";
    return NULL;
  }
}

DB* GetDB(CocoDataParameter::DB backend) {
  switch (backend) {
#ifdef USE_LEVELDB
  case CocoDataParameter_DB_LEVELDB:
    return new LevelDB();
#endif  // USE_LEVELDB
#ifdef USE_LMDB
  case CocoDataParameter_DB_LMDB:
    return new LMDB();
#endif  // USE_LMDB
  default:
    LOG(FATAL) << "Unknown database backend";
    return NULL;
  }
}

DB* GetDB(CPMBottomUpDataParameter::DB backend) {
  switch (backend) {
#ifdef USE_LEVELDB
  case CocoDataParameter_DB_LEVELDB:
    return new LevelDB();
#endif  // USE_LEVELDB
#ifdef USE_LMDB
  case CocoDataParameter_DB_LMDB:
    return new LMDB();
#endif  // USE_LMDB
  default:
    LOG(FATAL) << "Unknown database backend";
    return NULL;
  }
}

}  // namespace db
}  // namespace caffe
